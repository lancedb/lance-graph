// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion-based physical planner for graph queries
//!
//! This module translates graph logical plans into DataFusion logical plans for execution.
//! It implements a two-phase planning approach:
//!
//! ## Phase 1: Analysis
//! - Extracts metadata from the graph logical plan (from `logical_plan.rs`)
//! - Assigns unique IDs to relationship instances to avoid column name conflicts
//! - Collects variable-to-label mappings and required datasets
//!
//! ## Phase 2: Plan Building
//! - Converts graph operations to relational operations
//! - Nodes as Tables: Each node label becomes a table scan
//! - Relationships as Tables: Each relationship type becomes a linking table
//! - Graph traversals become SQL joins with qualified column names
//!
//! ## Key Design Decisions
//! - **Unique relationship aliases**: Each relationship expansion gets a unique alias
//!   (e.g., `knows_1`, `knows_2`) to support multi-hop queries without column conflicts
//! - **Relationship variables**: User-specified variables (e.g., `[r:KNOWS]`) take precedence
//! - **Column qualification**: All columns are qualified as `{variable}__{column}` to avoid ambiguity

use crate::ast::RelationshipDirection;
use crate::error::Result;
use crate::logical_plan::*;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::{
    col, lit, BinaryExpr, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Planner abstraction for graph-to-physical planning
pub trait GraphPhysicalPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan>;
}

/// DataFusion-based physical planner
pub struct DataFusionPlanner {
    config: crate::config::GraphConfig,
    catalog: Option<Arc<dyn GraphSourceCatalog>>,
}

impl DataFusionPlanner {
    pub fn new(config: crate::config::GraphConfig) -> Self {
        Self {
            config,
            catalog: None,
        }
    }

    pub fn with_catalog(
        config: crate::config::GraphConfig,
        catalog: Arc<dyn GraphSourceCatalog>,
    ) -> Self {
        Self {
            config,
            catalog: Some(catalog),
        }
    }
}

// ============================================================================
// Query Analysis Phase
// ============================================================================

/// Analysis result containing all metadata needed for planning
#[derive(Debug, Clone, Default)]
pub struct QueryAnalysis {
    /// Variable → Label mappings (e.g., "n" → "Person")
    pub var_to_label: HashMap<String, String>,

    /// Relationship instances with unique IDs to avoid column conflicts
    pub relationship_instances: Vec<RelationshipInstance>,

    /// All datasets required for this query
    pub required_datasets: HashSet<String>,
}

/// Represents a single relationship expansion with a unique instance ID
#[derive(Debug, Clone)]
pub struct RelationshipInstance {
    pub id: usize, // Unique instance number
    pub rel_type: String,
    pub source_var: String,
    pub target_var: String,
    pub direction: RelationshipDirection,
    pub alias: String, // e.g., "friend_of_1", "friend_of_2"
}

/// Parameters for joining source node to relationship
struct SourceJoinParams<'a> {
    source_variable: &'a str,
    rel_qualifier: &'a str,
    node_id_field: &'a str,
    rel_map: &'a crate::config::RelationshipMapping,
    direction: &'a RelationshipDirection,
}

/// Parameters for joining relationship to target node
struct TargetJoinParams<'a> {
    source_variable: &'a str,
    target_variable: &'a str,
    rel_qualifier: &'a str,
    node_map: &'a crate::config::NodeMapping,
    rel_map: &'a crate::config::RelationshipMapping,
    direction: &'a RelationshipDirection,
}

/// Planning context that tracks state during plan building
pub struct PlanningContext<'a> {
    pub analysis: &'a QueryAnalysis,
    relationship_instance_idx: HashMap<String, usize>,
}

impl<'a> PlanningContext<'a> {
    pub fn new(analysis: &'a QueryAnalysis) -> Self {
        Self {
            analysis,
            relationship_instance_idx: HashMap::new(),
        }
    }

    /// Get the next relationship instance for a given type (returns a clone)
    pub fn next_relationship_instance(&mut self, rel_type: &str) -> Result<RelationshipInstance> {
        let idx = self
            .relationship_instance_idx
            .entry(rel_type.to_string())
            .and_modify(|i| *i += 1)
            .or_insert(0);

        self.analysis
            .relationship_instances
            .iter()
            .filter(|r| r.rel_type == rel_type)
            .nth(*idx)
            .cloned()
            .ok_or_else(|| crate::error::GraphError::PlanError {
                message: format!("No relationship instance found for: {}", rel_type),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }
}

impl GraphPhysicalPlanner for DataFusionPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        // Phase 1: Analyze query structure
        let analysis = self.analyze(logical_plan)?;

        // Phase 2: Build execution plan with context
        let mut ctx = PlanningContext::new(&analysis);
        self.build_operator(&mut ctx, logical_plan)
    }
}

impl DataFusionPlanner {
    /// Enhanced planning with dynamic table registration to solve "table not found" issues
    pub fn plan_with_context(
        &self,
        logical_plan: &LogicalOperator,
        datasets: &std::collections::HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<LogicalPlan> {
        use crate::source_catalog::{InMemoryCatalog, SimpleTableSource};
        use std::sync::Arc;

        // Use the new analyze() method to extract metadata
        let analysis = self.analyze(logical_plan)?;

        // Build an in-memory catalog from provided datasets (nodes and relationships)
        let mut catalog = InMemoryCatalog::new();

        // Register node sources from required datasets
        for label in &analysis.required_datasets {
            if self.config.node_mappings.contains_key(label) {
                if let Some(batch) = datasets.get(label) {
                    let src = Arc::new(SimpleTableSource::new(batch.schema()));
                    catalog = catalog.with_node_source(label, src);
                }
            }
        }

        // Register relationship sources from required datasets
        for rel_type in &analysis.required_datasets {
            if self.config.relationship_mappings.contains_key(rel_type) {
                if let Some(batch) = datasets.get(rel_type) {
                    let src = Arc::new(SimpleTableSource::new(batch.schema()));
                    catalog = catalog.with_relationship_source(rel_type, src);
                }
            }
        }

        // Plan using a planner bound to this catalog so scans get qualified projections
        let planner_with_cat =
            DataFusionPlanner::with_catalog(self.config.clone(), Arc::new(catalog));
        planner_with_cat.plan(logical_plan)
    }

    /// Phase 1: Analyze the logical plan to extract metadata
    fn analyze(&self, logical_plan: &LogicalOperator) -> Result<QueryAnalysis> {
        let mut analysis = QueryAnalysis::default();
        let mut rel_counter: HashMap<String, usize> = HashMap::new();

        analyze_operator(logical_plan, &mut analysis, &mut rel_counter)?;
        Ok(analysis)
    }
}

/// Recursively analyze operators to build QueryAnalysis
fn analyze_operator(
    op: &LogicalOperator,
    analysis: &mut QueryAnalysis,
    rel_counter: &mut HashMap<String, usize>,
) -> Result<()> {
    match op {
        LogicalOperator::ScanByLabel {
            variable, label, ..
        } => {
            analysis
                .var_to_label
                .insert(variable.clone(), label.clone());
            analysis.required_datasets.insert(label.clone());
        }
        LogicalOperator::Expand {
            input,
            source_variable,
            target_variable,
            relationship_types,
            direction,
            relationship_variable,
            ..
        }
        | LogicalOperator::VariableLengthExpand {
            input,
            source_variable,
            target_variable,
            relationship_types,
            direction,
            relationship_variable,
            ..
        } => {
            // Recursively analyze input first
            analyze_operator(input, analysis, rel_counter)?;

            // Infer target variable's label from source variable
            // For (a:Person)-[:KNOWS]->(b), b also gets label Person
            if let Some(source_label) = analysis.var_to_label.get(source_variable).cloned() {
                analysis
                    .var_to_label
                    .insert(target_variable.clone(), source_label);
            }

            // Assign unique instance ID for this relationship
            if let Some(rel_type) = relationship_types.first() {
                let instance_id = rel_counter
                    .entry(rel_type.clone())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);

                // Use relationship variable if provided, otherwise use type_instanceId
                let alias = if let Some(rel_var) = relationship_variable {
                    rel_var.clone()
                } else {
                    format!("{}_{}", rel_type.to_lowercase(), instance_id)
                };

                analysis.relationship_instances.push(RelationshipInstance {
                    id: *instance_id,
                    rel_type: rel_type.clone(),
                    source_var: source_variable.clone(),
                    target_var: target_variable.clone(),
                    direction: direction.clone(),
                    alias,
                });

                analysis.required_datasets.insert(rel_type.clone());
            }
        }
        LogicalOperator::Filter { input, .. }
        | LogicalOperator::Project { input, .. }
        | LogicalOperator::Sort { input, .. }
        | LogicalOperator::Limit { input, .. }
        | LogicalOperator::Offset { input, .. }
        | LogicalOperator::Distinct { input } => {
            analyze_operator(input, analysis, rel_counter)?;
        }
        LogicalOperator::Join { left, right, .. } => {
            analyze_operator(left, analysis, rel_counter)?;
            analyze_operator(right, analysis, rel_counter)?;
        }
    }
    Ok(())
}

impl DataFusionPlanner {
    /// Phase 2: Build DataFusion LogicalPlan from logical operator with context
    fn build_operator(
        &self,
        ctx: &mut PlanningContext,
        op: &LogicalOperator,
    ) -> Result<LogicalPlan> {
        match op {
            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
                ..
            } => self.build_scan(ctx, variable, label, properties),
            LogicalOperator::Filter { input, predicate } => {
                let input_plan = self.build_operator(ctx, input)?;
                let expr = self.to_df_boolean_expr(predicate);
                Ok(LogicalPlanBuilder::from(input_plan)
                    .filter(expr)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Project { input, projections } => {
                let input_plan = self.build_operator(ctx, input)?;
                let exprs: Vec<Expr> = projections
                    .iter()
                    .map(|p| {
                        let expr = self.to_df_value_expr(&p.expression);
                        // Apply alias if provided
                        if let Some(alias) = &p.alias {
                            expr.alias(alias)
                        } else {
                            expr
                        }
                    })
                    .collect();
                Ok(LogicalPlanBuilder::from(input_plan)
                    .project(exprs)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Distinct { input } => {
                let input_plan = self.build_operator(ctx, input)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .distinct()
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Sort { input, sort_items } => {
                use datafusion::logical_expr::SortExpr;

                let input_plan = self.build_operator(ctx, input)?;

                // Convert sort items to DataFusion sort expressions
                let sort_exprs: Vec<SortExpr> = sort_items
                    .iter()
                    .map(|item| {
                        let expr = self.to_df_value_expr(&item.expression);
                        let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
                        SortExpr {
                            expr,
                            asc,
                            nulls_first: true,
                        }
                    })
                    .collect();

                Ok(LogicalPlanBuilder::from(input_plan)
                    .sort(sort_exprs)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Limit { input, count } => {
                let input_plan = self.build_operator(ctx, input)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .limit(0, Some((*count) as usize))
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Offset { input, offset } => {
                let input_plan = self.build_operator(ctx, input)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .limit((*offset) as usize, None)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                ..
            }
            | LogicalOperator::VariableLengthExpand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                ..
            } => self.build_expand(
                ctx,
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
            ),
            LogicalOperator::Join { left, .. } => {
                // Not yet implemented: explicit join. For now, use left branch
                self.build_operator(ctx, left)
            }
        }
    }

    // ============================================================================
    // Component Builders
    // ============================================================================

    /// Build a qualified node scan with property filters and column aliasing
    fn build_scan(
        &self,
        _ctx: &PlanningContext,
        variable: &str,
        label: &str,
        properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        // Try to use catalog if available
        if let Some(cat) = &self.catalog {
            if let Some(source) = cat.node_source(label) {
                // Get schema before moving source
                let schema = source.schema();
                let mut builder = LogicalPlanBuilder::scan(label, source, None).unwrap();

                // Apply property filters using unqualified names (before aliasing)
                for (k, v) in properties.iter() {
                    let lit_expr =
                        self.to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
                    let filter_expr = Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(col(k)),
                        op: Operator::Eq,
                        right: Box::new(lit_expr),
                    });
                    builder = builder.filter(filter_expr).unwrap();
                }

                // Create qualified column aliases: variable__property
                let qualified_exprs: Vec<Expr> = schema
                    .fields()
                    .iter()
                    .map(|field| {
                        let qualified_name = format!("{}__{}", variable, field.name());
                        col(field.name()).alias(&qualified_name)
                    })
                    .collect();

                // Add projection with qualified aliases
                builder = builder.project(qualified_exprs).unwrap();

                return Ok(builder.build().unwrap());
            }
        }

        // Fallback: create a simple table reference that DataFusion can resolve at execution time
        let empty_source = Arc::new(crate::source_catalog::SimpleTableSource::empty());
        let builder = LogicalPlanBuilder::scan(label, empty_source, None).map_err(|e| {
            crate::error::GraphError::PlanError {
                message: format!("Failed to create table scan for {}: {}", label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        builder
            .build()
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to build table scan for {}: {}", label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Build a relationship expansion (graph traversal) as a series of joins
    fn build_expand(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
    ) -> Result<LogicalPlan> {
        let left_plan = self.build_operator(ctx, input)?;

        // Get the unique relationship instance for this expand operation
        let Some(cat) = &self.catalog else {
            // Fallback: pass-through if catalog not available
            return Ok(left_plan);
        };

        let Some(rel_type) = relationship_types.first() else {
            return Ok(left_plan);
        };

        let rel_instance = ctx.next_relationship_instance(rel_type)?;
        let Some(rel_map) = self.config.relationship_mappings.get(rel_type) else {
            return Ok(left_plan);
        };

        let Some(src_label) = ctx.analysis.var_to_label.get(source_variable) else {
            return Ok(left_plan);
        };

        let Some(node_map) = self.config.node_mappings.get(src_label) else {
            return Ok(left_plan);
        };

        let Some(rel_source) = cat.relationship_source(&rel_map.relationship_type) else {
            return Ok(left_plan);
        };

        // Build relationship scan with qualified columns
        let rel_scan = self.build_relationship_scan(&rel_instance, rel_source)?;

        // Join source node with relationship
        let source_params = SourceJoinParams {
            source_variable,
            rel_qualifier: &rel_instance.alias,
            node_id_field: &node_map.id_field,
            rel_map,
            direction,
        };
        let builder = self.join_source_to_relationship(left_plan, rel_scan, &source_params)?;

        // Join relationship with target node
        let target_params = TargetJoinParams {
            source_variable,
            target_variable,
            rel_qualifier: &rel_instance.alias,
            node_map,
            rel_map,
            direction,
        };
        self.join_relationship_to_target(builder, cat, ctx, &target_params)
    }

    /// Build a qualified relationship scan
    fn build_relationship_scan(
        &self,
        rel_instance: &RelationshipInstance,
        rel_source: Arc<dyn datafusion::logical_expr::TableSource>,
    ) -> Result<LogicalPlan> {
        let rel_schema = rel_source.schema();
        let rel_builder =
            LogicalPlanBuilder::scan(&rel_instance.rel_type, rel_source, None).unwrap();

        // Use unique alias from rel_instance to avoid column conflicts
        let rel_qualified_exprs: Vec<Expr> = rel_schema
            .fields()
            .iter()
            .map(|field| {
                let qualified_name = format!("{}__{}", rel_instance.alias, field.name());
                col(field.name()).alias(&qualified_name)
            })
            .collect();

        Ok(rel_builder
            .project(rel_qualified_exprs)
            .unwrap()
            .build()
            .unwrap())
    }

    /// Join source node plan with relationship scan
    fn join_source_to_relationship(
        &self,
        left_plan: LogicalPlan,
        rel_scan: LogicalPlan,
        params: &SourceJoinParams,
    ) -> Result<LogicalPlanBuilder> {
        // Determine join keys based on direction
        let right_key = match params.direction {
            RelationshipDirection::Outgoing => &params.rel_map.source_id_field,
            RelationshipDirection::Incoming => &params.rel_map.target_id_field,
            RelationshipDirection::Undirected => &params.rel_map.source_id_field,
        };

        let qualified_left_key = format!("{}__{}", params.source_variable, params.node_id_field);
        let qualified_right_key = format!("{}__{}", params.rel_qualifier, right_key);

        Ok(LogicalPlanBuilder::from(left_plan)
            .join(
                rel_scan,
                JoinType::Inner,
                (vec![qualified_left_key], vec![qualified_right_key]),
                None,
            )
            .unwrap())
    }

    /// Join relationship with target node scan
    fn join_relationship_to_target(
        &self,
        mut builder: LogicalPlanBuilder,
        cat: &Arc<dyn GraphSourceCatalog>,
        ctx: &PlanningContext,
        params: &TargetJoinParams,
    ) -> Result<LogicalPlan> {
        // For now, assume target has same label as source (simplified)
        let Some(target_label) = ctx
            .analysis
            .var_to_label
            .get(params.source_variable)
            .cloned()
        else {
            return Ok(builder.build().unwrap());
        };

        let Some(target_source) = cat.node_source(&target_label) else {
            return Ok(builder.build().unwrap());
        };

        // Create target node scan with qualified column aliases
        let target_schema = target_source.schema();
        let target_builder = LogicalPlanBuilder::scan(&target_label, target_source, None).unwrap();

        let target_qualified_exprs: Vec<Expr> = target_schema
            .fields()
            .iter()
            .map(|field| {
                let qualified_name = format!("{}__{}", params.target_variable, field.name());
                col(field.name()).alias(&qualified_name)
            })
            .collect();

        let target_scan = target_builder
            .project(target_qualified_exprs)
            .unwrap()
            .build()
            .unwrap();

        // Determine target join keys
        let target_key = match params.direction {
            RelationshipDirection::Outgoing => &params.rel_map.target_id_field,
            RelationshipDirection::Incoming => &params.rel_map.source_id_field,
            RelationshipDirection::Undirected => &params.rel_map.target_id_field,
        };

        let qualified_rel_target_key = format!("{}__{}", params.rel_qualifier, target_key);
        let qualified_target_key =
            format!("{}__{}", params.target_variable, &params.node_map.id_field);

        builder = builder
            .join(
                target_scan,
                JoinType::Inner,
                (vec![qualified_rel_target_key], vec![qualified_target_key]),
                None,
            )
            .unwrap();

        Ok(builder.build().unwrap())
    }

    // ============================================================================
    // Expression Translators
    // ============================================================================

    fn to_df_boolean_expr(&self, expr: &crate::ast::BooleanExpression) -> Expr {
        use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO};
        match expr {
            BE::Comparison {
                left,
                operator,
                right,
            } => {
                let l = self.to_df_value_expr(left);
                let r = self.to_df_value_expr(right);
                let op = match operator {
                    CO::Equal => Operator::Eq,
                    CO::NotEqual => Operator::NotEq,
                    CO::LessThan => Operator::Lt,
                    CO::LessThanOrEqual => Operator::LtEq,
                    CO::GreaterThan => Operator::Gt,
                    CO::GreaterThanOrEqual => Operator::GtEq,
                };
                Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                })
            }
            BE::In { expression, list } => {
                use datafusion::logical_expr::expr::InList as DFInList;
                let expr = self.to_df_value_expr(expression);
                let list_exprs = list
                    .iter()
                    .map(|item| self.to_df_value_expr(item))
                    .collect::<Vec<_>>();
                Expr::InList(DFInList::new(Box::new(expr), list_exprs, false))
            }
            BE::And(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(self.to_df_boolean_expr(l)),
                op: Operator::And,
                right: Box::new(self.to_df_boolean_expr(r)),
            }),
            BE::Or(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(self.to_df_boolean_expr(l)),
                op: Operator::Or,
                right: Box::new(self.to_df_boolean_expr(r)),
            }),
            BE::Not(inner) => Expr::Not(Box::new(self.to_df_boolean_expr(inner))),
            BE::Exists(prop) => Expr::IsNotNull(Box::new(
                self.to_df_value_expr(&crate::ast::ValueExpression::Property(prop.clone())),
            )),
            _ => lit(true),
        }
    }

    fn to_df_value_expr(&self, expr: &crate::ast::ValueExpression) -> Expr {
        use crate::ast::{PropertyValue as PV, ValueExpression as VE};
        match expr {
            VE::Property(prop) => {
                // Create qualified column name: variable__property
                let qualified_name = format!("{}__{}", prop.variable, prop.property);
                col(&qualified_name)
            }
            VE::Variable(v) => col(v),
            VE::Literal(PV::String(s)) => lit(s.clone()),
            VE::Literal(PV::Integer(i)) => lit(*i),
            VE::Literal(PV::Float(f)) => lit(*f),
            VE::Literal(PV::Boolean(b)) => lit(*b),
            VE::Literal(PV::Null) => {
                datafusion::logical_expr::Expr::Literal(datafusion::scalar::ScalarValue::Null, None)
            }
            VE::Literal(PV::Parameter(_)) => lit(0),
            VE::Literal(PV::Property(prop)) => {
                // Create qualified column name: variable__property
                let qualified_name = format!("{}__{}", prop.variable, prop.property);
                col(&qualified_name)
            }
            VE::Function { .. } | VE::Arithmetic { .. } => lit(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, ValueExpression,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use crate::source_catalog::{InMemoryCatalog, SimpleTableSource};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn person_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]))
    }

    fn make_catalog() -> Arc<dyn crate::source_catalog::GraphSourceCatalog> {
        let person_src = Arc::new(SimpleTableSource::new(person_schema()));
        let knows_schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::Int64, false),
            Field::new("dst_person_id", DataType::Int64, false),
        ]));
        let knows_src = Arc::new(SimpleTableSource::new(knows_schema));
        Arc::new(
            InMemoryCatalog::new()
                .with_node_source("Person", person_src)
                .with_relationship_source("KNOWS", knows_src),
        )
    }

    #[test]
    fn test_df_boolean_expr_in_list() {
        let cfg = crate::config::GraphConfig::builder().build().unwrap();
        let planner = DataFusionPlanner::new(cfg);
        let expr = BooleanExpression::In {
            expression: ValueExpression::Property(PropertyRef {
                variable: "rel".into(),
                property: "relationship_type".into(),
            }),
            list: vec![
                ValueExpression::Literal(PropertyValue::String("WORKS_FOR".into())),
                ValueExpression::Literal(PropertyValue::String("PART_OF".into())),
            ],
        };

        if let Expr::InList(in_list) = planner.to_df_boolean_expr(&expr) {
            assert!(!in_list.negated);
            assert_eq!(in_list.list.len(), 2);
            match *in_list.expr {
                Expr::Column(ref col_expr) => {
                    assert_eq!(col_expr.name(), "rel__relationship_type");
                }
                other => panic!("Expected column expression, got {:?}", other),
            }
        } else {
            panic!("Expected InList expression");
        }
    }

    #[test]
    fn test_df_planner_scan_filter_project() {
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let pred = BooleanExpression::Comparison {
            left: ValueExpression::Property(PropertyRef {
                variable: "n".to_string(),
                property: "age".to_string(),
            }),
            operator: ComparisonOperator::GreaterThan,
            right: ValueExpression::Literal(PropertyValue::Integer(30)),
        };

        let filter = LogicalOperator::Filter {
            input: Box::new(scan),
            predicate: pred,
        };

        let project = LogicalOperator::Project {
            input: Box::new(filter),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(s.contains("Projection"), "plan missing Projection: {}", s);
        assert!(s.contains("Filter"), "plan missing Filter: {}", s);
        assert!(s.contains("TableScan"), "plan missing TableScan: {}", s);
        assert!(
            s.contains("Person") || s.contains("person"),
            "plan missing table name: {}",
            s
        );
    }

    #[test]
    fn test_df_planner_property_pushdown_filter() {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: props,
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&scan).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "plan missing Filter: {}", s);
        assert!(s.contains("TableScan"), "plan missing TableScan: {}", s);
        assert!(
            s.contains("Person") || s.contains("person"),
            "plan missing table name: {}",
            s
        );
    }

    #[test]
    fn test_df_planner_expand_creates_join_filter() {
        // MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "b".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("Join(") && s.contains("Inner"),
            "plan missing Inner Join: {}",
            s
        );
        assert!(
            s.contains("TableScan") && s.contains("person"),
            "plan missing person scan: {}",
            s
        );
        assert!(
            s.contains("TableScan") && (s.contains("KNOWS") || s.contains("knows")),
            "plan missing relationship scan: {}",
            s
        );
    }

    #[test]
    fn test_scan_aliasing_projects_variable_prefixed_columns() {
        // MATCH (n:Person) RETURN n.name
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();

        let s = format!("{:?}", df_plan);
        assert!(s.contains("Projection"), "plan missing Projection: {}", s);
        assert!(
            s.contains("n__name"),
            "missing qualified projected column n__name: {}",
            s
        );
    }

    #[test]
    fn test_expand_uses_qualified_join_keys_with_type_alias() {
        // MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("a__id"),
            "missing qualified node id in join: {}",
            s
        );
        assert!(
            s.contains("knows_1__src_person_id"),
            "missing qualified rel key in join: {}",
            s
        );
    }

    #[test]
    fn test_expand_uses_relationship_variable_for_alias() {
        // MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r.src_person_id
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "r".into(),
                    property: "src_person_id".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("r__src_person_id"),
            "missing rel-var qualified column: {}",
            s
        );
    }

    #[test]
    fn test_where_on_relationship_property_with_rel_var() {
        // MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE r.src_person_id = 1 RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
        };
        let filter = LogicalOperator::Filter {
            input: Box::new(expand),
            predicate: BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "r".into(),
                    property: "src_person_id".into(),
                }),
                operator: ComparisonOperator::Equal,
                right: ValueExpression::Literal(PropertyValue::Integer(1)),
            },
        };
        let project = LogicalOperator::Project {
            input: Box::new(filter),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "missing Filter: {}", s);
        assert!(
            s.contains("r__src_person_id"),
            "missing qualified rel column in filter: {}",
            s
        );
    }

    #[test]
    fn test_exists_on_relationship_property_is_qualified() {
        // MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE EXISTS(r.src_person_id) RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
        };
        let pred = BooleanExpression::Exists(PropertyRef {
            variable: "r".into(),
            property: "src_person_id".into(),
        });
        let filter = LogicalOperator::Filter {
            input: Box::new(expand),
            predicate: pred,
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&filter).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "missing Filter: {}", s);
        assert!(
            s.contains("r__src_person_id") || s.contains("IsNotNull"),
            "missing qualified rel column or IsNotNull in filter: {}",
            s
        );
    }

    #[test]
    fn test_in_list_on_relationship_property_is_qualified() {
        // MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE r.src_person_id IN [1,2] RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
        };
        let filter = LogicalOperator::Filter {
            input: Box::new(expand),
            predicate: BooleanExpression::In {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "r".into(),
                    property: "src_person_id".into(),
                }),
                list: vec![
                    ValueExpression::Literal(PropertyValue::Integer(1)),
                    ValueExpression::Literal(PropertyValue::Integer(2)),
                ],
            },
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&filter).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "missing Filter: {}", s);
        assert!(
            s.contains("r__src_person_id"),
            "missing qualified rel column in IN list filter: {}",
            s
        );
    }

    #[test]
    fn test_incoming_join_qualified_keys() {
        // MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Incoming,
            relationship_variable: None,
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("knows_1__dst_person_id"),
            "incoming join should use dst key: {}",
            s
        );
    }

    #[test]
    fn test_undirected_join_qualified_keys() {
        // MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Undirected,
            relationship_variable: None,
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(expand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("knows_1__src_person_id"),
            "undirected uses src key side for predicate: {}",
            s
        );
    }

    #[test]
    fn test_distinct_and_order_with_qualified_columns() {
        // ORDER is currently skipped in physical planner; just ensure Distinct appears and plan builds
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };
        let distinct = LogicalOperator::Distinct {
            input: Box::new(project),
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&distinct).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Distinct"), "missing Distinct in plan: {}", s);
    }

    #[test]
    fn test_skip_limit_after_aliasing() {
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };
        let offset = LogicalOperator::Offset {
            input: Box::new(project),
            offset: 5,
        };
        let limit = LogicalOperator::Limit {
            input: Box::new(offset),
            count: 10,
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&limit).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Limit"), "missing Limit in plan: {}", s);
    }

    #[test]
    fn test_where_rel_and_node_properties() {
        // WHERE r.src_person_id = 1 AND a.age > 30
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".into()),
            properties: Default::default(),
        };
        let pred = BooleanExpression::And(
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "r".into(),
                    property: "src_person_id".into(),
                }),
                operator: ComparisonOperator::Equal,
                right: ValueExpression::Literal(PropertyValue::Integer(1)),
            }),
            Box::new(BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "age".into(),
                }),
                operator: ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(PropertyValue::Integer(30)),
            }),
        );
        let filter = LogicalOperator::Filter {
            input: Box::new(expand),
            predicate: pred,
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&filter).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "missing Filter: {}", s);
        assert!(
            s.contains("r__src_person_id"),
            "missing qualified rel filter: {}",
            s
        );
        assert!(
            s.contains("a__age") || s.contains("age"),
            "missing node age filter: {}",
            s
        );
    }

    #[test]
    fn test_exists_and_in_on_node_props_materialized() {
        // EXISTS(a.name) and a.age IN [20,30]
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let pred = BooleanExpression::And(
            Box::new(BooleanExpression::Exists(PropertyRef {
                variable: "a".into(),
                property: "name".into(),
            })),
            Box::new(BooleanExpression::In {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "age".into(),
                }),
                list: vec![
                    ValueExpression::Literal(PropertyValue::Integer(20)),
                    ValueExpression::Literal(PropertyValue::Integer(30)),
                ],
            }),
        );
        let filter = LogicalOperator::Filter {
            input: Box::new(scan_a),
            predicate: pred,
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&filter).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Filter"), "missing Filter: {}", s);
        assert!(
            s.contains("a__name") || s.contains("IsNotNull"),
            "missing EXISTS on a__name: {}",
            s
        );
        assert!(
            s.contains("a__age") || s.contains("age"),
            "missing IN on a.age: {}",
            s
        );
    }

    #[test]
    fn test_varlength_expand_placeholder_builds() {
        // MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let vlexpand = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan_a),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".into()),
            min_length: Some(1),
            max_length: Some(2),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "a".into(),
                    property: "name".into(),
                }),
                alias: None,
            }],
        };
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(
            s.contains("Join(") && s.contains("Inner"),
            "missing Inner Join: {}",
            s
        );
    }

    #[test]
    fn test_query_analysis_single_hop() {
        // Test that analysis correctly identifies relationship instances
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::new(cfg);
        let analysis = planner.analyze(&expand).unwrap();

        // Should have two variable mappings: a and b both map to Person
        assert_eq!(analysis.var_to_label.len(), 2);
        assert_eq!(analysis.var_to_label.get("a"), Some(&"Person".to_string()));
        assert_eq!(analysis.var_to_label.get("b"), Some(&"Person".to_string()));

        // Should have one relationship instance
        assert_eq!(analysis.relationship_instances.len(), 1);
        assert_eq!(analysis.relationship_instances[0].rel_type, "KNOWS");
        assert_eq!(analysis.relationship_instances[0].alias, "knows_1");
        assert_eq!(analysis.relationship_instances[0].id, 1);
    }

    #[test]
    fn test_query_analysis_two_hop() {
        // Test that two-hop queries get unique relationship instances
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".into(),
            label: "Person".into(),
            properties: Default::default(),
        };
        let expand1 = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
        };
        let expand2 = LogicalOperator::Expand {
            input: Box::new(expand1),
            source_variable: "b".into(),
            target_variable: "c".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::new(cfg);
        let analysis = planner.analyze(&expand2).unwrap();

        // Should have two relationship instances with UNIQUE aliases
        assert_eq!(analysis.relationship_instances.len(), 2);
        assert_eq!(analysis.relationship_instances[0].alias, "knows_1");
        assert_eq!(analysis.relationship_instances[1].alias, "knows_2");

        // Both should be KNOWS but with different IDs
        assert_eq!(analysis.relationship_instances[0].rel_type, "KNOWS");
        assert_eq!(analysis.relationship_instances[1].rel_type, "KNOWS");
        assert_eq!(analysis.relationship_instances[0].id, 1);
        assert_eq!(analysis.relationship_instances[1].id, 2);
    }

    #[test]
    fn test_planning_context_tracks_instances() {
        // Test that PlanningContext correctly iterates through instances
        let instances = vec![
            RelationshipInstance {
                id: 1,
                rel_type: "KNOWS".to_string(),
                source_var: "a".to_string(),
                target_var: "b".to_string(),
                direction: crate::ast::RelationshipDirection::Outgoing,
                alias: "knows_1".to_string(),
            },
            RelationshipInstance {
                id: 2,
                rel_type: "KNOWS".to_string(),
                source_var: "b".to_string(),
                target_var: "c".to_string(),
                direction: crate::ast::RelationshipDirection::Outgoing,
                alias: "knows_2".to_string(),
            },
        ];

        let analysis = QueryAnalysis {
            var_to_label: Default::default(),
            relationship_instances: instances,
            required_datasets: Default::default(),
        };

        let mut ctx = PlanningContext::new(&analysis);

        // First call should return knows_1
        let inst1 = ctx.next_relationship_instance("KNOWS").unwrap();
        assert_eq!(inst1.alias, "knows_1");
        assert_eq!(inst1.id, 1);

        // Second call should return knows_2
        let inst2 = ctx.next_relationship_instance("KNOWS").unwrap();
        assert_eq!(inst2.alias, "knows_2");
        assert_eq!(inst2.id, 2);

        // Third call should fail (no more instances)
        assert!(ctx.next_relationship_instance("KNOWS").is_err());
    }

    #[test]
    fn test_order_by_single_column_asc() {
        use crate::ast::{PropertyRef, SortDirection, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Build: Project -> Sort
        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                alias: None,
            }],
        };

        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![SortItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                direction: SortDirection::Ascending,
            }],
        };

        let df_plan = planner.plan(&sort).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain Sort operator
        println!("Plan: {}", s);
        assert!(s.contains("Sort") || s.contains("sort"));
        assert!(s.contains("n__name"));
    }

    #[test]
    fn test_order_by_multiple_columns() {
        use crate::ast::{PropertyRef, SortDirection, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    }),
                    alias: None,
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    alias: None,
                },
            ],
        };

        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![
                SortItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    direction: SortDirection::Descending,
                },
                SortItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "n".to_string(),
                        property: "name".to_string(),
                    }),
                    direction: SortDirection::Ascending,
                },
            ],
        };

        let df_plan = planner.plan(&sort).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain Sort with both columns
        assert!(s.contains("Sort") || s.contains("sort"));
        assert!(s.contains("n__age"));
        assert!(s.contains("n__name"));
    }

    #[test]
    fn test_order_by_with_limit() {
        use crate::ast::{PropertyRef, SortDirection, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                alias: None,
            }],
        };

        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![SortItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                direction: SortDirection::Ascending,
            }],
        };

        let limit = LogicalOperator::Limit {
            input: Box::new(sort),
            count: 10,
        };

        let df_plan = planner.plan(&limit).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain both Limit and Sort
        assert!(s.contains("Limit") || s.contains("limit"));
        assert!(s.contains("Sort") || s.contains("sort"));
        assert!(s.contains("n__name"));
    }

    #[test]
    fn test_project_with_alias() {
        use crate::ast::{PropertyRef, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                alias: Some("person_name".to_string()),
            }],
        };

        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain the alias
        assert!(s.contains("person_name"));
    }

    #[test]
    fn test_project_with_multiple_aliases() {
        use crate::ast::{PropertyRef, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "p".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "name".to_string(),
                    }),
                    alias: Some("name".to_string()),
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "age".to_string(),
                    }),
                    alias: Some("age".to_string()),
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain both aliases
        assert!(s.contains("name"));
        assert!(s.contains("age"));
    }

    #[test]
    fn test_project_mixed_with_and_without_alias() {
        use crate::ast::{PropertyRef, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "p".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "name".to_string(),
                    }),
                    alias: Some("full_name".to_string()),
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "age".to_string(),
                    }),
                    alias: None, // No alias - should use qualified name
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain the alias and the qualified name
        assert!(s.contains("full_name"));
        assert!(s.contains("p__age"));
    }
}
