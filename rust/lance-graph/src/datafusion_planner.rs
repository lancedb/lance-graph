// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion-based physical planner for graph queries
//!
//! This module implements the proper graph-to-relational mapping:
//! - Nodes as Tables: Each node label becomes a table
//! - Relationships as Tables: Each relationship type becomes a linking table
//! - Cypher traversal becomes SQL joins
//!
//! Uses DataFusion's LogicalPlan and optimizer for world-class query optimization.

use crate::error::Result;
use crate::logical_plan::*;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::{
    col, lit, BinaryExpr, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator,
};
use std::sync::Arc;

/// Planner abstraction for graph-to-physical planning
pub trait GraphPhysicalPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan>;
}

/// DataFusion-based physical planner
pub struct DataFusionPlanner {
    #[allow(dead_code)]
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

impl GraphPhysicalPlanner for DataFusionPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        use std::collections::HashMap;
        let mut var_labels: HashMap<String, String> = HashMap::new();
        self.plan_operator_with_ctx(logical_plan, &mut var_labels)
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
        use std::collections::HashMap as StdHashMap;
        use std::sync::Arc;

        // Collect variable -> label mappings from the logical plan
        let mut variable_mappings: StdHashMap<String, String> = StdHashMap::new();
        Self::collect_variable_mappings(logical_plan, &mut variable_mappings)?;

        // Build an in-memory catalog from provided datasets (nodes and relationships)
        let mut catalog = InMemoryCatalog::new();
        let mut added_labels = std::collections::HashSet::new();
        for label in variable_mappings.values() {
            if added_labels.insert(label.clone()) {
                if let Some(batch) = datasets.get(label) {
                    let src = Arc::new(SimpleTableSource::new(batch.schema()));
                    catalog = catalog.with_node_source(label, src);
                }
            }
        }

        // Register relationship sources if datasets include them
        for rel_type in self.config.relationship_mappings.keys() {
            if let Some(batch) = datasets.get(rel_type) {
                let src = Arc::new(SimpleTableSource::new(batch.schema()));
                catalog = catalog.with_relationship_source(rel_type, src);
            }
        }

        // Plan using a planner bound to this catalog so scans get qualified projections
        let planner_with_cat =
            DataFusionPlanner::with_catalog(self.config.clone(), Arc::new(catalog));
        planner_with_cat.plan(logical_plan)
    }

    /// Collect variable to label mappings from logical plan
    fn collect_variable_mappings(
        op: &LogicalOperator,
        mappings: &mut std::collections::HashMap<String, String>,
    ) -> Result<()> {
        match op {
            LogicalOperator::ScanByLabel {
                variable, label, ..
            } => {
                mappings.insert(variable.clone(), label.clone());
            }
            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                ..
            } => {
                Self::collect_variable_mappings(input, mappings)?;
                // For expand, we need to infer the target variable's label
                // For now, assume target has same label as source (simplified)
                if let Some(source_label) = mappings.get(source_variable).cloned() {
                    mappings.insert(target_variable.clone(), source_label);
                }
            }
            LogicalOperator::Filter { input, .. }
            | LogicalOperator::Project { input, .. }
            | LogicalOperator::Distinct { input }
            | LogicalOperator::Limit { input, .. }
            | LogicalOperator::Offset { input, .. }
            | LogicalOperator::Sort { input, .. } => {
                Self::collect_variable_mappings(input, mappings)?;
            }
            LogicalOperator::VariableLengthExpand { input, .. }
            | LogicalOperator::Join { left: input, .. } => {
                Self::collect_variable_mappings(input, mappings)?;
            }
        }
        Ok(())
    }

    fn plan_operator_with_ctx(
        &self,
        op: &LogicalOperator,
        var_labels: &mut std::collections::HashMap<String, String>,
    ) -> Result<LogicalPlan> {
        match op {
            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
                ..
            } => {
                // Track variable -> label mapping
                var_labels.insert(variable.clone(), label.clone());

                // Try to use catalog if available
                if let Some(cat) = &self.catalog {
                    if let Some(source) = cat.node_source(label) {
                        // Get schema before moving source
                        let schema = source.schema();
                        let mut builder = LogicalPlanBuilder::scan(label, source, None).unwrap();

                        // Apply property filters using unqualified names (before aliasing)
                        for (k, v) in properties.iter() {
                            let lit_expr = self
                                .to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
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
                // Use LogicalPlanBuilder to create a proper scan
                let empty_source = Arc::new(crate::source_catalog::SimpleTableSource::empty());
                let builder = LogicalPlanBuilder::scan(label, empty_source, None).map_err(|e| {
                    crate::error::GraphError::PlanError {
                        message: format!("Failed to create table scan for {}: {}", label, e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?;

                Ok(builder
                    .build()
                    .map_err(|e| crate::error::GraphError::PlanError {
                        message: format!("Failed to build table scan for {}: {}", label, e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?)
            }
            LogicalOperator::Filter { input, predicate } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                let expr = self.to_df_boolean_expr(predicate);
                Ok(LogicalPlanBuilder::from(input_plan)
                    .filter(expr)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Project { input, projections } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                let exprs: Vec<Expr> = projections
                    .iter()
                    .map(|p| self.to_df_value_expr(&p.expression))
                    .collect();
                Ok(LogicalPlanBuilder::from(input_plan)
                    .project(exprs)
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Distinct { input } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .distinct()
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Sort { input, .. } => {
                // Schema-less placeholder: skip sort for now
                self.plan_operator_with_ctx(input, var_labels)
            }
            LogicalOperator::Limit { input, count } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
                Ok(LogicalPlanBuilder::from(input_plan)
                    .limit(0, Some((*count) as usize))
                    .unwrap()
                    .build()
                    .unwrap())
            }
            LogicalOperator::Offset { input, offset } => {
                let input_plan = self.plan_operator_with_ctx(input, var_labels)?;
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
                let left_plan = self.plan_operator_with_ctx(input, var_labels)?;
                // TODO(two-hop+): Support chaining multiple hops in the physical plan.
                // For single hop we scan the relationship table and filter with an ON expression.
                // For two-hop (e.g., a-[:R1]->m-[:R2]->b), we should:
                //   1) Join a with R1 (as done here)
                //   2) Join the result with R2
                //   3) Join the result with the b node scan
                // Ensure we maintain/propagate variable->label mapping (var_labels) and
                // project/qualify columns to avoid ambiguity across joins.
                // For VariableLengthExpand with bounds, consider unrolling small fixed bounds
                // (e.g., *1..2) into a UNION ALL of 1-hop and 2-hop plans.
                // Attempt first hop: source node -> relationship table
                if let (Some(cat), Some(rel_type)) = (&self.catalog, relationship_types.first()) {
                    if let Some(rel_map) = self.config.relationship_mappings.get(rel_type) {
                        if let Some(src_label) = var_labels.get(source_variable) {
                            if let Some(node_map) = self.config.node_mappings.get(src_label) {
                                if let Some(rel_source) =
                                    cat.relationship_source(&rel_map.relationship_type)
                                {
                                    // Create relationship scan with qualified column aliases
                                    let rel_schema = rel_source.schema();
                                    let rel_builder = LogicalPlanBuilder::scan(
                                        &rel_map.relationship_type,
                                        rel_source,
                                        None,
                                    )
                                    .unwrap();

                                    // Create qualified column aliases for relationship
                                    // Use relationship variable if available, otherwise use relationship type (lowercase)
                                    let rel_type_lower = rel_map.relationship_type.to_lowercase();
                                    let rel_qualifier =
                                        relationship_variable.as_deref().unwrap_or(&rel_type_lower);
                                    let rel_qualified_exprs: Vec<Expr> = rel_schema
                                        .fields()
                                        .iter()
                                        .map(|field| {
                                            let qualified_name =
                                                format!("{}__{}", rel_qualifier, field.name());
                                            col(field.name()).alias(&qualified_name)
                                        })
                                        .collect();

                                    let rel_scan = rel_builder
                                        .project(rel_qualified_exprs)
                                        .unwrap()
                                        .build()
                                        .unwrap();

                                    // Determine join keys based on direction
                                    let (left_key, right_key) = match direction {
                                        crate::ast::RelationshipDirection::Outgoing => {
                                            (&node_map.id_field, &rel_map.source_id_field)
                                        }
                                        crate::ast::RelationshipDirection::Incoming => {
                                            (&node_map.id_field, &rel_map.target_id_field)
                                        }
                                        crate::ast::RelationshipDirection::Undirected => {
                                            (&node_map.id_field, &rel_map.source_id_field)
                                        }
                                    };
                                    // Use qualified column names for both sides of the join
                                    let qualified_left_key =
                                        format!("{}__{}", source_variable, left_key);
                                    let qualified_right_key =
                                        format!("{}__{}", rel_qualifier, right_key);

                                    // Use proper inner join instead of CrossJoin + Filter
                                    let mut builder = LogicalPlanBuilder::from(left_plan)
                                        .join(
                                            rel_scan,
                                            JoinType::Inner,
                                            (
                                                vec![qualified_left_key.clone()],
                                                vec![qualified_right_key.clone()],
                                            ),
                                            None,
                                        )
                                        .unwrap();

                                    // Add target node scan and join
                                    // For now, assume target has same label as source (simplified)
                                    if let Some(target_label) =
                                        var_labels.get(source_variable).cloned()
                                    {
                                        if let Some(target_source) = cat.node_source(&target_label)
                                        {
                                            // Create target node scan with qualified column aliases
                                            let target_schema = target_source.schema();
                                            let target_builder = LogicalPlanBuilder::scan(
                                                &target_label,
                                                target_source,
                                                None,
                                            )
                                            .unwrap();

                                            // Create qualified column aliases for target: target_variable__property
                                            let target_qualified_exprs: Vec<Expr> = target_schema
                                                .fields()
                                                .iter()
                                                .map(|field| {
                                                    let qualified_name = format!(
                                                        "{}__{}",
                                                        target_variable,
                                                        field.name()
                                                    );
                                                    col(field.name()).alias(&qualified_name)
                                                })
                                                .collect();

                                            let target_scan = target_builder
                                                .project(target_qualified_exprs)
                                                .unwrap()
                                                .build()
                                                .unwrap();

                                            // Determine target join keys
                                            let target_key = match direction {
                                                crate::ast::RelationshipDirection::Outgoing => {
                                                    &rel_map.target_id_field
                                                }
                                                crate::ast::RelationshipDirection::Incoming => {
                                                    &rel_map.source_id_field
                                                }
                                                crate::ast::RelationshipDirection::Undirected => {
                                                    &rel_map.target_id_field
                                                }
                                            };
                                            let qualified_rel_target_key =
                                                format!("{}__{}", rel_qualifier, target_key);
                                            let qualified_target_key = format!(
                                                "{}__{}",
                                                target_variable, &node_map.id_field
                                            );

                                            // Use proper inner join for relationship->target
                                            builder = builder
                                                .join(
                                                    target_scan,
                                                    JoinType::Inner,
                                                    (
                                                        vec![qualified_rel_target_key],
                                                        vec![qualified_target_key],
                                                    ),
                                                    None,
                                                )
                                                .unwrap();

                                            // Track target variable label
                                            var_labels
                                                .insert(target_variable.clone(), target_label);
                                        }
                                    }

                                    return Ok(builder.build().unwrap());
                                }
                            }
                        }
                    }
                }
                // Fallback: pass-through
                var_labels
                    .entry(target_variable.clone())
                    .or_insert_with(|| "Node".to_string());
                Ok(self.plan_operator_with_ctx(input, var_labels)?)
            }
            LogicalOperator::Join { left, .. } => {
                // Not yet implemented: explicit join. For now, use left branch
                self.plan_operator_with_ctx(left, var_labels)
            }
        }
    }

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
            s.contains("KNOWS__src_person_id") || s.contains("knows__src_person_id"),
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
            s.contains("KNOWS__dst_person_id") || s.contains("knows__dst_person_id"),
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
            s.contains("KNOWS__src_person_id") || s.contains("knows__src_person_id"),
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
}

/*
The main issues to fix:
1. Column import path: Use datafusion::common::Column instead of datafusion::logical_expr::Column
2. TableSource trait: Need to use LogicalTableSource or create proper table sources
3. ScalarValue::Null needs Option<FieldMetadata> parameter
4. SortExpr type issues with DataFusion's Expr system

Reference implementation should be here when these issues are resolved.
*/
