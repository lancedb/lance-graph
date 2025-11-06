// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion-based physical planner for graph queries
//!
//! Translates graph logical plans into DataFusion logical plans using a two-phase approach:
//!
//! ## Phase 1: Analysis
//! - Assigns unique IDs to relationship instances to avoid column conflicts
//! - Collects variable-to-label mappings and required datasets
//!
//! ## Phase 2: Plan Building
//! - Nodes → Table scans, Relationships → Linking tables, Traversals → Joins
//! - Variable-length paths (`*1..3`) use unrolling: generate fixed-length plans + UNION
//! - All columns qualified as `{variable}__{column}` to avoid ambiguity

use crate::ast::RelationshipDirection;
use crate::error::Result;
use crate::logical_plan::*;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::{
    col, lit, BinaryExpr, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator,
};
use datafusion_functions_aggregate::count::count;
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

    /// Helper to convert DataFusion builder errors into GraphError::PlanError with context
    fn plan_error<E: std::fmt::Display>(
        &self,
        context: &str,
        error: E,
    ) -> crate::error::GraphError {
        crate::error::GraphError::PlanError {
            message: format!("{}: {}", context, error),
            location: snafu::Location::new(file!(), line!(), column!()),
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
    target_variable: &'a str,
    rel_qualifier: &'a str,
    node_map: &'a crate::config::NodeMapping,
    rel_map: &'a crate::config::RelationshipMapping,
    direction: &'a RelationshipDirection,
    target_properties: &'a HashMap<String, crate::ast::PropertyValue>,
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
            target_label,
            relationship_types,
            direction,
            relationship_variable,
            ..
        } => {
            // Recursively analyze input first
            analyze_operator(input, analysis, rel_counter)?;

            // Register the target variable with its label from the logical plan
            analysis
                .var_to_label
                .insert(target_variable.clone(), target_label.clone());

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
        LogicalOperator::VariableLengthExpand {
            input,
            source_variable,
            target_variable,
            relationship_types,
            direction,
            relationship_variable,
            min_length,
            max_length,
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

            // For variable-length paths, register multiple instances (one per hop)
            // We need to register instances for all possible hop counts
            if let Some(rel_type) = relationship_types.first() {
                let max_hops = max_length.unwrap_or(crate::MAX_VARIABLE_LENGTH_HOPS);
                let min_hops = min_length.unwrap_or(1).max(1);

                // Register instances for each hop count we'll generate
                for hop_count in min_hops..=max_hops {
                    for _ in 0..hop_count {
                        let instance_id = rel_counter
                            .entry(rel_type.clone())
                            .and_modify(|c| *c += 1)
                            .or_insert(1);

                        // Use relationship variable if provided, otherwise use type_instanceId
                        let alias = if let Some(rel_var) = relationship_variable {
                            format!("{}_{}", rel_var, instance_id)
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
                    }
                }

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
                let expr = Self::to_df_boolean_expr(predicate);
                LogicalPlanBuilder::from(input_plan)
                    .filter(expr)
                    .map_err(|e| self.plan_error("Failed to build filter", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            LogicalOperator::Project { input, projections } => {
                let input_plan = self.build_operator(ctx, input)?;

                // Check if any projection contains an aggregate function
                let has_aggregates = projections
                    .iter()
                    .any(|p| Self::contains_aggregate(&p.expression));

                if has_aggregates {
                    // Build aggregate plan
                    // Separate group expressions (non-aggregates) from aggregate expressions
                    let mut group_exprs = Vec::new();
                    let mut agg_exprs = Vec::new();
                    // Store computed aliases for aggregates to reuse in final projection
                    let mut agg_aliases = Vec::new();

                    for p in projections {
                        let expr = Self::to_df_value_expr(&p.expression);

                        if Self::contains_aggregate(&p.expression) {
                            // Aggregate expressions get aliased
                            let alias = if let Some(alias) = &p.alias {
                                alias.clone()
                            } else {
                                self.to_cypher_column_name(&p.expression)
                            };
                            agg_exprs.push(expr.alias(&alias));
                            agg_aliases.push(alias);
                        } else {
                            // Group expressions: use raw expression for grouping, no alias
                            group_exprs.push(expr);
                        }
                    }

                    // After aggregation, add a projection to apply aliases to group columns
                    let mut final_projection = Vec::new();
                    let mut agg_idx = 0;
                    for p in projections {
                        if !Self::contains_aggregate(&p.expression) {
                            // Re-create the expression and apply alias
                            let expr = Self::to_df_value_expr(&p.expression);
                            let aliased = if let Some(alias) = &p.alias {
                                expr.alias(alias)
                            } else {
                                let cypher_name = self.to_cypher_column_name(&p.expression);
                                expr.alias(cypher_name)
                            };
                            final_projection.push(aliased);
                        } else {
                            // For aggregates, reference the column using the same alias we computed earlier
                            final_projection.push(col(&agg_aliases[agg_idx]));
                            agg_idx += 1;
                        }
                    }

                    LogicalPlanBuilder::from(input_plan)
                        .aggregate(group_exprs, agg_exprs)
                        .map_err(|e| self.plan_error("Failed to build aggregate", e))?
                        .project(final_projection)
                        .map_err(|e| self.plan_error("Failed to project after aggregate", e))?
                        .build()
                        .map_err(|e| self.plan_error("Failed to build plan", e))
                } else {
                    // Regular projection
                    let exprs: Vec<Expr> = projections
                        .iter()
                        .map(|p| {
                            let expr = Self::to_df_value_expr(&p.expression);
                            // Apply alias if provided, otherwise use Cypher dot notation
                            if let Some(alias) = &p.alias {
                                expr.alias(alias)
                            } else {
                                // Convert to Cypher dot notation (e.g., p__name -> p.name)
                                let cypher_name = self.to_cypher_column_name(&p.expression);
                                expr.alias(cypher_name)
                            }
                        })
                        .collect();
                    LogicalPlanBuilder::from(input_plan)
                        .project(exprs)
                        .map_err(|e| self.plan_error("Failed to build projection", e))?
                        .build()
                        .map_err(|e| self.plan_error("Failed to build plan", e))
                }
            }
            LogicalOperator::Distinct { input } => {
                let input_plan = self.build_operator(ctx, input)?;
                LogicalPlanBuilder::from(input_plan)
                    .distinct()
                    .map_err(|e| self.plan_error("Failed to build distinct", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            LogicalOperator::Sort { input, sort_items } => {
                use datafusion::logical_expr::SortExpr;

                let input_plan = self.build_operator(ctx, input)?;

                // Convert sort items to DataFusion sort expressions
                let sort_exprs: Vec<SortExpr> = sort_items
                    .iter()
                    .map(|item| {
                        let expr = Self::to_df_value_expr(&item.expression);
                        let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
                        SortExpr {
                            expr,
                            asc,
                            nulls_first: true,
                        }
                    })
                    .collect();

                LogicalPlanBuilder::from(input_plan)
                    .sort(sort_exprs)
                    .map_err(|e| self.plan_error("Failed to build sort", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            LogicalOperator::Limit { input, count } => {
                let input_plan = self.build_operator(ctx, input)?;
                LogicalPlanBuilder::from(input_plan)
                    .limit(0, Some((*count) as usize))
                    .map_err(|e| self.plan_error("Failed to build limit", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            LogicalOperator::Offset { input, offset } => {
                let input_plan = self.build_operator(ctx, input)?;
                LogicalPlanBuilder::from(input_plan)
                    .limit((*offset) as usize, None)
                    .map_err(|e| self.plan_error("Failed to build offset", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
                properties,
                target_properties,
                ..
            } => self.build_expand(
                ctx,
                input,
                source_variable,
                target_variable,
                target_label,
                relationship_types,
                direction,
                properties,
                target_properties,
            ),
            LogicalOperator::VariableLengthExpand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                min_length,
                max_length,
                target_properties,
                ..
            } => self.build_variable_length_expand(
                ctx,
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                *min_length,
                *max_length,
                target_properties,
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
            // Catalog exists - check if label is registered
            if let Some(source) = cat.node_source(label) {
                // Get schema before moving source
                let schema = source.schema();
                let mut builder = LogicalPlanBuilder::scan(label, source, None).map_err(|e| {
                    self.plan_error(&format!("Failed to scan node source '{}'", label), e)
                })?;

                // Combine property filters into single predicate for efficiency
                if !properties.is_empty() {
                    let filter_exprs: Vec<Expr> = properties
                        .iter()
                        .map(|(k, v)| {
                            let lit_expr = Self::to_df_value_expr(
                                &crate::ast::ValueExpression::Literal(v.clone()),
                            );
                            Expr::BinaryExpr(BinaryExpr {
                                left: Box::new(col(k)),
                                op: Operator::Eq,
                                right: Box::new(lit_expr),
                            })
                        })
                        .collect();

                    // Combine with AND if multiple filters
                    let combined_filter = filter_exprs
                        .into_iter()
                        .reduce(|acc, expr| {
                            Expr::BinaryExpr(BinaryExpr {
                                left: Box::new(acc),
                                op: Operator::And,
                                right: Box::new(expr),
                            })
                        })
                        .unwrap();

                    builder = builder
                        .filter(combined_filter)
                        .map_err(|e| self.plan_error("Failed to apply property filters", e))?;
                }

                // Create qualified column aliases: variable__property
                // Optimization: Pre-allocate string capacity to reduce allocations
                let qualified_exprs: Vec<Expr> = schema
                    .fields()
                    .iter()
                    .map(|field| {
                        let mut qualified_name =
                            String::with_capacity(variable.len() + field.name().len() + 2);
                        qualified_name.push_str(variable);
                        qualified_name.push_str("__");
                        qualified_name.push_str(field.name());
                        col(field.name()).alias(&qualified_name)
                    })
                    .collect();

                // Add projection with qualified aliases
                builder = builder
                    .project(qualified_exprs)
                    .map_err(|e| self.plan_error("Failed to project qualified columns", e))?;

                return builder
                    .build()
                    .map_err(|e| self.plan_error("Failed to build scan plan", e));
            } else {
                // Catalog exists but label not found - fail fast
                return Err(crate::error::GraphError::ConfigError {
                    message: format!(
                        "Node label '{}' not found in catalog. \
                         Ensure the label is registered in your GraphConfig with .with_node_label()",
                        label
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        // No catalog attached - create empty source fallback for flexibility
        // This allows planners created with DataFusionPlanner::new() to work
        // without requiring a catalog, though they won't have actual data sources
        let empty_source = Arc::new(crate::source_catalog::SimpleTableSource::empty());
        let builder = LogicalPlanBuilder::scan(label, empty_source, None).map_err(|e| {
            self.plan_error(&format!("Failed to create table scan for '{}'", label), e)
        })?;

        builder
            .build()
            .map_err(|e| self.plan_error("Failed to build scan plan", e))
    }

    /// Build a relationship expansion (graph traversal) as a series of joins
    #[allow(clippy::too_many_arguments)]
    fn build_expand(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        target_label: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
        relationship_properties: &HashMap<String, crate::ast::PropertyValue>,
        target_properties: &HashMap<String, crate::ast::PropertyValue>,
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

        // Build relationship scan with qualified columns and property filters
        let rel_scan =
            self.build_relationship_scan(&rel_instance, rel_source, relationship_properties)?;

        // Join source node with relationship
        let source_params = SourceJoinParams {
            source_variable,
            rel_qualifier: &rel_instance.alias,
            node_id_field: &node_map.id_field,
            rel_map,
            direction,
        };
        let builder = self.join_source_to_relationship(left_plan, rel_scan, &source_params)?;

        // Join relationship with target node using the explicit target_label
        let target_node_map = self.config.node_mappings.get(target_label).ok_or_else(|| {
            crate::error::GraphError::ConfigError {
                message: format!("No mapping found for target label: {}", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        let target_params = TargetJoinParams {
            target_variable,
            rel_qualifier: &rel_instance.alias,
            node_map: target_node_map,
            rel_map,
            direction,
            target_properties,
        };
        self.join_relationship_to_target(builder, cat, ctx, &target_params)
    }

    /// Build a qualified relationship scan with property filters
    fn build_relationship_scan(
        &self,
        rel_instance: &RelationshipInstance,
        rel_source: Arc<dyn datafusion::logical_expr::TableSource>,
        relationship_properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        let rel_schema = rel_source.schema();
        let mut rel_builder = LogicalPlanBuilder::scan(&rel_instance.rel_type, rel_source, None)
            .map_err(|e| {
                self.plan_error(
                    &format!("Failed to scan relationship '{}'", rel_instance.rel_type),
                    e,
                )
            })?;

        // Apply relationship property filters (e.g., -[r {since: 2020}]->)
        for (k, v) in relationship_properties.iter() {
            let lit_expr = Self::to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
            let filter_expr = Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col(k)),
                op: Operator::Eq,
                right: Box::new(lit_expr),
            });
            rel_builder = rel_builder.filter(filter_expr).map_err(|e| {
                self.plan_error(
                    &format!("Failed to apply relationship filter on '{}'", k),
                    e,
                )
            })?;
        }

        // Use unique alias from rel_instance to avoid column conflicts
        let rel_qualified_exprs: Vec<Expr> = rel_schema
            .fields()
            .iter()
            .map(|field| {
                let qualified_name = format!("{}__{}", rel_instance.alias, field.name());
                col(field.name()).alias(&qualified_name)
            })
            .collect();

        rel_builder
            .project(rel_qualified_exprs)
            .map_err(|e| self.plan_error("Failed to project relationship columns", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build relationship scan", e))
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

        LogicalPlanBuilder::from(left_plan)
            .join(
                rel_scan,
                JoinType::Inner,
                (vec![qualified_left_key], vec![qualified_right_key]),
                None,
            )
            .map_err(|e| self.plan_error("Failed to join source to relationship", e))
    }

    /// Join relationship with target node scan
    fn join_relationship_to_target(
        &self,
        mut builder: LogicalPlanBuilder,
        cat: &Arc<dyn GraphSourceCatalog>,
        ctx: &PlanningContext,
        params: &TargetJoinParams,
    ) -> Result<LogicalPlan> {
        // Get the target label from the analysis (which now has the correct label from Expand)
        let Some(target_label) = ctx
            .analysis
            .var_to_label
            .get(params.target_variable)
            .cloned()
        else {
            return builder
                .build()
                .map_err(|e| self.plan_error("Failed to build plan (no target label)", e));
        };

        let Some(target_source) = cat.node_source(&target_label) else {
            return builder
                .build()
                .map_err(|e| self.plan_error("Failed to build plan (no target source)", e));
        };

        // Create target node scan with qualified column aliases and property filters
        let target_schema = target_source.schema();
        let mut target_builder = LogicalPlanBuilder::scan(&target_label, target_source, None)
            .map_err(|e| {
                self.plan_error(&format!("Failed to scan target node '{}'", target_label), e)
            })?;

        // Apply target property filters (e.g., (b {age: 30}))
        for (k, v) in params.target_properties.iter() {
            let lit_expr = Self::to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
            let filter_expr = Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col(k)),
                op: Operator::Eq,
                right: Box::new(lit_expr),
            });
            target_builder = target_builder.filter(filter_expr).map_err(|e| {
                self.plan_error(&format!("Failed to apply target filter on '{}'", k), e)
            })?;
        }

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
            .map_err(|e| self.plan_error("Failed to project target columns", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build target scan", e))?;

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
            .map_err(|e| self.plan_error("Failed to join relationship to target", e))?;

        builder
            .build()
            .map_err(|e| self.plan_error("Failed to build final join plan", e))
    }

    /// Get the expected qualified column names for variable-length path results
    ///
    /// Derives the column set from actual source and target node schemas rather than
    /// using fragile prefix matching. This prevents accidentally including intermediate
    /// node columns or missing renamed properties.
    fn get_expected_varlength_columns(
        &self,
        ctx: &PlanningContext,
        source_variable: &str,
        target_variable: &str,
    ) -> Result<std::collections::HashSet<String>> {
        use std::collections::HashSet;

        let mut expected = HashSet::new();

        let Some(cat) = &self.catalog else {
            return Ok(expected);
        };

        // Get source node label and schema
        if let Some(source_label) = ctx.analysis.var_to_label.get(source_variable) {
            if let Some(source) = cat.node_source(source_label) {
                let source_schema = source.schema();
                for field in source_schema.fields() {
                    let qualified_name = format!("{}__{}", source_variable, field.name());
                    expected.insert(qualified_name);
                }
            }
        }

        // Get target node label and schema
        if let Some(target_label) = ctx.analysis.var_to_label.get(target_variable) {
            if let Some(target) = cat.node_source(target_label) {
                let target_schema = target.schema();
                for field in target_schema.fields() {
                    let qualified_name = format!("{}__{}", target_variable, field.name());
                    expected.insert(qualified_name);
                }
            }
        }

        Ok(expected)
    }

    /// Build variable-length path expansion using unrolling + UNION strategy
    ///
    /// For a query like: (a)-[:KNOWS*1..3]->(b)
    /// This generates:
    ///   1-hop plan UNION 2-hop plan UNION 3-hop plan
    #[allow(clippy::too_many_arguments)]
    fn build_variable_length_expand(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
        min_length: Option<u32>,
        max_length: Option<u32>,
        target_properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        let min_hops = min_length.unwrap_or(1).max(1);
        let max_hops = max_length.unwrap_or(crate::MAX_VARIABLE_LENGTH_HOPS);

        // Validate range
        if min_hops > max_hops {
            return Err(crate::error::GraphError::InvalidPattern {
                message: format!(
                    "Invalid variable-length range: min {} > max {}",
                    min_hops, max_hops
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        if max_hops > crate::MAX_VARIABLE_LENGTH_HOPS {
            return Err(crate::error::GraphError::UnsupportedFeature {
                feature: format!(
                    "Variable-length paths with max length > {} (got {})",
                    crate::MAX_VARIABLE_LENGTH_HOPS,
                    max_hops
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Build the input plan (source node scan)
        let input_plan = self.build_operator(ctx, input)?;

        // Derive expected column names from source and target node schemas
        // This ensures we only project columns that actually belong to source/target nodes
        let expected_columns =
            self.get_expected_varlength_columns(ctx, source_variable, target_variable)?;

        // Generate a plan for each hop count and UNION them
        let mut plans = Vec::new();

        for hop_count in min_hops..=max_hops {
            let mut plan = self.build_fixed_length_path(
                ctx,
                input_plan.clone(),
                source_variable,
                target_variable,
                relationship_types,
                direction,
                hop_count,
                target_properties,
            )?;

            // Project only source and target columns to ensure consistent schema for UNION
            // This removes intermediate node columns that vary by hop count
            // Use the pre-computed expected column set derived from actual node schemas
            let projection: Vec<Expr> = plan
                .schema()
                .fields()
                .iter()
                .filter(|f| expected_columns.contains(f.name().as_str()))
                .map(|f| col(f.name()))
                .collect();

            plan = LogicalPlanBuilder::from(plan)
                .project(projection)
                .map_err(|e| crate::error::GraphError::PlanError {
                    message: format!("Failed to project for UNION: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?
                .build()
                .map_err(|e| crate::error::GraphError::PlanError {
                    message: format!("Failed to build projection: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            plans.push(plan);
        }

        // UNION all plans together
        if plans.len() == 1 {
            Ok(plans.into_iter().next().unwrap())
        } else {
            // Build UNION of all plans
            let mut union_plan = plans[0].clone();
            for plan in plans.into_iter().skip(1) {
                union_plan = LogicalPlanBuilder::from(union_plan)
                    .union(plan)
                    .map_err(|e| crate::error::GraphError::PlanError {
                        message: format!("Failed to UNION variable-length paths: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?
                    .build()
                    .map_err(|e| crate::error::GraphError::PlanError {
                        message: format!("Failed to build UNION plan: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
            }
            Ok(union_plan)
        }
    }

    /// Build a fixed-length path of N hops
    ///
    /// For hop_count=3: (a)-[:KNOWS]->(temp1)-[:KNOWS]->(temp2)-[:KNOWS]->(b)
    #[allow(clippy::too_many_arguments)]
    fn build_fixed_length_path(
        &self,
        ctx: &mut PlanningContext,
        input_plan: LogicalPlan,
        source_variable: &str,
        target_variable: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
        hop_count: u32,
        target_properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        let mut current_plan = input_plan;
        let mut current_source = source_variable.to_string();

        for hop_index in 0..hop_count {
            let is_last_hop = hop_index == hop_count - 1;

            // Target variable: use actual target on last hop, temp variable otherwise
            let current_target = if is_last_hop {
                target_variable.to_string()
            } else {
                format!("_temp_{}_{}", source_variable, hop_index + 1)
            };

            // Build the expansion on top of current plan
            // Apply target properties only on the last hop
            let props_to_apply = if is_last_hop {
                target_properties
            } else {
                &HashMap::new()
            };

            current_plan = self.build_expand_on_plan(
                ctx,
                current_plan,
                &current_source,
                &current_target,
                relationship_types,
                direction,
                props_to_apply,
            )?;

            // Move to next hop
            current_source = current_target;
        }

        Ok(current_plan)
    }

    /// Build a single-hop expansion on top of an existing plan
    #[allow(clippy::too_many_arguments)]
    fn build_expand_on_plan(
        &self,
        ctx: &mut PlanningContext,
        input_plan: LogicalPlan,
        source_variable: &str,
        target_variable: &str,
        relationship_types: &[String],
        direction: &RelationshipDirection,
        target_properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        let rel_type =
            relationship_types
                .first()
                .ok_or_else(|| crate::error::GraphError::InvalidPattern {
                    message: "Expand requires at least one relationship type".to_string(),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        let rel_instance = ctx.next_relationship_instance(rel_type)?;
        let rel_map = self.get_relationship_mapping(rel_type)?;
        let (target_label, node_map) = self.get_target_node_mapping(ctx, target_variable)?;
        let catalog = self.get_catalog()?;

        // Build relationship scan and join
        let rel_scan = self.build_qualified_relationship_scan(catalog, &rel_instance)?;
        let mut builder = self.join_relationship_to_input(
            input_plan,
            rel_scan,
            source_variable,
            &rel_instance,
            rel_map,
            node_map,
            direction,
        )?;

        // Build target node scan and join
        let target_scan = self.build_qualified_target_scan(
            catalog,
            &target_label,
            target_variable,
            target_properties,
        )?;
        builder = self.join_target_to_builder(
            builder,
            target_scan,
            target_variable,
            &rel_instance,
            rel_map,
            node_map,
            direction,
        )?;

        builder
            .build()
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to build expansion plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Get relationship mapping from config
    fn get_relationship_mapping(
        &self,
        rel_type: &str,
    ) -> Result<&crate::config::RelationshipMapping> {
        self.config
            .relationship_mappings
            .get(rel_type)
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: format!("No mapping found for relationship type: {}", rel_type),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Get target node mapping from context
    fn get_target_node_mapping(
        &self,
        ctx: &PlanningContext,
        target_variable: &str,
    ) -> Result<(String, &crate::config::NodeMapping)> {
        // Try to get label from analysis first
        let target_label = if let Some(label) = ctx.analysis.var_to_label.get(target_variable) {
            label.clone()
        } else if target_variable.starts_with("_temp_") {
            // For temporary variables in multi-hop paths (e.g., "_temp_a_1" or "_temp_foo_bar_1"),
            // infer the label from the source variable by extracting the base name
            // Format: _temp_{source}_{hop_index}
            // Note: source can contain underscores, so we reconstruct it from all parts
            // between the _temp prefix and the final hop index
            let parts: Vec<&str> = target_variable.split('_').collect();
            if parts.len() >= 4 {
                // parts[0] = "", parts[1] = "temp", parts[2..len-1] = source variable parts, parts[len-1] = hop index
                let source_var = parts[2..parts.len() - 1].join("_");
                ctx.analysis
                    .var_to_label
                    .get(&source_var)
                    .ok_or_else(|| crate::error::GraphError::ConfigError {
                        message: format!(
                            "Cannot infer label for temporary variable '{}' \
                             from source variable '{}'",
                            target_variable, source_var
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?
                    .clone()
            } else {
                return Err(crate::error::GraphError::ConfigError {
                    message: format!(
                        "Invalid temporary variable format: '{}'. \
                         Expected format: _temp_{{source}}_{{index}}",
                        target_variable
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        } else {
            // Not in analysis and not a temp variable - this is an error
            return Err(crate::error::GraphError::ConfigError {
                message: format!(
                    "Cannot determine target node label for variable '{}'. \
                     This variable was not found in the query analysis. \
                     Ensure the query properly defines this node variable.",
                    target_variable
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        };

        let node_map = self
            .config
            .node_mappings
            .get(&target_label)
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: format!("No mapping found for node label: {}", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        Ok((target_label, node_map))
    }

    /// Get catalog reference
    fn get_catalog(&self) -> Result<&Arc<dyn GraphSourceCatalog>> {
        self.catalog
            .as_ref()
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: "Catalog not available".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Build a qualified relationship scan for expansion
    fn build_qualified_relationship_scan(
        &self,
        catalog: &Arc<dyn GraphSourceCatalog>,
        rel_instance: &RelationshipInstance,
    ) -> Result<LogicalPlan> {
        let rel_source = catalog
            .relationship_source(&rel_instance.rel_type)
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: format!(
                    "No table source found for relationship: {}",
                    rel_instance.rel_type
                ),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let rel_schema = rel_source.schema();
        let rel_builder = LogicalPlanBuilder::scan(&rel_instance.rel_type, rel_source, None)
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to scan relationship: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let rel_qualified_exprs: Vec<Expr> = rel_schema
            .fields()
            .iter()
            .map(|field| {
                let qualified_name = format!("{}__{}", rel_instance.alias, field.name());
                col(field.name()).alias(&qualified_name)
            })
            .collect();

        rel_builder
            .project(rel_qualified_exprs)
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to project relationship: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .build()
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to build relationship scan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Get relationship join key based on direction (source side)
    fn get_source_join_key<'a>(
        direction: &RelationshipDirection,
        rel_map: &'a crate::config::RelationshipMapping,
    ) -> &'a str {
        match direction {
            RelationshipDirection::Outgoing => &rel_map.source_id_field,
            RelationshipDirection::Incoming => &rel_map.target_id_field,
            RelationshipDirection::Undirected => &rel_map.source_id_field,
        }
    }

    /// Get relationship join key based on direction (target side)
    fn get_target_join_key<'a>(
        direction: &RelationshipDirection,
        rel_map: &'a crate::config::RelationshipMapping,
    ) -> &'a str {
        match direction {
            RelationshipDirection::Outgoing => &rel_map.target_id_field,
            RelationshipDirection::Incoming => &rel_map.source_id_field,
            RelationshipDirection::Undirected => &rel_map.target_id_field,
        }
    }

    /// Join input plan with relationship scan
    #[allow(clippy::too_many_arguments)]
    fn join_relationship_to_input(
        &self,
        input_plan: LogicalPlan,
        rel_scan: LogicalPlan,
        source_variable: &str,
        rel_instance: &RelationshipInstance,
        rel_map: &crate::config::RelationshipMapping,
        node_map: &crate::config::NodeMapping,
        direction: &RelationshipDirection,
    ) -> Result<LogicalPlanBuilder> {
        let source_key = Self::get_source_join_key(direction, rel_map);
        let qualified_source_key = format!("{}__{}", source_variable, &node_map.id_field);
        let qualified_rel_source_key = format!("{}__{}", rel_instance.alias, source_key);

        LogicalPlanBuilder::from(input_plan)
            .join(
                rel_scan,
                JoinType::Inner,
                (vec![qualified_source_key], vec![qualified_rel_source_key]),
                None,
            )
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to join with relationship: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Build a qualified target node scan with property filters
    fn build_qualified_target_scan(
        &self,
        catalog: &Arc<dyn GraphSourceCatalog>,
        target_label: &str,
        target_variable: &str,
        target_properties: &HashMap<String, crate::ast::PropertyValue>,
    ) -> Result<LogicalPlan> {
        let target_source = catalog.node_source(target_label).ok_or_else(|| {
            crate::error::GraphError::ConfigError {
                message: format!("No table source found for node label: {}", target_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            }
        })?;

        let target_schema = target_source.schema();
        let mut target_builder = LogicalPlanBuilder::scan(target_label, target_source, None)
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to scan target node: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Apply target property filters
        for (k, v) in target_properties.iter() {
            let lit_expr = Self::to_df_value_expr(&crate::ast::ValueExpression::Literal(v.clone()));
            let filter_expr = Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col(k)),
                op: Operator::Eq,
                right: Box::new(lit_expr),
            });
            target_builder = target_builder.filter(filter_expr).map_err(|e| {
                crate::error::GraphError::PlanError {
                    message: format!("Failed to apply target property filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
        }

        let target_qualified_exprs: Vec<Expr> = target_schema
            .fields()
            .iter()
            .map(|field| {
                let qualified_name = format!("{}__{}", target_variable, field.name());
                col(field.name()).alias(&qualified_name)
            })
            .collect();

        target_builder
            .project(target_qualified_exprs)
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to project target node: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?
            .build()
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to build target scan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Join builder with target node scan
    #[allow(clippy::too_many_arguments)]
    fn join_target_to_builder(
        &self,
        builder: LogicalPlanBuilder,
        target_scan: LogicalPlan,
        target_variable: &str,
        rel_instance: &RelationshipInstance,
        rel_map: &crate::config::RelationshipMapping,
        node_map: &crate::config::NodeMapping,
        direction: &RelationshipDirection,
    ) -> Result<LogicalPlanBuilder> {
        let target_key = Self::get_target_join_key(direction, rel_map);
        let qualified_rel_target_key = format!("{}__{}", rel_instance.alias, target_key);
        let qualified_target_key = format!("{}__{}", target_variable, &node_map.id_field);

        builder
            .join(
                target_scan,
                JoinType::Inner,
                (vec![qualified_rel_target_key], vec![qualified_target_key]),
                None,
            )
            .map_err(|e| crate::error::GraphError::PlanError {
                message: format!("Failed to join with target node: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    // ============================================================================
    // Expression Translators
    // ============================================================================

    fn to_df_boolean_expr(expr: &crate::ast::BooleanExpression) -> Expr {
        use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO};
        match expr {
            BE::Comparison {
                left,
                operator,
                right,
            } => {
                let l = Self::to_df_value_expr(left);
                let r = Self::to_df_value_expr(right);
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
                let expr = Self::to_df_value_expr(expression);
                let list_exprs = list.iter().map(Self::to_df_value_expr).collect::<Vec<_>>();
                Expr::InList(DFInList::new(Box::new(expr), list_exprs, false))
            }
            BE::And(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Self::to_df_boolean_expr(l)),
                op: Operator::And,
                right: Box::new(Self::to_df_boolean_expr(r)),
            }),
            BE::Or(l, r) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Self::to_df_boolean_expr(l)),
                op: Operator::Or,
                right: Box::new(Self::to_df_boolean_expr(r)),
            }),
            BE::Not(inner) => Expr::Not(Box::new(Self::to_df_boolean_expr(inner))),
            BE::Exists(prop) => Expr::IsNotNull(Box::new(Self::to_df_value_expr(
                &crate::ast::ValueExpression::Property(prop.clone()),
            ))),
            _ => lit(true),
        }
    }

    /// Convert ValueExpression to DataFusion Expr
    fn to_df_value_expr(expr: &crate::ast::ValueExpression) -> Expr {
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
            VE::Function { name, args } => {
                // Handle aggregation functions
                match name.to_lowercase().as_str() {
                    "count" => {
                        if args.len() == 1 {
                            // Check for COUNT(*)
                            let arg_expr = if let VE::Variable(v) = &args[0] {
                                if v == "*" {
                                    lit(1)
                                } else {
                                    Self::to_df_value_expr(&args[0])
                                }
                            } else {
                                Self::to_df_value_expr(&args[0])
                            };

                            // Use DataFusion's count helper function
                            count(arg_expr)
                        } else {
                            // Invalid argument count - return placeholder
                            lit(0)
                        }
                    }
                    _ => {
                        // Unsupported function - return placeholder for now
                        lit(0)
                    }
                }
            }
            VE::Arithmetic { .. } => lit(0),
        }
    }

    /// Check if a ValueExpression contains an aggregate function
    fn contains_aggregate(expr: &crate::ast::ValueExpression) -> bool {
        use crate::ast::ValueExpression as VE;
        match expr {
            VE::Function { name, args } => {
                // Check if this is an aggregate function
                let is_aggregate = matches!(
                    name.to_lowercase().as_str(),
                    "count" | "sum" | "avg" | "min" | "max"
                );
                // Also check arguments recursively
                is_aggregate || args.iter().any(Self::contains_aggregate)
            }
            VE::Arithmetic { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            _ => false,
        }
    }

    /// Convert a ValueExpression to Cypher dot notation for column naming
    ///
    /// This generates user-friendly column names following Cypher conventions:
    /// - Property references: `p.name` (variable.property)
    /// - Functions: `function_name(arg)` with simplified argument representation
    /// - Other expressions: Use the expression as-is
    ///
    /// This is used when no explicit alias is provided in RETURN clauses.
    fn to_cypher_column_name(&self, expr: &crate::ast::ValueExpression) -> String {
        use crate::ast::ValueExpression as VE;
        match expr {
            VE::Property(prop) => {
                // Convert to Cypher dot notation: variable.property
                format!("{}.{}", prop.variable, prop.property)
            }
            VE::Variable(v) => v.clone(),
            VE::Literal(crate::ast::PropertyValue::Property(prop)) => {
                // Handle nested property references
                format!("{}.{}", prop.variable, prop.property)
            }
            VE::Function { name, args } => {
                // Generate descriptive function name: count(*), count(p.name), etc.
                if args.len() == 1 {
                    let arg_repr = match &args[0] {
                        VE::Variable(v) => v.clone(),
                        VE::Property(prop) => format!("{}.{}", prop.variable, prop.property),
                        _ => "expr".to_string(),
                    };
                    format!("{}({})", name.to_lowercase(), arg_repr)
                } else if args.is_empty() {
                    format!("{}()", name.to_lowercase())
                } else {
                    // Multiple args - just use function name
                    name.to_lowercase()
                }
            }
            _ => {
                // For other expressions (literals, arithmetic), use a generic name
                // In practice, these should always have explicit aliases
                "expr".to_string()
            }
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

        if let Expr::InList(in_list) = DataFusionPlanner::to_df_boolean_expr(&expr) {
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
    fn test_df_planner_inline_property_filter() {
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Incoming,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Undirected,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".into()),
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_properties: HashMap::new(),
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
    fn test_varlength_expand_single_hop() {
        // MATCH (a:Person)-[:KNOWS*1..1]->(b:Person) - equivalent to single hop
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
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(1),
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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

        // Should have joins but no UNION (only 1 hop)
        assert!(s.contains("Join("));
        // Single hop shouldn't have Union
        assert!(!s.contains("Union"));
    }

    #[test]
    fn test_varlength_expand_with_union() {
        // MATCH (a:Person)-[:KNOWS*2..3]->(b:Person) - should have UNION
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
            relationship_variable: None,
            min_length: Some(2),
            max_length: Some(3),
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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

        // Should have UNION for multiple hop counts
        assert!(s.contains("Union") || s.contains("union"));
        assert!(s.contains("Join("));
    }

    #[test]
    fn test_varlength_expand_default_min() {
        // MATCH (a:Person)-[:KNOWS*..3]->(b:Person) - min defaults to 1
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
            relationship_variable: None,
            min_length: None, // Should default to 1
            max_length: Some(3),
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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

        // Should build successfully with default min
        assert!(s.contains("Join("));
    }

    #[test]
    fn test_varlength_expand_default_max() {
        // MATCH (a:Person)-[:KNOWS*2..]->(b:Person) - max defaults to 20
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
            relationship_variable: None,
            min_length: Some(2),
            max_length: None, // Should default to MAX_VARIABLE_LENGTH_HOPS (20)
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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

        // Should build successfully with default max
        assert!(s.contains("Union") || s.contains("union"));
        assert!(s.contains("Join("));
    }

    #[test]
    fn test_varlength_expand_invalid_range() {
        // MATCH (a:Person)-[:KNOWS*3..2]->(b:Person) - min > max should error
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
            relationship_variable: None,
            min_length: Some(3),
            max_length: Some(2), // Invalid: min > max
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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
        let result = planner.plan(&project);

        // Should return error
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Invalid variable-length range"));
    }

    #[test]
    fn test_varlength_expand_exceeds_max() {
        // MATCH (a:Person)-[:KNOWS*1..25]->(b:Person) - exceeds MAX (20)
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
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(25), // Exceeds MAX_VARIABLE_LENGTH_HOPS
            target_properties: HashMap::new(),
        };
        let project = LogicalOperator::Project {
            input: Box::new(vlexpand),
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
        let result = planner.plan(&project);

        // Should return error
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Variable-length paths with max length > 20"));
    }

    #[test]
    fn test_varlength_expand_with_filter() {
        // MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) WHERE b.age > 30 RETURN b.name
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
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(2),
            target_properties: HashMap::new(),
        };
        let filter = LogicalOperator::Filter {
            input: Box::new(vlexpand),
            predicate: BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "b".into(),
                    property: "age".into(),
                }),
                operator: ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(PropertyValue::Integer(30)),
            },
        };
        let project = LogicalOperator::Project {
            input: Box::new(filter),
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

        // Should have filter and joins
        assert!(s.contains("Filter") || s.contains("filter"));
        assert!(s.contains("Join("));
    }

    #[test]
    fn test_varlength_expand_analysis_registers_instances() {
        // Test that analysis phase correctly registers multiple relationship instances
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
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(2),
            target_properties: HashMap::new(),
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::new(cfg);
        let analysis = planner.analyze(&vlexpand).unwrap();

        // For *1..2, should register 1 + 2 = 3 instances
        let knows_instances: Vec<_> = analysis
            .relationship_instances
            .iter()
            .filter(|r| r.rel_type == "KNOWS")
            .collect();

        assert_eq!(
            knows_instances.len(),
            3,
            "Should register 3 KNOWS instances for *1..2"
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
            target_label: "Person".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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
            target_label: "Person".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        };
        let expand2 = LogicalOperator::Expand {
            input: Box::new(expand1),
            source_variable: "b".into(),
            target_variable: "c".into(),
            target_label: "Person".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
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

    #[test]
    fn test_temp_variable_with_underscores_in_source() {
        // Test that temporary variables work correctly when source variable contains underscores
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Create a scan with a variable name containing underscores
        let scan = LogicalOperator::ScanByLabel {
            variable: "foo_bar".to_string(), // Variable with underscores
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let var_expand = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan),
            source_variable: "foo_bar".to_string(), // Will generate _temp_foo_bar_1
            target_variable: "baz".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            min_length: Some(2),
            max_length: Some(2),
            relationship_variable: None,
            target_properties: Default::default(),
        };

        let result = planner.plan(&var_expand);

        // Should succeed - the temp variable parsing should handle underscores correctly
        assert!(
            result.is_ok(),
            "Should handle source variables with underscores: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_cypher_dot_notation_simple_property() {
        // Test that projections without aliases use Cypher dot notation
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

        // Project without alias - should use Cypher dot notation
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef {
                    variable: "p".to_string(),
                    property: "name".to_string(),
                }),
                alias: None, // No explicit alias
            }],
        };

        let df_plan = planner.plan(&project).unwrap();
        let plan_str = format!("{:?}", df_plan);

        // Should contain Cypher dot notation "p.name", not "p__name"
        assert!(
            plan_str.contains("p.name"),
            "Plan should contain Cypher dot notation 'p.name': {}",
            plan_str
        );
        assert!(
            !plan_str.contains("p__name AS"),
            "Plan should not contain DataFusion qualified name 'p__name AS': {}",
            plan_str
        );
    }

    #[test]
    fn test_cypher_dot_notation_multiple_properties() {
        // Test multiple properties from the same variable
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

        // Project multiple properties without aliases
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "name".to_string(),
                    }),
                    alias: None,
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "age".to_string(),
                    }),
                    alias: None,
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let plan_str = format!("{:?}", df_plan);

        // Should contain both Cypher dot notations
        assert!(
            plan_str.contains("p.name"),
            "Plan should contain 'p.name': {}",
            plan_str
        );
        assert!(
            plan_str.contains("p.age"),
            "Plan should contain 'p.age': {}",
            plan_str
        );
    }

    #[test]
    fn test_cypher_dot_notation_mixed_with_and_without_alias() {
        // Test mix of aliased and non-aliased projections
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
                    alias: Some("full_name".to_string()), // Explicit alias
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".to_string(),
                        property: "age".to_string(),
                    }),
                    alias: None, // No alias - should use dot notation
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let plan_str = format!("{:?}", df_plan);

        // Should contain explicit alias
        assert!(
            plan_str.contains("full_name"),
            "Plan should contain explicit alias 'full_name': {}",
            plan_str
        );
        // Should contain Cypher dot notation for non-aliased property
        assert!(
            plan_str.contains("p.age"),
            "Plan should contain Cypher dot notation 'p.age': {}",
            plan_str
        );
    }

    // ========================================================================
    // Failure Scenario Tests
    // ========================================================================

    #[test]
    fn test_scan_missing_node_label_with_catalog_fails_fast() {
        // Test that when a catalog is attached, scanning a non-existent label fails fast
        // This catches typos and configuration issues at planning time
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "x".to_string(),
            label: "NonExistentLabel".to_string(), // This label doesn't exist in catalog
            properties: Default::default(),
        };

        let result = planner.plan(&scan);

        // Should return ConfigError with helpful message
        assert!(
            result.is_err(),
            "Should fail when catalog exists but label is missing"
        );
        match result {
            Err(crate::error::GraphError::ConfigError { message, .. }) => {
                assert!(
                    message.contains("NonExistentLabel"),
                    "Error should mention the missing label"
                );
                assert!(
                    message.contains("not found"),
                    "Error should indicate label not found"
                );
            }
            _ => panic!("Expected ConfigError for missing node label"),
        }
    }

    #[test]
    fn test_scan_without_catalog_uses_empty_source() {
        // Test that when no catalog is attached, scanning creates an empty source fallback
        // This allows DataFusionPlanner::new() to work without requiring a catalog
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::new(cfg); // No catalog attached

        let scan = LogicalOperator::ScanByLabel {
            variable: "x".to_string(),
            label: "AnyLabel".to_string(), // Any label works without catalog
            properties: Default::default(),
        };

        let result = planner.plan(&scan);

        // Should succeed with empty source fallback
        assert!(
            result.is_ok(),
            "Should succeed with empty source when no catalog attached"
        );
    }

    #[test]
    fn test_expand_with_missing_relationship() {
        // Test that expanding with non-existent relationship type handles gracefully
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let expand = LogicalOperator::Expand {
            input: Box::new(scan),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["NONEXISTENT_REL".to_string()], // Doesn't exist
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        };

        let result = planner.plan(&expand);

        // Should handle gracefully - either error or empty result
        // The key is no panic
        match result {
            Ok(_) => {} // Graceful handling
            Err(e) => {
                // Should be a PlanError
                assert!(matches!(e, crate::error::GraphError::PlanError { .. }));
            }
        }
    }

    #[test]
    fn test_filter_preserves_error_context() {
        // Test that filter errors include helpful context
        use crate::ast::{BooleanExpression, PropertyRef, ValueExpression};

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

        // Create a filter with a property reference
        let filter = LogicalOperator::Filter {
            input: Box::new(scan),
            predicate: BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".to_string(),
                    property: "age".to_string(),
                }),
                operator: crate::ast::ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(crate::ast::PropertyValue::Integer(30)),
            },
        };

        let result = planner.plan(&filter);

        // Should succeed - this tests that valid filters work
        assert!(result.is_ok(), "Valid filter should succeed");
    }

    #[test]
    fn test_variable_length_with_invalid_range() {
        // Test that invalid variable-length ranges are caught
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let var_expand = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            min_length: Some(5), // min > max
            max_length: Some(2),
            relationship_variable: None,
            target_properties: Default::default(),
        };

        let result = planner.plan(&var_expand);

        // Should return InvalidPattern error
        assert!(result.is_err(), "Invalid range should return error");
        match result {
            Err(crate::error::GraphError::InvalidPattern { message, .. }) => {
                assert!(message.contains("min"), "Error should mention min");
                assert!(message.contains("max"), "Error should mention max");
            }
            _ => panic!("Expected InvalidPattern error"),
        }
    }

    #[test]
    fn test_variable_length_exceeds_max_hops() {
        // Test that exceeding MAX_VARIABLE_LENGTH_HOPS is caught
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let var_expand = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            min_length: Some(1),
            max_length: Some(100), // Way too high
            relationship_variable: None,
            target_properties: Default::default(),
        };

        let result = planner.plan(&var_expand);

        // Should return UnsupportedFeature error
        assert!(result.is_err(), "Exceeding max hops should return error");
        match result {
            Err(crate::error::GraphError::UnsupportedFeature { feature, .. }) => {
                assert!(
                    feature.contains("Variable-length"),
                    "Error should mention variable-length"
                );
            }
            _ => panic!("Expected UnsupportedFeature error"),
        }
    }
}
