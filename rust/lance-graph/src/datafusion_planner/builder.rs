// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan Building Phase
//!
//! Converts logical operators into DataFusion logical plans

use super::analysis::PlanningContext;
use super::join_ops::{SourceJoinParams, TargetJoinParams};
use super::DataFusionPlanner;
use crate::ast::RelationshipDirection;
use crate::error::Result;
use crate::logical_plan::*;
use datafusion::logical_expr::{col, LogicalPlan, LogicalPlanBuilder, SortExpr};
use std::collections::HashMap;

impl DataFusionPlanner {
    /// Phase 2: Build DataFusion LogicalPlan from logical operator with context
    pub(crate) fn build_operator(
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
                self.build_filter(ctx, input, predicate)
            }
            LogicalOperator::Project { input, projections } => {
                self.build_project(ctx, input, projections)
            }
            LogicalOperator::Distinct { input } => self.build_distinct(ctx, input),
            LogicalOperator::Sort { input, sort_items } => self.build_sort(ctx, input, sort_items),
            LogicalOperator::Limit { input, count } => self.build_limit(ctx, input, count),
            LogicalOperator::Offset { input, offset } => self.build_offset(ctx, input, offset),
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

    fn build_filter(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        predicate: &crate::ast::BooleanExpression,
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;
        let expr = super::expression::to_df_boolean_expr(predicate);
        LogicalPlanBuilder::from(input_plan)
            .filter(expr)
            .map_err(|e| self.plan_error("Failed to build filter", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build plan", e))
    }

    fn build_project(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;

        // Check if any projection contains an aggregate function
        let has_aggregates = projections
            .iter()
            .any(|p| super::expression::contains_aggregate(&p.expression));

        if has_aggregates {
            self.build_project_with_aggregates(input_plan, projections)
        } else {
            self.build_simple_project(input_plan, projections)
        }
    }

    fn build_project_with_aggregates(
        &self,
        input_plan: LogicalPlan,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        // Separate group expressions (non-aggregates) from aggregate expressions
        let mut group_exprs = Vec::new();
        let mut agg_exprs = Vec::new();
        // Store computed aliases for aggregates to reuse in final projection
        let mut agg_aliases = Vec::new();

        for p in projections {
            let expr = super::expression::to_df_value_expr(&p.expression);

            if super::expression::contains_aggregate(&p.expression) {
                // Aggregate expressions get aliased
                let alias = if let Some(alias) = &p.alias {
                    alias.clone()
                } else {
                    super::expression::to_cypher_column_name(&p.expression)
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
            if !super::expression::contains_aggregate(&p.expression) {
                // Re-create the expression and apply alias
                let expr = super::expression::to_df_value_expr(&p.expression);
                let aliased = if let Some(alias) = &p.alias {
                    expr.alias(alias)
                } else {
                    let cypher_name = super::expression::to_cypher_column_name(&p.expression);
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
    }

    fn build_simple_project(
        &self,
        input_plan: LogicalPlan,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        let exprs: Vec<datafusion::logical_expr::Expr> = projections
            .iter()
            .map(|p| {
                let expr = super::expression::to_df_value_expr(&p.expression);
                // Apply alias if provided, otherwise use Cypher dot notation
                if let Some(alias) = &p.alias {
                    expr.alias(alias)
                } else {
                    // Convert to Cypher dot notation (e.g., p__name -> p.name)
                    let cypher_name = super::expression::to_cypher_column_name(&p.expression);
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

    fn build_distinct(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;
        LogicalPlanBuilder::from(input_plan)
            .distinct()
            .map_err(|e| self.plan_error("Failed to build distinct", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build plan", e))
    }

    fn build_sort(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        sort_items: &[SortItem],
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;

        // Convert sort items to DataFusion sort expressions
        let sort_exprs: Vec<SortExpr> = sort_items
            .iter()
            .map(|item| {
                let expr = super::expression::to_df_value_expr(&item.expression);
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

    fn build_limit(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        count: &u64,
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;
        LogicalPlanBuilder::from(input_plan)
            .limit(0, Some((*count) as usize))
            .map_err(|e| self.plan_error("Failed to build limit", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build plan", e))
    }

    fn build_offset(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        offset: &u64,
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;
        LogicalPlanBuilder::from(input_plan)
            .limit((*offset) as usize, None)
            .map_err(|e| self.plan_error("Failed to build offset", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build plan", e))
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
            let projection: Vec<datafusion::logical_expr::Expr> = plan
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
}

#[cfg(test)]
mod tests {
    use crate::ast::{
        BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, ValueExpression,
    };
    use crate::datafusion_planner::{
        test_fixtures::{make_catalog, person_config, person_knows_config, person_scan},
        DataFusionPlanner, GraphPhysicalPlanner,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use std::collections::HashMap;

    #[test]
    fn test_df_planner_scan_filter_project() {
        let scan = person_scan("n");

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

        let cfg = person_config();
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
    fn test_df_planner_expand_creates_join_filter() {
        // MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let scan_a = person_scan("a");
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

        let cfg = person_knows_config();
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
    fn test_distinct_and_order_with_qualified_columns() {
        // ORDER is currently skipped in physical planner; just ensure Distinct appears and plan builds
        let scan = person_scan("n");
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
        let planner = DataFusionPlanner::with_catalog(person_config(), make_catalog());
        let df_plan = planner.plan(&distinct).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Distinct"), "missing Distinct in plan: {}", s);
    }

    #[test]
    fn test_skip_limit_after_aliasing() {
        let scan = person_scan("n");
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
        let planner = DataFusionPlanner::with_catalog(person_config(), make_catalog());
        let df_plan = planner.plan(&limit).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Limit"), "missing Limit in plan: {}", s);
    }

    #[test]
    fn test_varlength_expand_placeholder_builds() {
        // MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) RETURN a.name
        let scan_a = person_scan("a");
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
        let planner = DataFusionPlanner::with_catalog(person_knows_config(), make_catalog());
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
        let scan_a = person_scan("a");
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
        let planner = DataFusionPlanner::with_catalog(person_knows_config(), make_catalog());
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
        let scan_a = person_scan("a");
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
    fn test_project_with_aggregate_alias() {
        use crate::ast::{PropertyRef, ValueExpression};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = person_scan("p");

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Function {
                        name: "COUNT".into(),
                        args: vec![ValueExpression::Property(PropertyRef {
                            variable: "p".into(),
                            property: "id".into(),
                        })],
                    },
                    alias: Some("total_people".into()),
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".into(),
                        property: "name".into(),
                    }),
                    alias: None,
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let plan_str = format!("{:?}", df_plan);

        assert!(
            plan_str.contains("Aggregate"),
            "Plan should contain aggregate: {}",
            plan_str
        );
        assert!(
            plan_str.contains("total_people"),
            "Aggregate alias should appear in projection: {}",
            plan_str
        );
        assert!(
            plan_str.contains("p.name") || plan_str.contains("p__name"),
            "Grouped column should be projected: {}",
            plan_str
        );
    }

    #[test]
    fn test_project_with_aggregate_without_alias() {
        use crate::ast::{PropertyRef, ValueExpression};

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "p".into(),
            label: "Person".into(),
            properties: Default::default(),
        };

        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Function {
                        name: "COUNT".into(),
                        args: vec![ValueExpression::Property(PropertyRef {
                            variable: "p".into(),
                            property: "id".into(),
                        })],
                    },
                    alias: None,
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "p".into(),
                        property: "name".into(),
                    }),
                    alias: None,
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let plan_str = format!("{:?}", df_plan);

        assert!(
            plan_str.contains("Aggregate"),
            "Plan should contain aggregate: {}",
            plan_str
        );
        let count_str = "count(p.id)";
        assert!(
            plan_str.contains(count_str) || plan_str.contains(&count_str.to_lowercase()),
            "Default aggregate alias should be generated: {}",
            plan_str
        );
        assert!(
            plan_str.contains("p.name") || plan_str.contains("p__name"),
            "Grouped column should be projected: {}",
            plan_str
        );
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
        let scan = person_scan("n");

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

        let scan = person_scan("p");

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
}
