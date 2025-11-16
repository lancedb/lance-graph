// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph traversal operations: Expand and Variable-Length Expand

use crate::ast::RelationshipDirection;
use crate::datafusion_planner::analysis::PlanningContext;
use crate::datafusion_planner::join_ops::{SourceJoinParams, TargetJoinParams};
use crate::datafusion_planner::DataFusionPlanner;
use crate::error::Result;
use crate::logical_plan::*;
use datafusion::logical_expr::{col, LogicalPlan, LogicalPlanBuilder};
use std::collections::HashMap;

impl DataFusionPlanner {
    /// Build a relationship expansion (graph traversal) as a series of joins
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_expand(
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
    pub(crate) fn build_variable_length_expand(
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
    pub(crate) fn build_fixed_length_path(
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
    pub(crate) fn build_expand_on_plan(
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
        test_fixtures::{make_catalog, person_knows_config, person_scan},
        DataFusionPlanner, GraphPhysicalPlanner,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use std::collections::HashMap;

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
}
