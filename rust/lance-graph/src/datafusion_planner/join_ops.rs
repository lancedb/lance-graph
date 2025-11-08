// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Join Operations
//!
//! Encapsulates the mechanics of wiring node scans to relationship scans
//! when building DataFusion logical plans. The helpers defined here hide the
//! join-key construction, filter push-downs, and aliasing conventions so that
//! higher-level planner code can remain focused on traversal semantics.

use super::analysis::{PlanningContext, RelationshipInstance};
use super::DataFusionPlanner;
use crate::ast::{PropertyValue, RelationshipDirection};
use crate::config::{NodeMapping, RelationshipMapping};
use crate::error::Result;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::{
    col, BinaryExpr, Expr, JoinType, LogicalPlan, LogicalPlanBuilder, Operator,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Parameters for joining source node to relationship
pub(crate) struct SourceJoinParams<'a> {
    pub source_variable: &'a str,
    pub rel_qualifier: &'a str,
    pub node_id_field: &'a str,
    pub rel_map: &'a RelationshipMapping,
    pub direction: &'a RelationshipDirection,
}

/// Parameters for joining relationship to target node
pub(crate) struct TargetJoinParams<'a> {
    pub target_variable: &'a str,
    pub rel_qualifier: &'a str,
    pub node_map: &'a NodeMapping,
    pub rel_map: &'a RelationshipMapping,
    pub direction: &'a RelationshipDirection,
    pub target_properties: &'a HashMap<String, PropertyValue>,
}

impl DataFusionPlanner {
    /// Join source node plan with relationship scan
    pub(crate) fn join_source_to_relationship(
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
    pub(crate) fn join_relationship_to_target(
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
            let lit_expr = super::expression::to_df_value_expr(
                &crate::ast::ValueExpression::Literal(v.clone()),
            );
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

    /// Get relationship join key based on direction (source side)
    pub(crate) fn get_source_join_key<'a>(
        direction: &RelationshipDirection,
        rel_map: &'a RelationshipMapping,
    ) -> &'a str {
        match direction {
            RelationshipDirection::Outgoing => &rel_map.source_id_field,
            RelationshipDirection::Incoming => &rel_map.target_id_field,
            RelationshipDirection::Undirected => &rel_map.source_id_field,
        }
    }

    /// Get relationship join key based on direction (target side)
    pub(crate) fn get_target_join_key<'a>(
        direction: &RelationshipDirection,
        rel_map: &'a RelationshipMapping,
    ) -> &'a str {
        match direction {
            RelationshipDirection::Outgoing => &rel_map.target_id_field,
            RelationshipDirection::Incoming => &rel_map.source_id_field,
            RelationshipDirection::Undirected => &rel_map.target_id_field,
        }
    }

    /// Join input plan with relationship scan
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn join_relationship_to_input(
        &self,
        input_plan: LogicalPlan,
        rel_scan: LogicalPlan,
        source_variable: &str,
        rel_instance: &RelationshipInstance,
        rel_map: &RelationshipMapping,
        node_map: &NodeMapping,
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

    /// Join builder with target node scan
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn join_target_to_builder(
        &self,
        builder: LogicalPlanBuilder,
        target_scan: LogicalPlan,
        target_variable: &str,
        rel_instance: &RelationshipInstance,
        rel_map: &RelationshipMapping,
        node_map: &NodeMapping,
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

    // Tests for join_ops

    #[test]
    fn test_expand_uses_qualified_join_keys_with_type_alias() {
        // MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name
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
        let scan_a = person_scan("a");
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

        let planner = DataFusionPlanner::with_catalog(person_knows_config(), make_catalog());
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
    fn test_where_rel_and_node_properties() {
        // WHERE r.src_person_id = 1 AND a.age > 30
        let scan_a = person_scan("a");
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
}
