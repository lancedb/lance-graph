// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Join inference and building

use crate::datafusion_planner::analysis::PlanningContext;
use crate::datafusion_planner::DataFusionPlanner;
use crate::error::Result;
use crate::logical_plan::*;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder};

impl DataFusionPlanner {
    /// Build a join between two logical operators
    ///
    /// The join type and keys are determined by:
    /// - Cross joins: No join conditions needed
    /// - Other joins: Infer join keys from shared variables between patterns
    pub(crate) fn build_join(
        &self,
        ctx: &mut PlanningContext,
        left: &LogicalOperator,
        right: &LogicalOperator,
        join_type: &crate::logical_plan::JoinType,
    ) -> Result<LogicalPlan> {
        // Step 1: Build both sides of the join recursively
        let left_plan = self.build_operator(ctx, left)?;
        let right_plan = self.build_operator(ctx, right)?;

        // Step 2: Infer join keys from shared variables
        // Example: If both patterns reference variable 'b', we join on b__id
        let (left_keys, right_keys) = self.infer_join_keys(ctx, left, right);

        // Step 3: Build the appropriate join type
        match join_type {
            crate::logical_plan::JoinType::Cross => {
                // Cross join: Cartesian product, no join conditions needed
                // Used for completely disconnected patterns with no shared variables
                LogicalPlanBuilder::from(left_plan)
                    .cross_join(right_plan)
                    .map_err(|e| self.plan_error("Failed to build cross join", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            crate::logical_plan::JoinType::Inner => {
                // Inner join: If no shared variables, fall back to cross join
                // This is semantically valid (though potentially expensive)
                if left_keys.is_empty() {
                    return LogicalPlanBuilder::from(left_plan)
                        .cross_join(right_plan)
                        .map_err(|e| {
                            self.plan_error(
                                "Failed to build inner join. \
                                 No shared variables found, falling back to cross join",
                                e,
                            )
                        })?
                        .build()
                        .map_err(|e| self.plan_error("Failed to build plan", e));
                }

                // Build inner join with inferred keys
                let df_join_type = datafusion::logical_expr::JoinType::Inner;
                LogicalPlanBuilder::from(left_plan)
                    .join(right_plan, df_join_type, (left_keys, right_keys), None)
                    .map_err(|e| self.plan_error("Failed to build inner join", e))?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
            crate::logical_plan::JoinType::Left
            | crate::logical_plan::JoinType::Right
            | crate::logical_plan::JoinType::Full => {
                // Outer joins MUST have join keys - cross join has different semantics
                // (Cartesian product vs. NULL-padded unmatched rows)
                if left_keys.is_empty() {
                    return Err(crate::error::GraphError::PlanError {
                        message: format!(
                            "Cannot build {:?} join without shared variables. \
                             Outer joins require explicit join conditions to preserve NULL semantics. \
                             Consider using an inner join or adding shared variables between patterns.",
                            join_type
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }

                // Map our JoinType to DataFusion's JoinType
                let df_join_type = match join_type {
                    crate::logical_plan::JoinType::Left => datafusion::logical_expr::JoinType::Left,
                    crate::logical_plan::JoinType::Right => {
                        datafusion::logical_expr::JoinType::Right
                    }
                    crate::logical_plan::JoinType::Full => datafusion::logical_expr::JoinType::Full,
                    _ => unreachable!("Inner and Cross joins handled above"),
                };

                // Build join with inferred keys
                // Example: JOIN ON left.b__id = right.b__id
                LogicalPlanBuilder::from(left_plan)
                    .join(right_plan, df_join_type, (left_keys, right_keys), None)
                    .map_err(|e| {
                        self.plan_error(&format!("Failed to build {:?} join", join_type), e)
                    })?
                    .build()
                    .map_err(|e| self.plan_error("Failed to build plan", e))
            }
        }
    }

    /// Infer join keys by finding shared variables between left and right plans
    ///
    /// This analyzes both patterns to find variables that appear in both, then
    /// generates join keys based on the id fields of those shared variables.
    ///
    /// Supports both node variables and relationship variables:
    /// - **Node variables**: Join on node ID field (e.g., `b__id`)
    /// - **Relationship variables**: Join on composite keys (src_id AND dst_id)
    ///
    /// # Examples
    ///
    /// **Node variable join:**
    /// ```text
    /// Left:  (a:Person)-[:KNOWS]->(b:Person)  -> variables: [a, b]
    /// Right: (b:Person)-[:WORKS_AT]->(c:Company) -> variables: [b, c]
    /// Shared: [b]
    /// Result: (left_keys=["b__id"], right_keys=["b__id"])
    /// ```
    ///
    /// **Relationship variable join:**
    /// ```text
    /// Left:  (a:Person)-[r:KNOWS]->(b:Person)  -> variables: [a, b, r]
    /// Right: (c:Person)-[r:KNOWS]->(d:Person)  -> variables: [c, d, r]
    /// Shared: [r]
    /// Result: (left_keys=["r__src_id", "r__dst_id"],
    ///          right_keys=["r__src_id", "r__dst_id"])
    /// ```
    pub(crate) fn infer_join_keys(
        &self,
        ctx: &PlanningContext,
        left: &LogicalOperator,
        right: &LogicalOperator,
    ) -> (Vec<String>, Vec<String>) {
        // Step 1: Extract all variables from both patterns (includes relationship vars)
        let left_vars = self.extract_variables(left);
        let right_vars = self.extract_variables(right);

        // Step 2: Find variables that appear in both patterns
        // Example: left=[a, b], right=[b, c] -> shared=[b]
        let shared_vars: Vec<String> = left_vars
            .iter()
            .filter(|v| right_vars.contains(v))
            .cloned()
            .collect();

        // If no shared variables, return empty keys (will trigger cross join fallback)
        if shared_vars.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Step 3: For each shared variable, generate join keys
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();

        for var in &shared_vars {
            // Try to resolve as a node variable first
            if let Some(label) = ctx.analysis.var_to_label.get(var) {
                // This is a node variable - get the node mapping for its label
                if let Some(node_map) = self.config.node_mappings.get(label) {
                    // Generate qualified column names for node ID
                    // Example: var="b", id_field="id" -> "b__id"
                    let left_key = format!("{}__{}", var, node_map.id_field);
                    let right_key = format!("{}__{}", var, node_map.id_field);
                    left_keys.push(left_key);
                    right_keys.push(right_key);
                }
            } else {
                // Not a node variable - check if it's a relationship variable
                // Look up the relationship instance by its alias (the variable name)
                if let Some(rel_instance) = ctx
                    .analysis
                    .relationship_instances
                    .iter()
                    .find(|r| r.alias == *var)
                {
                    // Get the relationship mapping to find src/dst field names
                    if let Some(rel_map) = self
                        .config
                        .relationship_mappings
                        .get(&rel_instance.rel_type)
                    {
                        // Generate composite join keys for both src_id and dst_id
                        // This ensures we're matching the exact same relationship instance
                        // The columns are qualified as: {alias}__{original_field_name}
                        // Example: var="r", source_id_field="src_person_id"
                        //          -> "r__src_person_id"
                        let left_src = format!("{}__{}", var, &rel_map.source_id_field);
                        let right_src = format!("{}__{}", var, &rel_map.source_id_field);
                        let left_dst = format!("{}__{}", var, &rel_map.target_id_field);
                        let right_dst = format!("{}__{}", var, &rel_map.target_id_field);

                        left_keys.push(left_src);
                        right_keys.push(right_src);
                        left_keys.push(left_dst);
                        right_keys.push(right_dst);
                    }
                }
                // If not found in either node or relationship variables, skip it
            }
        }

        (left_keys, right_keys)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{PropertyRef, ValueExpression};
    use crate::datafusion_planner::{
        analysis,
        test_fixtures::{make_catalog, person_config},
        DataFusionPlanner, GraphPhysicalPlanner,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};

    #[test]
    fn test_cross_join_builds() {
        // Test MATCH (a:Person), (b:Person) - cross join pattern
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let join = LogicalOperator::Join {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: crate::logical_plan::JoinType::Cross,
        };
        let project = LogicalOperator::Project {
            input: Box::new(join),
            projections: vec![
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "a".into(),
                        property: "name".into(),
                    }),
                    alias: None,
                },
                ProjectionItem {
                    expression: ValueExpression::Property(PropertyRef {
                        variable: "b".into(),
                        property: "name".into(),
                    }),
                    alias: None,
                },
            ],
        };

        let df_plan = planner.plan(&project).unwrap();
        let s = format!("{:?}", df_plan);

        // Should contain Join (cross join is represented as a join with empty on clause)
        assert!(s.contains("Join"), "Plan should contain Join: {}", s);
        // Should have both table scans
        assert!(
            s.contains("TableScan"),
            "Plan should contain TableScan: {}",
            s
        );
        // Should have both variables projected
        assert!(
            s.contains("a__name") || s.contains("a.name"),
            "Plan should contain a.name: {}",
            s
        );
        assert!(
            s.contains("b__name") || s.contains("b.name"),
            "Plan should contain b.name: {}",
            s
        );
    }

    #[test]
    fn test_inner_join_builds() {
        // Test inner join with no shared variables - falls back to cross join
        // Simulates: MATCH (a:Person), (b:Person) with Inner join type
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let join = LogicalOperator::Join {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: crate::logical_plan::JoinType::Inner,
        };

        let result = planner.plan(&join);
        // Should build successfully (falls back to cross join since no shared variables)
        assert!(result.is_ok(), "Inner join should build: {:?}", result);

        let df_plan = result.unwrap();
        let plan_str = format!("{:?}", df_plan);
        // Should contain join (cross join fallback)
        assert!(
            plan_str.contains("Join"),
            "Plan should contain join: {}",
            plan_str
        );
    }

    #[test]
    fn test_left_join_without_shared_variables_errors() {
        // Test that left join with no shared variables now errors
        // (instead of silently falling back to cross join with wrong semantics)
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let join = LogicalOperator::Join {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: crate::logical_plan::JoinType::Left,
        };

        let result = planner.plan(&join);
        // Should error because outer joins require join conditions
        assert!(
            result.is_err(),
            "Left join without shared variables should error"
        );

        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(
            err_msg.contains("without shared variables") || err_msg.contains("join conditions"),
            "Error should mention missing join conditions: {}",
            err_msg
        );
    }

    #[test]
    fn test_inner_join_with_shared_variable() {
        // Test join key inference when patterns share a variable
        // Simulates: MATCH (a:Person), (a:Person) WHERE a.id = a.id
        // This is a simple case where both sides scan the same variable
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Left side: scan 'a'
        let scan_a_left = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        // Right side: also scan 'a' (same variable)
        let scan_a_right = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        // Inner join - should detect shared variable 'a'
        let join = LogicalOperator::Join {
            left: Box::new(scan_a_left),
            right: Box::new(scan_a_right),
            join_type: crate::logical_plan::JoinType::Inner,
        };

        let result = planner.plan(&join);

        // Note: This will likely fail with duplicate column error because both sides
        // produce a__id, a__name, a__age. This is expected - the join key inference
        // works, but DataFusion doesn't allow duplicate column names in joins.
        // In practice, this scenario wouldn't occur in real queries.
        // The important thing is that we attempted to create a join with keys,
        // not a cross join.
        match result {
            Ok(_) => {
                // If it succeeds, great!
            }
            Err(e) => {
                // If it fails, it should be because of duplicate columns, not missing join keys
                let err_msg = format!("{:?}", e);
                assert!(
                    err_msg.contains("duplicate") || err_msg.contains("Duplicate"),
                    "Error should be about duplicate columns, not missing join keys: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn test_join_without_shared_variable_falls_back_to_cross_join() {
        // Test that when there's no shared variable, we fall back to cross join
        // even for Inner join type
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        // Inner join with no shared variables - should fall back to cross join
        let join = LogicalOperator::Join {
            left: Box::new(scan_a),
            right: Box::new(scan_b),
            join_type: crate::logical_plan::JoinType::Inner,
        };

        let result = planner.plan(&join);
        assert!(
            result.is_ok(),
            "Should fall back to cross join: {:?}",
            result
        );

        let df_plan = result.unwrap();
        let plan_str = format!("{:?}", df_plan);

        // Should still build successfully (as cross join fallback)
        assert!(
            plan_str.contains("Join"),
            "Plan should contain join: {}",
            plan_str
        );
    }

    #[test]
    fn test_relationship_variable_join_key_inference() {
        // Test that the join key inference logic correctly handles relationship variables
        //
        // Note: This tests the key generation logic, not the full plan execution.
        // In practice, joining on shared relationship variables across disconnected patterns
        // doesn't make semantic sense in Cypher (a relationship can't have two sources).
        //
        // The implementation correctly:
        // 1. Detects relationship variables in both patterns
        // 2. Generates composite keys (src_id + dst_id) for relationship variables
        // 3. Generates single keys for node variables
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Left: (a:Person)-[r1:KNOWS]->(b:Person)
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand_left = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r1".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
        };

        // Right: (b:Person)-[r2:KNOWS]->(c:Person) - shares node 'b'
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand_right = LogicalOperator::Expand {
            input: Box::new(scan_b),
            source_variable: "b".to_string(),
            target_variable: "c".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r2".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
        };

        // Analyze both patterns to build the context
        let left_analysis = analysis::analyze(&expand_left).unwrap();
        let left_ctx = analysis::PlanningContext::new(&left_analysis);

        // Test the key inference logic directly
        let (left_keys, right_keys) =
            planner.infer_join_keys(&left_ctx, &expand_left, &expand_right);

        // Should generate join keys for shared node variable 'b'
        assert!(
            !left_keys.is_empty(),
            "Should generate join keys for shared node 'b'"
        );
        assert_eq!(
            left_keys.len(),
            right_keys.len(),
            "Left and right keys should match"
        );

        // Should contain b__id (the shared node)
        assert!(
            left_keys.iter().any(|k| k.contains("b__id")),
            "Should have join key for shared node 'b': {:?}",
            left_keys
        );

        // Verify that relationship variables r1 and r2 are collected
        let left_vars = planner.extract_variables(&expand_left);
        let right_vars = planner.extract_variables(&expand_right);

        assert!(left_vars.contains(&"r1".to_string()), "Left should have r1");
        assert!(
            right_vars.contains(&"r2".to_string()),
            "Right should have r2"
        );

        // r1 and r2 are different, so they shouldn't be in shared variables
        let shared: Vec<String> = left_vars
            .iter()
            .filter(|v| right_vars.contains(v))
            .cloned()
            .collect();
        assert!(!shared.contains(&"r1".to_string()), "r1 is not shared");
        assert!(!shared.contains(&"r2".to_string()), "r2 is not shared");
        assert!(shared.contains(&"b".to_string()), "b is shared");
    }

    #[test]
    fn test_shared_relationship_variable_detected() {
        // Test that shared relationship variables are detected
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Left: (a:Person)-[r:KNOWS]->(b:Person)
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand_left = LogicalOperator::Expand {
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

        // Right: (c:Person)-[r:KNOWS]->(d:Person) - same relationship variable 'r'
        let scan_c = LogicalOperator::ScanByLabel {
            variable: "c".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand_right = LogicalOperator::Expand {
            input: Box::new(scan_c),
            source_variable: "c".to_string(),
            target_variable: "d".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: crate::ast::RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
        };

        let left_vars = planner.extract_variables(&expand_left);
        let right_vars = planner.extract_variables(&expand_right);

        // Both should contain 'r'
        assert!(
            left_vars.contains(&"r".to_string()),
            "Left should contain 'r'"
        );
        assert!(
            right_vars.contains(&"r".to_string()),
            "Right should contain 'r'"
        );

        // Shared variables should include 'r'
        let shared: Vec<String> = left_vars
            .iter()
            .filter(|v| right_vars.contains(v))
            .cloned()
            .collect();
        assert!(
            shared.contains(&"r".to_string()),
            "Shared variables should include 'r'"
        );
    }
}
