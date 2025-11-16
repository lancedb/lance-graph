// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Helper utilities for plan building

use crate::datafusion_planner::DataFusionPlanner;
use crate::logical_plan::*;

impl DataFusionPlanner {
    /// Extract all variables from a logical operator tree
    ///
    /// Variables come from:
    /// - Node variables: ScanByLabel, Expand source/target
    /// - Relationship variables: Expand and VariableLengthExpand relationship_variable
    pub(crate) fn extract_variables(&self, op: &LogicalOperator) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_variables(op, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    /// Recursively collect variables from a logical operator
    ///
    /// Collects both node variables and relationship variables to support
    /// join key inference when patterns share relationship aliases.
    fn collect_variables(op: &LogicalOperator, vars: &mut Vec<String>) {
        match op {
            // Base case: ScanByLabel introduces a node variable
            LogicalOperator::ScanByLabel { variable, .. } => {
                vars.push(variable.clone());
            }
            // Unary operators: recurse into input
            LogicalOperator::Filter { input, .. } => {
                Self::collect_variables(input, vars);
            }
            LogicalOperator::Project { input, .. } => {
                Self::collect_variables(input, vars);
            }
            LogicalOperator::Distinct { input } => {
                Self::collect_variables(input, vars);
            }
            LogicalOperator::Sort { input, .. } => {
                Self::collect_variables(input, vars);
            }
            LogicalOperator::Limit { input, .. } => {
                Self::collect_variables(input, vars);
            }
            LogicalOperator::Offset { input, .. } => {
                Self::collect_variables(input, vars);
            }
            // Expand: recurse into input and add source, target, and relationship variables
            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                relationship_variable,
                ..
            } => {
                Self::collect_variables(input, vars);
                vars.push(source_variable.clone());
                vars.push(target_variable.clone());
                // Also collect relationship variable if present
                if let Some(rel_var) = relationship_variable {
                    vars.push(rel_var.clone());
                }
            }
            LogicalOperator::VariableLengthExpand {
                input,
                source_variable,
                target_variable,
                relationship_variable,
                ..
            } => {
                Self::collect_variables(input, vars);
                vars.push(source_variable.clone());
                vars.push(target_variable.clone());
                // Also collect relationship variable if present
                if let Some(rel_var) = relationship_variable {
                    vars.push(rel_var.clone());
                }
            }
            // Binary operator: recurse into both left and right
            LogicalOperator::Join { left, right, .. } => {
                Self::collect_variables(left, vars);
                Self::collect_variables(right, vars);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::RelationshipDirection;
    use crate::datafusion_planner::{
        test_fixtures::{make_catalog, person_config},
        DataFusionPlanner,
    };
    use crate::logical_plan::{JoinType, LogicalOperator};
    use std::collections::HashMap;

    #[test]
    fn test_collect_variables_includes_relationship_variables() {
        // Test that collect_variables now captures relationship variables
        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Build: (a:Person)-[r:KNOWS]->(b:Person)
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

        let vars = planner.extract_variables(&expand);

        // Should contain: a (source), b (target), r (relationship)
        assert!(
            vars.contains(&"a".to_string()),
            "Should contain source variable 'a'"
        );
        assert!(
            vars.contains(&"b".to_string()),
            "Should contain target variable 'b'"
        );
        assert!(
            vars.contains(&"r".to_string()),
            "Should contain relationship variable 'r'"
        );
    }

    #[test]
    fn test_extract_variables_deduplicates_and_sorts() {
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        // Left pattern: (a)-[r]->(b)
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let left_expand = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: Some("r".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
        };

        // Right pattern: (b)-[r2]->(c)
        let scan_b = LogicalOperator::ScanByLabel {
            variable: "b".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let right_expand = LogicalOperator::Expand {
            input: Box::new(scan_b),
            source_variable: "b".to_string(),
            target_variable: "c".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: Some("r2".to_string()),
            properties: Default::default(),
            target_properties: Default::default(),
        };

        let join_plan = LogicalOperator::Join {
            left: Box::new(left_expand),
            right: Box::new(right_expand),
            join_type: JoinType::Inner,
        };

        let vars = planner.extract_variables(&join_plan);

        assert_eq!(vars, vec!["a", "b", "c", "r", "r2"]);
    }

    #[test]
    fn test_extract_variables_handles_varlength_expand_without_relationship_var() {
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());

        let scan = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };

        let varlength = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan),
            source_variable: "a".into(),
            target_variable: "b".into(),
            relationship_types: vec!["KNOWS".into()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            min_length: Some(1),
            max_length: Some(3),
            target_properties: HashMap::new(),
        };

        let vars = planner.extract_variables(&varlength);

        assert_eq!(vars, vec!["a", "b"]);
    }
}
