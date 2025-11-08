// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query Analysis Phase
//!
//! Assigns unique IDs to relationship instances and collects variable-to-label mappings

use crate::ast::RelationshipDirection;
use crate::error::Result;
use crate::logical_plan::*;
use std::collections::{HashMap, HashSet};

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

/// Planning context that tracks state during plan building
pub struct PlanningContext<'a> {
    pub analysis: &'a QueryAnalysis,
    pub(crate) relationship_instance_idx: HashMap<String, usize>,
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

/// Analyze the logical plan to extract metadata
pub fn analyze(logical_plan: &LogicalOperator) -> Result<QueryAnalysis> {
    let mut analysis = QueryAnalysis::default();
    let mut rel_counter: HashMap<String, usize> = HashMap::new();

    analyze_operator(logical_plan, &mut analysis, &mut rel_counter)?;
    Ok(analysis)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::RelationshipDirection;
    use crate::logical_plan::LogicalOperator;
    use std::collections::HashMap;

    #[test]
    fn test_query_analysis_single_hop() {
        // Test that analysis correctly identifies relationship instances
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
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let _planner = crate::datafusion_planner::DataFusionPlanner::new(cfg);
        let analysis = analyze(&expand).unwrap();

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
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let expand1 = LogicalOperator::Expand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        };
        let expand2 = LogicalOperator::Expand {
            input: Box::new(expand1),
            source_variable: "b".to_string(),
            target_variable: "c".to_string(),
            target_label: "Person".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: RelationshipDirection::Outgoing,
            relationship_variable: None,
            properties: Default::default(),
            target_properties: Default::default(),
        };

        let cfg = crate::config::GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();
        let _planner = crate::datafusion_planner::DataFusionPlanner::new(cfg);
        let analysis = analyze(&expand2).unwrap();

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
    fn test_varlength_expand_analysis_registers_instances() {
        // Test that analysis phase correctly registers multiple relationship instances
        let scan_a = LogicalOperator::ScanByLabel {
            variable: "a".to_string(),
            label: "Person".to_string(),
            properties: Default::default(),
        };
        let vlexpand = LogicalOperator::VariableLengthExpand {
            input: Box::new(scan_a),
            source_variable: "a".to_string(),
            target_variable: "b".to_string(),
            relationship_types: vec!["KNOWS".to_string()],
            direction: RelationshipDirection::Outgoing,
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
        let _planner = crate::datafusion_planner::DataFusionPlanner::new(cfg);
        let analysis = analyze(&vlexpand).unwrap();

        // For *1..2, should register 1 + 2 = 3 instances
        let knows_instances: Vec<_> = analysis
            .relationship_instances
            .iter()
            .filter(|r| r.rel_type == "KNOWS")
            .collect();

        assert_eq!(
            knows_instances.len(),
            3,
            "Expected 3 KNOWS instances (1 for 1-hop + 2 for 2-hop)"
        );
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
                direction: RelationshipDirection::Outgoing,
                alias: "knows_1".to_string(),
            },
            RelationshipInstance {
                id: 2,
                rel_type: "KNOWS".to_string(),
                source_var: "b".to_string(),
                target_var: "c".to_string(),
                direction: RelationshipDirection::Outgoing,
                alias: "knows_2".to_string(),
            },
        ];

        let analysis = QueryAnalysis {
            var_to_label: HashMap::new(),
            relationship_instances: instances,
            required_datasets: HashSet::new(),
        };

        let mut ctx = PlanningContext::new(&analysis);

        // First call should return first instance
        let inst1 = ctx.next_relationship_instance("KNOWS").unwrap();
        assert_eq!(inst1.alias, "knows_1");

        // Second call should return second instance
        let inst2 = ctx.next_relationship_instance("KNOWS").unwrap();
        assert_eq!(inst2.alias, "knows_2");

        // Third call should error (no more instances)
        assert!(ctx.next_relationship_instance("KNOWS").is_err());
    }
}
