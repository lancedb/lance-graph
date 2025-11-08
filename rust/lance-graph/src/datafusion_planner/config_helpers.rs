// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Configuration Resolution Helpers
//!
//! Provides lightweight accessors that resolve graph configuration
//! metadata and catalog handles needed during physical plan construction.

use super::analysis::PlanningContext;
use super::DataFusionPlanner;
use crate::config::{NodeMapping, RelationshipMapping};
use crate::error::Result;
use crate::source_catalog::GraphSourceCatalog;
use std::sync::Arc;

impl DataFusionPlanner {
    /// Get relationship mapping from config
    pub(crate) fn get_relationship_mapping(&self, rel_type: &str) -> Result<&RelationshipMapping> {
        self.config
            .relationship_mappings
            .get(rel_type)
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: format!("No mapping found for relationship type: {}", rel_type),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }

    /// Get target node mapping from context
    pub(crate) fn get_target_node_mapping(
        &self,
        ctx: &PlanningContext,
        target_variable: &str,
    ) -> Result<(String, &NodeMapping)> {
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
    pub(crate) fn get_catalog(&self) -> Result<&Arc<dyn GraphSourceCatalog>> {
        self.catalog
            .as_ref()
            .ok_or_else(|| crate::error::GraphError::ConfigError {
                message: "Catalog not available".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;
    use crate::datafusion_planner::analysis::{PlanningContext, QueryAnalysis};
    use crate::datafusion_planner::test_fixtures::make_catalog;

    fn planner_with_basic_config() -> DataFusionPlanner {
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src", "dst")
            .build()
            .unwrap();
        DataFusionPlanner::with_catalog(cfg, make_catalog())
    }

    #[test]
    fn test_get_relationship_mapping_success() {
        let planner = planner_with_basic_config();
        let mapping = planner.get_relationship_mapping("KNOWS").unwrap();
        assert_eq!(mapping.relationship_type, "KNOWS");
        assert_eq!(mapping.source_id_field, "src");
        assert_eq!(mapping.target_id_field, "dst");
    }

    #[test]
    fn test_get_relationship_mapping_missing() {
        let planner = planner_with_basic_config();
        let err = planner.get_relationship_mapping("FRIENDS").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("No mapping found for relationship type"));
        assert!(msg.contains("FRIENDS"));
    }

    #[test]
    fn test_get_target_node_mapping_with_existing_label() {
        let planner = planner_with_basic_config();
        let mut analysis = QueryAnalysis::default();
        analysis
            .var_to_label
            .insert("b".to_string(), "Person".to_string());
        let ctx = PlanningContext::new(&analysis);

        let (label, node_map) = planner
            .get_target_node_mapping(&ctx, "b")
            .expect("expected mapping");
        assert_eq!(label, "Person");
        assert_eq!(node_map.id_field, "id");
    }

    #[test]
    fn test_get_target_node_mapping_infers_temp_variable() {
        let planner = planner_with_basic_config();
        let mut analysis = QueryAnalysis::default();
        analysis
            .var_to_label
            .insert("a".to_string(), "Person".to_string());
        let ctx = PlanningContext::new(&analysis);

        let (label, node_map) = planner
            .get_target_node_mapping(&ctx, "_temp_a_1")
            .expect("expected mapping");
        assert_eq!(label, "Person");
        assert_eq!(node_map.id_field, "id");
    }

    #[test]
    fn test_get_target_node_mapping_invalid_temp_variable() {
        let planner = planner_with_basic_config();
        let analysis = QueryAnalysis::default();
        let ctx = PlanningContext::new(&analysis);

        let err = planner
            .get_target_node_mapping(&ctx, "_temp_invalid")
            .unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid temporary variable format"));
    }

    #[test]
    fn test_get_target_node_mapping_missing_label() {
        let planner = planner_with_basic_config();
        let mut analysis = QueryAnalysis::default();
        analysis
            .var_to_label
            .insert("a".to_string(), "Person".to_string());
        let ctx = PlanningContext::new(&analysis);

        let err = planner.get_target_node_mapping(&ctx, "c").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Cannot determine target node label"));
    }

    #[test]
    fn test_get_target_node_mapping_missing_config_entry() {
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let mut analysis = QueryAnalysis::default();
        analysis
            .var_to_label
            .insert("b".to_string(), "Organization".to_string());
        let ctx = PlanningContext::new(&analysis);

        let err = planner.get_target_node_mapping(&ctx, "b").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("No mapping found for node label"));
    }

    #[test]
    fn test_get_catalog_success() {
        let planner = planner_with_basic_config();
        let catalog = planner.get_catalog().unwrap();
        // Compare pointer string to avoid trait object issues
        let ptr_str = format!("{:p}", catalog);
        assert!(!ptr_str.is_empty());
    }

    #[test]
    fn test_get_catalog_missing() {
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();
        let planner = DataFusionPlanner::new(cfg);

        let err = match planner.get_catalog() {
            Ok(_) => panic!("expected catalog error"),
            Err(err) => err,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Catalog not available"));
    }
}
