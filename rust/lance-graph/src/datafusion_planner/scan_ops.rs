// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Scan Operations
//!
//! Helpers for constructing table scans with qualified columns

use super::analysis::{PlanningContext, RelationshipInstance};
use crate::ast::PropertyValue;
use crate::error::Result;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::{col, BinaryExpr, Expr, LogicalPlan, LogicalPlanBuilder, Operator};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// Forward declare DataFusionPlanner to add methods to it
use super::DataFusionPlanner;

impl DataFusionPlanner {
    /// Build a qualified node scan with property filters and column aliasing
    pub(crate) fn build_scan(
        &self,
        _ctx: &PlanningContext,
        variable: &str,
        label: &str,
        properties: &HashMap<String, PropertyValue>,
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
                            let lit_expr = super::expression::to_df_value_expr(
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

    /// Build a qualified relationship scan with property filters
    pub(crate) fn build_relationship_scan(
        &self,
        rel_instance: &RelationshipInstance,
        rel_source: Arc<dyn datafusion::logical_expr::TableSource>,
        relationship_properties: &HashMap<String, PropertyValue>,
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
            let lit_expr = super::expression::to_df_value_expr(
                &crate::ast::ValueExpression::Literal(v.clone()),
            );
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

    /// Get the expected qualified column names for variable-length path results
    ///
    /// Derives the column set from actual source and target node schemas rather than
    /// using fragile prefix matching. This prevents accidentally including intermediate
    /// node columns or missing renamed properties.
    pub(crate) fn get_expected_varlength_columns(
        &self,
        ctx: &PlanningContext,
        source_variable: &str,
        target_variable: &str,
    ) -> Result<HashSet<String>> {
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

    /// Build a qualified relationship scan for expansion
    pub(crate) fn build_qualified_relationship_scan(
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

    /// Build a qualified target node scan with property filters
    pub(crate) fn build_qualified_target_scan(
        &self,
        catalog: &Arc<dyn GraphSourceCatalog>,
        target_label: &str,
        target_variable: &str,
        target_properties: &HashMap<String, PropertyValue>,
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
            let lit_expr = super::expression::to_df_value_expr(
                &crate::ast::ValueExpression::Literal(v.clone()),
            );
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
}

#[cfg(test)]
mod tests {
    use crate::ast::{PropertyRef, PropertyValue, ValueExpression};
    use crate::datafusion_planner::{
        test_fixtures::{make_catalog, person_config, person_scan},
        DataFusionPlanner, GraphPhysicalPlanner,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem};

    #[test]
    fn test_df_planner_inline_property_filter() {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );

        let mut scan = person_scan("n");
        if let LogicalOperator::ScanByLabel { properties, .. } = &mut scan {
            *properties = props;
        }

        let planner = DataFusionPlanner::with_catalog(person_config(), make_catalog());
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
    fn test_scan_aliasing_projects_variable_prefixed_columns() {
        // MATCH (n:Person) RETURN n.name
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
}
