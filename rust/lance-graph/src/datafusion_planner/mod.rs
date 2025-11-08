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
//! - Nodes -> Table scans, Relationships -> Linking tables, Traversals -> Joins
//! - Variable-length paths (`*1..3`) use unrolling: generate fixed-length plans + UNION
//! - All columns qualified as `{variable}__{column}` to avoid ambiguity

pub mod analysis;
mod builder;
mod config_helpers;
mod expression;
mod join_ops;
mod scan_ops;

#[cfg(test)]
mod test_fixtures;

// Re-export public types
pub use analysis::{PlanningContext, QueryAnalysis, RelationshipInstance};

use crate::config::GraphConfig;
use crate::error::Result;
use crate::logical_plan::LogicalOperator;
use crate::source_catalog::GraphSourceCatalog;
use datafusion::logical_expr::LogicalPlan;
use std::sync::Arc;

/// Planner abstraction for graph-to-physical planning
pub trait GraphPhysicalPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan>;
}

/// DataFusion-based physical planner
pub struct DataFusionPlanner {
    pub(crate) config: GraphConfig,
    pub(crate) catalog: Option<Arc<dyn GraphSourceCatalog>>,
}

impl DataFusionPlanner {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            catalog: None,
        }
    }

    pub fn with_catalog(config: GraphConfig, catalog: Arc<dyn GraphSourceCatalog>) -> Self {
        Self {
            config,
            catalog: Some(catalog),
        }
    }

    pub fn plan_with_context(
        &self,
        logical_plan: &LogicalOperator,
        datasets: &std::collections::HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<LogicalPlan> {
        use crate::source_catalog::{InMemoryCatalog, SimpleTableSource};

        // Use the analyze() method to extract metadata
        let analysis = analysis::analyze(logical_plan)?;

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

    /// Helper to convert DataFusion builder errors into GraphError::PlanError with context
    pub(crate) fn plan_error<E: std::fmt::Display>(
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

impl GraphPhysicalPlanner for DataFusionPlanner {
    fn plan(&self, logical_plan: &LogicalOperator) -> Result<LogicalPlan> {
        // Phase 1: Analyze query structure
        let analysis = analysis::analyze(logical_plan)?;

        // Phase 2: Build execution plan with context
        let mut ctx = PlanningContext::new(&analysis);
        self.build_operator(&mut ctx, logical_plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, RelationshipDirection,
        ValueExpression,
    };
    use crate::logical_plan::LogicalOperator;
    use test_fixtures::{make_catalog, person_config, person_knows_config, person_scan};

    #[test]
    fn test_filter_preserves_error_context() {
        // Test that filter errors include helpful context
        let planner = DataFusionPlanner::with_catalog(person_config(), make_catalog());

        let scan = person_scan("p");

        // Create a filter with a property reference
        let filter = LogicalOperator::Filter {
            input: Box::new(scan),
            predicate: BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "p".to_string(),
                    property: "age".to_string(),
                }),
                operator: ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(PropertyValue::Integer(30)),
            },
        };

        let result = planner.plan(&filter);

        // Should succeed - this tests that valid filters work
        assert!(result.is_ok(), "Valid filter should succeed");
    }

    #[test]
    fn test_exists_on_relationship_property_is_qualified() {
        // Test that EXISTS on relationship properties uses qualified column names
        let planner = DataFusionPlanner::with_catalog(person_knows_config(), make_catalog());

        let scan_a = person_scan("a");
        let expand = LogicalOperator::Expand {
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
        let pred = BooleanExpression::Exists(PropertyRef {
            variable: "r".into(),
            property: "src_person_id".into(),
        });
        let filter = LogicalOperator::Filter {
            input: Box::new(expand),
            predicate: pred,
        };
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
        // Test that IN lists on relationship properties use qualified column names
        let planner = DataFusionPlanner::with_catalog(person_knows_config(), make_catalog());

        let scan_a = person_scan("a");
        let expand = LogicalOperator::Expand {
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
    fn test_exists_and_in_on_node_props_materialized() {
        // Test that EXISTS and IN expressions on node properties work correctly
        let planner = DataFusionPlanner::with_catalog(person_config(), make_catalog());

        let scan_a = person_scan("a");
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
}
