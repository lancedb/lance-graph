// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Basic operations: Filter, Project, Distinct, Sort, Limit, Offset

use crate::datafusion_planner::analysis::PlanningContext;
use crate::datafusion_planner::DataFusionPlanner;
use crate::error::Result;
use crate::logical_plan::*;
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder, SortExpr};

impl DataFusionPlanner {
    pub(crate) fn build_filter(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        predicate: &crate::ast::BooleanExpression,
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;
        let expr = super::super::expression::to_df_boolean_expr(predicate);
        LogicalPlanBuilder::from(input_plan)
            .filter(expr)
            .map_err(|e| self.plan_error("Failed to build filter", e))?
            .build()
            .map_err(|e| self.plan_error("Failed to build plan", e))
    }

    pub(crate) fn build_project(
        &self,
        ctx: &mut PlanningContext,
        input: &LogicalOperator,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        let input_plan = self.build_operator(ctx, input)?;

        // Check if any projection contains an aggregate function
        let has_aggregates = projections
            .iter()
            .any(|p| super::super::expression::contains_aggregate(&p.expression));

        if has_aggregates {
            self.build_project_with_aggregates(input_plan, projections)
        } else {
            self.build_simple_project(input_plan, projections)
        }
    }

    pub(crate) fn build_simple_project(
        &self,
        input_plan: LogicalPlan,
        projections: &[ProjectionItem],
    ) -> Result<LogicalPlan> {
        let exprs: Vec<datafusion::logical_expr::Expr> = projections
            .iter()
            .map(|p| {
                let expr = super::super::expression::to_df_value_expr(&p.expression);
                // Apply alias if provided, otherwise use Cypher dot notation
                if let Some(alias) = &p.alias {
                    expr.alias(alias)
                } else {
                    // Convert to Cypher dot notation (e.g., p__name -> p.name)
                    let cypher_name =
                        super::super::expression::to_cypher_column_name(&p.expression);
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

    pub(crate) fn build_distinct(
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

    pub(crate) fn build_sort(
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
                let expr = super::super::expression::to_df_value_expr(&item.expression);
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

    pub(crate) fn build_limit(
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

    pub(crate) fn build_offset(
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
}

#[cfg(test)]
mod tests {
    use crate::ast::{
        BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, SortDirection,
        ValueExpression,
    };
    use crate::datafusion_planner::{
        test_fixtures::{make_catalog, person_config, person_scan},
        DataFusionPlanner, GraphPhysicalPlanner,
    };
    use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

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
    fn test_distinct_and_order_with_qualified_columns() {
        let scan = person_scan("n");
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                alias: None,
            }],
        };
        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![SortItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                direction: SortDirection::Ascending,
            }],
        };
        let distinct = LogicalOperator::Distinct {
            input: Box::new(sort),
        };
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&distinct).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Distinct"), "plan missing Distinct: {}", s);
        assert!(s.contains("Sort"), "plan missing Sort: {}", s);
    }

    #[test]
    fn test_skip_limit_after_aliasing() {
        let scan = person_scan("n");
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                alias: Some("person_name".to_string()),
            }],
        };
        let limit = LogicalOperator::Limit {
            input: Box::new(project),
            count: 10,
        };
        let offset = LogicalOperator::Offset {
            input: Box::new(limit),
            offset: 5,
        };
        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&offset).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Limit"), "plan missing Limit: {}", s);
    }

    #[test]
    fn test_order_by_single_column_asc() {
        use crate::ast::{PropertyRef, SortDirection, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

        let scan = person_scan("n");
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                alias: None,
            }],
        };
        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![SortItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "age")),
                direction: SortDirection::Ascending,
            }],
        };

        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&sort).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Sort"), "plan missing Sort: {}", s);
    }

    #[test]
    fn test_order_by_multiple_columns() {
        let scan = person_scan("n");
        let project = LogicalOperator::Project {
            input: Box::new(scan),
            projections: vec![ProjectionItem {
                expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                alias: None,
            }],
        };
        let sort = LogicalOperator::Sort {
            input: Box::new(project),
            sort_items: vec![
                SortItem {
                    expression: ValueExpression::Property(PropertyRef::new("n", "name")),
                    direction: SortDirection::Ascending,
                },
                SortItem {
                    expression: ValueExpression::Property(PropertyRef::new("n", "age")),
                    direction: SortDirection::Descending,
                },
            ],
        };

        let cfg = person_config();
        let planner = DataFusionPlanner::with_catalog(cfg, make_catalog());
        let df_plan = planner.plan(&sort).unwrap();
        let s = format!("{:?}", df_plan);
        assert!(s.contains("Sort"), "plan missing Sort: {}", s);
    }

    #[test]
    fn test_order_by_with_limit() {
        use crate::ast::{PropertyRef, SortDirection, ValueExpression};
        use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};

        let cfg = person_config();
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

        let cfg = person_config();
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

        let cfg = person_config();
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

        let cfg = person_config();
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
        let cfg = person_config();
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
        let cfg = person_config();
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
        let cfg = person_config();
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
