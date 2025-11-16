// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan Building Phase
//!
//! Converts logical operators into DataFusion logical plans
//!
//! This module is split into several submodules for better organization:
//! - `basic_ops`: Basic operations (filter, project, sort, limit, offset, distinct)
//! - `expand_ops`: Graph traversal operations (expand, variable-length expand)
//! - `aggregate_ops`: Aggregation and grouping operations
//! - `join_builder`: Join inference and building
//! - `helpers`: Utility functions

mod aggregate_ops;
mod basic_ops;
mod expand_ops;
mod helpers;
mod join_builder;

use super::DataFusionPlanner;
use crate::error::Result;
use crate::logical_plan::*;
use datafusion::logical_expr::LogicalPlan;

use super::analysis::PlanningContext;

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
            LogicalOperator::Join {
                left,
                right,
                join_type,
            } => self.build_join(ctx, left, right, join_type),
        }
    }
}
