//! SQL converter for logical plans
//!
//! This module converts our logical plan representation to optimized SQL
//! that can be executed by DataFusion or LanceDB for better query optimization.
//!
//! # Target Platforms
//! - **DataFusion**: Direct SQL execution via DataFusion's SQL interface
//! - **LanceDB**: Native SQL execution with potential for vector/full-text extensions

use crate::ast::{
    BooleanExpression, ComparisonOperator, PropertyRef, PropertyValue, RelationshipDirection,
    ValueExpression,
};
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use crate::logical_plan::{LogicalOperator, ProjectionItem, SortItem};
use std::collections::HashMap;

/// Converts logical plans to SQL for DataFusion or LanceDB execution
pub struct LogicalPlanToSqlConverter<'a> {
    config: &'a Option<GraphConfig>,
    variable_counter: u32,
    table_aliases: HashMap<String, String>,
}

impl<'a> LogicalPlanToSqlConverter<'a> {
    pub fn new(config: &'a Option<GraphConfig>) -> Self {
        Self {
            config,
            variable_counter: 0,
            table_aliases: HashMap::new(),
        }
    }

    /// Convert a logical plan to SQL compatible with DataFusion and LanceDB
    pub fn convert(&mut self, plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::Project { input, projections } => {
                self.convert_project(input, projections)
            }

            LogicalOperator::Filter { input, predicate } => self.convert_filter(input, predicate),

            LogicalOperator::ScanByLabel {
                variable,
                label,
                properties,
            } => self.convert_scan(variable, label, properties),

            LogicalOperator::Expand {
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                relationship_variable,
                properties,
                ..
            } => self.convert_expand(
                input,
                source_variable,
                target_variable,
                relationship_types,
                direction,
                relationship_variable,
                properties,
            ),

            LogicalOperator::Distinct { input } => self.convert_distinct(input),

            LogicalOperator::Limit { input, count } => self.convert_limit(input, *count as i64),

            LogicalOperator::Offset { input, offset } => self.convert_offset(input, *offset as i64),

            LogicalOperator::Sort { input, sort_items } => self.convert_sort(input, sort_items),

            LogicalOperator::VariableLengthExpand { .. } => Err(GraphError::PlanError {
                message: "Variable length paths not supported in SQL conversion".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),

            LogicalOperator::Join { .. } => Err(GraphError::PlanError {
                message: "Complex joins not supported in SQL conversion".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    fn convert_project(
        &mut self,
        input: &LogicalOperator,
        projections: &[ProjectionItem],
    ) -> Result<String> {
        let input_sql = self.convert(input)?;

        if projections.is_empty() {
            return Ok(format!("SELECT * FROM ({})", input_sql));
        }

        let proj_list = projections
            .iter()
            .map(|p| self.projection_to_sql(p))
            .collect::<Result<Vec<_>>>()?
            .join(", ");

        Ok(format!("SELECT {} FROM ({})", proj_list, input_sql))
    }

    fn convert_filter(
        &mut self,
        input: &LogicalOperator,
        predicate: &BooleanExpression,
    ) -> Result<String> {
        let input_sql = self.convert(input)?;
        let where_clause = self.boolean_expr_to_sql(predicate)?;
        Ok(format!(
            "SELECT * FROM ({}) WHERE {}",
            input_sql, where_clause
        ))
    }

    fn convert_scan(
        &mut self,
        variable: &str,
        label: &str,
        properties: &HashMap<String, PropertyValue>,
    ) -> Result<String> {
        // Store table alias for this variable - use the variable name as the alias
        self.table_aliases
            .insert(variable.to_string(), variable.to_string());

        let mut sql = format!("SELECT * FROM {} AS {}", label, variable);

        if !properties.is_empty() {
            let filters = properties
                .iter()
                .map(|(k, v)| {
                    Ok(format!(
                        "{}.{} = {}",
                        variable,
                        k,
                        self.property_value_to_sql(v)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .join(" AND ");
            sql = format!("{} WHERE {}", sql, filters);
        }

        Ok(sql)
    }

    #[allow(clippy::too_many_arguments)]
    fn convert_expand(
        &mut self,
        _input: &LogicalOperator,
        source_variable: &str,
        target_variable: &str,
        relationship_types: &[String],
        _direction: &RelationshipDirection,
        relationship_variable: &Option<String>,
        _properties: &HashMap<String, PropertyValue>,
    ) -> Result<String> {
        let _input_sql = self.convert(_input)?;
        let rel_type = relationship_types
            .first()
            .ok_or_else(|| GraphError::PlanError {
                message: "No relationship type specified".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        let config = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Config required for relationship queries".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        let _rel_mapping =
            config
                .get_relationship_mapping(rel_type)
                .ok_or_else(|| GraphError::PlanError {
                    message: format!("No relationship mapping for {}", rel_type),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        // Generate unique aliases
        let src_alias = format!("src_{}", self.variable_counter);
        let rel_alias = format!("rel_{}", self.variable_counter);
        let tgt_alias = format!("tgt_{}", self.variable_counter);
        self.variable_counter += 1;

        // Store aliases for variables
        self.table_aliases
            .insert(source_variable.to_string(), src_alias.clone());
        self.table_aliases
            .insert(target_variable.to_string(), tgt_alias.clone());
        if let Some(rel_var) = relationship_variable {
            self.table_aliases
                .insert(rel_var.clone(), rel_alias.clone());
        }

        // TODO: CRITICAL BUG - This code hardcodes "id" but should use configured ID fields
        // The problem is we need to track variable -> label mappings to get the right ID field
        // For now, we'll return an error since relationship queries are unsupported anyway
        Err(GraphError::PlanError {
            message: "Relationship traversal not supported in SQL conversion - would require proper ID field mapping".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    fn convert_distinct(&mut self, input: &LogicalOperator) -> Result<String> {
        let input_sql = self.convert(input)?;
        Ok(format!("SELECT DISTINCT * FROM ({})", input_sql))
    }

    fn convert_limit(&mut self, input: &LogicalOperator, count: i64) -> Result<String> {
        let input_sql = self.convert(input)?;
        Ok(format!("SELECT * FROM ({}) LIMIT {}", input_sql, count))
    }

    fn convert_offset(&mut self, input: &LogicalOperator, offset: i64) -> Result<String> {
        let input_sql = self.convert(input)?;
        Ok(format!("SELECT * FROM ({}) OFFSET {}", input_sql, offset))
    }

    fn convert_sort(
        &mut self,
        input: &LogicalOperator,
        _sort_items: &[SortItem],
    ) -> Result<String> {
        // For now, just pass through the input (ORDER BY is complex to implement)
        // TODO: Implement proper ORDER BY conversion
        self.convert(input)
    }

    fn projection_to_sql(&self, projection: &ProjectionItem) -> Result<String> {
        let expr_sql = self.value_expr_to_sql(&projection.expression)?;

        if let Some(alias) = &projection.alias {
            Ok(format!("{} AS {}", expr_sql, alias))
        } else {
            Ok(expr_sql)
        }
    }

    fn boolean_expr_to_sql(&self, expr: &BooleanExpression) -> Result<String> {
        match expr {
            BooleanExpression::Comparison {
                left,
                operator,
                right,
            } => {
                let left_sql = self.value_expr_to_sql(left)?;
                let right_sql = self.value_expr_to_sql(right)?;
                let op_sql = match operator {
                    ComparisonOperator::Equal => "=",
                    ComparisonOperator::NotEqual => "!=",
                    ComparisonOperator::LessThan => "<",
                    ComparisonOperator::LessThanOrEqual => "<=",
                    ComparisonOperator::GreaterThan => ">",
                    ComparisonOperator::GreaterThanOrEqual => ">=",
                };
                Ok(format!("{} {} {}", left_sql, op_sql, right_sql))
            }

            BooleanExpression::In { expression, list } => {
                let expr_sql = self.value_expr_to_sql(expression)?;
                let list_sql = list
                    .iter()
                    .map(|v| self.value_expr_to_sql(v))
                    .collect::<Result<Vec<_>>>()?
                    .join(", ");
                Ok(format!("{} IN ({})", expr_sql, list_sql))
            }

            BooleanExpression::And(left, right) => {
                let left_sql = self.boolean_expr_to_sql(left)?;
                let right_sql = self.boolean_expr_to_sql(right)?;
                Ok(format!("({}) AND ({})", left_sql, right_sql))
            }

            BooleanExpression::Or(left, right) => {
                let left_sql = self.boolean_expr_to_sql(left)?;
                let right_sql = self.boolean_expr_to_sql(right)?;
                Ok(format!("({}) OR ({})", left_sql, right_sql))
            }

            BooleanExpression::Not(inner) => {
                let inner_sql = self.boolean_expr_to_sql(inner)?;
                Ok(format!("NOT ({})", inner_sql))
            }

            BooleanExpression::Exists(prop) => {
                let prop_sql = self.property_ref_to_sql(prop)?;
                Ok(format!("{} IS NOT NULL", prop_sql))
            }

            _ => Err(GraphError::PlanError {
                message: "Unsupported boolean expression in SQL conversion".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    fn value_expr_to_sql(&self, expr: &ValueExpression) -> Result<String> {
        match expr {
            ValueExpression::Property(prop) => self.property_ref_to_sql(prop),
            ValueExpression::Variable(var) => Ok(var.clone()),
            ValueExpression::Literal(value) => self.property_value_to_sql(value),
            _ => Err(GraphError::PlanError {
                message: "Unsupported value expression in SQL conversion".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            }),
        }
    }

    fn property_ref_to_sql(&self, prop: &PropertyRef) -> Result<String> {
        if let Some(table_alias) = self.table_aliases.get(&prop.variable) {
            Ok(format!("{}.{}", table_alias, prop.property))
        } else {
            // Fallback to unqualified column name
            Ok(prop.property.clone())
        }
    }

    fn property_value_to_sql(&self, value: &PropertyValue) -> Result<String> {
        match value {
            PropertyValue::String(s) => Ok(format!("'{}'", s.replace('\'', "''"))), // Escape single quotes
            PropertyValue::Integer(i) => Ok(i.to_string()),
            PropertyValue::Float(f) => Ok(f.to_string()),
            PropertyValue::Boolean(b) => Ok(b.to_string()),
            PropertyValue::Null => Ok("NULL".to_string()),
            PropertyValue::Parameter(p) => Ok(format!("${}", p)), // Parameter placeholder
            PropertyValue::Property(prop) => self.property_ref_to_sql(prop),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BooleanExpression, ComparisonOperator, PropertyRef, ValueExpression};
    use crate::logical_plan::{LogicalOperator, ProjectionItem};
    use std::collections::HashMap;

    #[test]
    fn test_simple_scan_conversion() {
        let mut converter = LogicalPlanToSqlConverter::new(&None);

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: HashMap::new(),
        };

        let sql = converter.convert(&scan).unwrap();
        assert_eq!(sql, "SELECT * FROM Person AS n");
    }

    #[test]
    fn test_scan_with_properties() {
        let mut converter = LogicalPlanToSqlConverter::new(&None);

        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties,
        };

        let sql = converter.convert(&scan).unwrap();
        assert_eq!(sql, "SELECT * FROM Person AS n WHERE n.name = 'Alice'");
    }

    #[test]
    fn test_project_conversion() {
        let mut converter = LogicalPlanToSqlConverter::new(&None);

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: HashMap::new(),
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

        let sql = converter.convert(&project).unwrap();
        assert_eq!(sql, "SELECT n.name FROM (SELECT * FROM Person AS n)");
    }

    #[test]
    fn test_filter_conversion() {
        let mut converter = LogicalPlanToSqlConverter::new(&None);

        let scan = LogicalOperator::ScanByLabel {
            variable: "n".to_string(),
            label: "Person".to_string(),
            properties: HashMap::new(),
        };

        let filter = LogicalOperator::Filter {
            input: Box::new(scan),
            predicate: BooleanExpression::Comparison {
                left: ValueExpression::Property(PropertyRef {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                operator: ComparisonOperator::GreaterThan,
                right: ValueExpression::Literal(PropertyValue::Integer(30)),
            },
        };

        let sql = converter.convert(&filter).unwrap();
        assert_eq!(
            sql,
            "SELECT * FROM (SELECT * FROM Person AS n) WHERE n.age > 30"
        );
    }
}
