// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Semantic analysis for graph queries
//!
//! This module implements the semantic analysis phase of the query pipeline:
//! Parse → **Semantic Analysis** → Logical Plan → Physical Plan
//!
//! Semantic analysis validates the query and enriches the AST with type information.

use crate::ast::*;
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};

/// Semantic analyzer - validates and enriches the AST
pub struct SemanticAnalyzer {
    config: GraphConfig,
    variables: HashMap<String, VariableInfo>,
    current_scope: ScopeType,
}

/// Information about a variable in the query
#[derive(Debug, Clone)]
pub struct VariableInfo {
    pub name: String,
    pub variable_type: VariableType,
    pub labels: Vec<String>,
    pub properties: HashSet<String>,
    pub defined_in: ScopeType,
}

/// Type of a variable
#[derive(Debug, Clone, PartialEq)]
pub enum VariableType {
    Node,
    Relationship,
    Path,
    Property,
}

/// Scope where a variable is defined
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    Match,
    Where,
    Return,
    OrderBy,
}

/// Semantic analysis result with validated and enriched AST
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub query: CypherQuery,
    pub variables: HashMap<String, VariableInfo>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl SemanticAnalyzer {
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            variables: HashMap::new(),
            current_scope: ScopeType::Match,
        }
    }

    /// Analyze a Cypher query AST
    pub fn analyze(&mut self, query: &CypherQuery) -> Result<SemanticResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Phase 1: Variable discovery in MATCH clauses
        self.current_scope = ScopeType::Match;
        for match_clause in &query.match_clauses {
            if let Err(e) = self.analyze_match_clause(match_clause) {
                errors.push(format!("MATCH clause error: {}", e));
            }
        }

        // Phase 2: Validate WHERE clause
        if let Some(where_clause) = &query.where_clause {
            self.current_scope = ScopeType::Where;
            if let Err(e) = self.analyze_where_clause(where_clause) {
                errors.push(format!("WHERE clause error: {}", e));
            }
        }

        // Phase 3: Validate RETURN clause
        self.current_scope = ScopeType::Return;
        if let Err(e) = self.analyze_return_clause(&query.return_clause) {
            errors.push(format!("RETURN clause error: {}", e));
        }

        // Phase 4: Validate ORDER BY clause
        if let Some(order_by) = &query.order_by {
            self.current_scope = ScopeType::OrderBy;
            if let Err(e) = self.analyze_order_by_clause(order_by) {
                errors.push(format!("ORDER BY clause error: {}", e));
            }
        }

        // Phase 5: Schema validation
        self.validate_schema(&mut warnings);

        // Phase 6: Type checking
        self.validate_types(&mut errors);

        Ok(SemanticResult {
            query: query.clone(),
            variables: self.variables.clone(),
            errors,
            warnings,
        })
    }

    /// Analyze MATCH clause and discover variables
    fn analyze_match_clause(&mut self, match_clause: &MatchClause) -> Result<()> {
        for pattern in &match_clause.patterns {
            self.analyze_graph_pattern(pattern)?;
        }
        Ok(())
    }

    /// Analyze a graph pattern and register variables
    fn analyze_graph_pattern(&mut self, pattern: &GraphPattern) -> Result<()> {
        match pattern {
            GraphPattern::Node(node) => {
                self.register_node_variable(node)?;
            }
            GraphPattern::Path(path) => {
                // Register start node
                self.register_node_variable(&path.start_node)?;

                // Register variables in each segment
                for segment in &path.segments {
                    // Validate relationship length constraints if present
                    self.validate_length_range(&segment.relationship)?;
                    // Register relationship variable if present
                    if let Some(rel_var) = &segment.relationship.variable {
                        self.register_relationship_variable(rel_var, &segment.relationship)?;
                    }

                    // Register end node
                    self.register_node_variable(&segment.end_node)?;
                }
            }
        }
        Ok(())
    }

    /// Register a node variable
    fn register_node_variable(&mut self, node: &NodePattern) -> Result<()> {
        if let Some(var_name) = &node.variable {
            if let Some(existing) = self.variables.get_mut(var_name) {
                if existing.variable_type != VariableType::Node {
                    return Err(GraphError::PlanError {
                        message: format!("Variable '{}' redefined with different type", var_name),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }
                for label in &node.labels {
                    if !existing.labels.contains(label) {
                        existing.labels.push(label.clone());
                    }
                }
                for prop in node.properties.keys() {
                    existing.properties.insert(prop.clone());
                }
            } else {
                let var_info = VariableInfo {
                    name: var_name.clone(),
                    variable_type: VariableType::Node,
                    labels: node.labels.clone(),
                    properties: node.properties.keys().cloned().collect(),
                    defined_in: self.current_scope.clone(),
                };
                self.variables.insert(var_name.clone(), var_info);
            }
        }
        Ok(())
    }

    /// Register a relationship variable
    fn register_relationship_variable(
        &mut self,
        var_name: &str,
        rel: &RelationshipPattern,
    ) -> Result<()> {
        if let Some(existing) = self.variables.get_mut(var_name) {
            if existing.variable_type != VariableType::Relationship {
                return Err(GraphError::PlanError {
                    message: format!("Variable '{}' redefined with different type", var_name),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
            for rel_type in &rel.types {
                if !existing.labels.contains(rel_type) {
                    existing.labels.push(rel_type.clone());
                }
            }
            for prop in rel.properties.keys() {
                existing.properties.insert(prop.clone());
            }
        } else {
            let var_info = VariableInfo {
                name: var_name.to_string(),
                variable_type: VariableType::Relationship,
                labels: rel.types.clone(), // Relationship types are like labels
                properties: rel.properties.keys().cloned().collect(),
                defined_in: self.current_scope.clone(),
            };
            self.variables.insert(var_name.to_string(), var_info);
        }
        Ok(())
    }

    /// Analyze WHERE clause
    fn analyze_where_clause(&mut self, where_clause: &WhereClause) -> Result<()> {
        self.analyze_boolean_expression(&where_clause.expression)
    }

    /// Analyze boolean expression and check variable references
    fn analyze_boolean_expression(&mut self, expr: &BooleanExpression) -> Result<()> {
        match expr {
            BooleanExpression::Comparison { left, right, .. } => {
                self.analyze_value_expression(left)?;
                self.analyze_value_expression(right)?;
            }
            BooleanExpression::And(left, right) | BooleanExpression::Or(left, right) => {
                self.analyze_boolean_expression(left)?;
                self.analyze_boolean_expression(right)?;
            }
            BooleanExpression::Not(inner) => {
                self.analyze_boolean_expression(inner)?;
            }
            BooleanExpression::Exists(prop_ref) => {
                self.validate_property_reference(prop_ref)?;
            }
            BooleanExpression::In { expression, list } => {
                self.analyze_value_expression(expression)?;
                for item in list {
                    self.analyze_value_expression(item)?;
                }
            }
            BooleanExpression::Like { expression, .. } => {
                self.analyze_value_expression(expression)?;
            }
            BooleanExpression::IsNull(expression) => {
                self.analyze_value_expression(expression)?;
            }
            BooleanExpression::IsNotNull(expression) => {
                self.analyze_value_expression(expression)?;
            }
        }
        Ok(())
    }

    /// Analyze value expression and check variable references
    fn analyze_value_expression(&mut self, expr: &ValueExpression) -> Result<()> {
        match expr {
            ValueExpression::Property(prop_ref) => {
                self.validate_property_reference(prop_ref)?;
            }
            ValueExpression::Literal(_) => {
                // Literals are always valid
            }
            ValueExpression::Variable(var) => {
                if !self.variables.contains_key(var) {
                    return Err(GraphError::PlanError {
                        message: format!("Undefined variable: '{}'", var),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }
            }
            ValueExpression::Function { name, args } => {
                // Validate function-specific arity and signature rules
                match name.to_lowercase().as_str() {
                    "count" | "sum" | "avg" | "min" | "max" => {
                        if args.len() != 1 {
                            return Err(GraphError::PlanError {
                                message: format!(
                                    "{} requires exactly 1 argument, got {}",
                                    name.to_uppercase(),
                                    args.len()
                                ),
                                location: snafu::Location::new(file!(), line!(), column!()),
                            });
                        }

                        // Additional validation for SUM, AVG, MIN, MAX: they require properties, not bare variables
                        // Only COUNT allows bare variables (COUNT(*) or COUNT(p))
                        if matches!(name.to_lowercase().as_str(), "sum" | "avg" | "min" | "max") {
                            if let Some(ValueExpression::Variable(v)) = args.first() {
                                return Err(GraphError::PlanError {
                                    message: format!(
                                        "{}({}) is invalid - {} requires a property like {}({}.property). You cannot {} a node/entity.",
                                        name.to_uppercase(), v, name.to_uppercase(), name.to_uppercase(), v, name.to_lowercase()
                                    ),
                                    location: snafu::Location::new(file!(), line!(), column!()),
                                });
                            }
                        }
                    }
                    _ => {
                        // Other functions - no validation yet
                    }
                }

                // Validate arguments recursively
                for arg in args {
                    self.analyze_value_expression(arg)?;
                }
            }
            ValueExpression::Arithmetic { left, right, .. } => {
                // Validate arithmetic operands recursively
                self.analyze_value_expression(left)?;
                self.analyze_value_expression(right)?;

                // If both sides are literals, ensure they are numeric
                let is_numeric_literal = |pv: &PropertyValue| {
                    matches!(pv, PropertyValue::Integer(_) | PropertyValue::Float(_))
                };

                if let (ValueExpression::Literal(l1), ValueExpression::Literal(l2)) =
                    (&**left, &**right)
                {
                    if !(is_numeric_literal(l1) && is_numeric_literal(l2)) {
                        return Err(GraphError::PlanError {
                            message: "Arithmetic requires numeric literal operands".to_string(),
                            location: snafu::Location::new(file!(), line!(), column!()),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate property reference
    fn validate_property_reference(&self, prop_ref: &PropertyRef) -> Result<()> {
        if !self.variables.contains_key(&prop_ref.variable) {
            return Err(GraphError::PlanError {
                message: format!("Undefined variable: '{}'", prop_ref.variable),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }
        Ok(())
    }

    /// Analyze RETURN clause
    fn analyze_return_clause(&mut self, return_clause: &ReturnClause) -> Result<()> {
        for item in &return_clause.items {
            self.analyze_value_expression(&item.expression)?;
        }
        Ok(())
    }

    /// Analyze ORDER BY clause
    fn analyze_order_by_clause(&mut self, order_by: &OrderByClause) -> Result<()> {
        for item in &order_by.items {
            self.analyze_value_expression(&item.expression)?;
        }
        Ok(())
    }

    /// Validate schema references against configuration
    fn validate_schema(&self, warnings: &mut Vec<String>) {
        for var_info in self.variables.values() {
            match var_info.variable_type {
                VariableType::Node => {
                    for label in &var_info.labels {
                        if self.config.get_node_mapping(label).is_none() {
                            warnings.push(format!("Node label '{}' not found in schema", label));
                        }
                    }
                }
                VariableType::Relationship => {
                    for rel_type in &var_info.labels {
                        if self.config.get_relationship_mapping(rel_type).is_none() {
                            warnings.push(format!(
                                "Relationship type '{}' not found in schema",
                                rel_type
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Validate types and operations
    fn validate_types(&self, errors: &mut Vec<String>) {
        // TODO: Implement type checking
        // - Check that properties exist on nodes/relationships
        // - Check that comparison operations are valid for data types
        // - Check that arithmetic operations are valid

        // Check that properties referenced in patterns exist in schema when property fields are defined
        for var_info in self.variables.values() {
            match var_info.variable_type {
                VariableType::Node => {
                    // Collect property_fields from all known label mappings that specify properties
                    let mut label_property_sets: Vec<&[String]> = Vec::new();
                    for label in &var_info.labels {
                        if let Some(mapping) = self.config.get_node_mapping(label) {
                            if !mapping.property_fields.is_empty() {
                                label_property_sets.push(&mapping.property_fields);
                            }
                        }
                    }

                    if !label_property_sets.is_empty() {
                        'prop: for prop in &var_info.properties {
                            // Property is valid if present in at least one label's property_fields
                            for fields in &label_property_sets {
                                if fields.iter().any(|f| f == prop) {
                                    continue 'prop;
                                }
                            }
                            errors.push(format!(
                                "Property '{}' not found on labels {:?}",
                                prop, var_info.labels
                            ));
                        }
                    }
                }
                VariableType::Relationship => {
                    // Collect property_fields from all known relationship mappings that specify properties
                    let mut rel_property_sets: Vec<&[String]> = Vec::new();
                    for rel_type in &var_info.labels {
                        if let Some(mapping) = self.config.get_relationship_mapping(rel_type) {
                            if !mapping.property_fields.is_empty() {
                                rel_property_sets.push(&mapping.property_fields);
                            }
                        }
                    }

                    if !rel_property_sets.is_empty() {
                        'prop_rel: for prop in &var_info.properties {
                            for fields in &rel_property_sets {
                                if fields.iter().any(|f| f == prop) {
                                    continue 'prop_rel;
                                }
                            }
                            errors.push(format!(
                                "Property '{}' not found on relationship types {:?}",
                                prop, var_info.labels
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

impl SemanticAnalyzer {
    fn validate_length_range(&self, rel: &RelationshipPattern) -> Result<()> {
        if let Some(len) = &rel.length {
            if let (Some(min), Some(max)) = (len.min, len.max) {
                if min > max {
                    return Err(GraphError::PlanError {
                        message: "Invalid path length range: min > max".to_string(),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        ArithmeticOperator, BooleanExpression, CypherQuery, GraphPattern, LengthRange, MatchClause,
        NodePattern, PathPattern, PathSegment, PropertyRef, PropertyValue, RelationshipDirection,
        RelationshipPattern, ReturnClause, ReturnItem, ValueExpression, WhereClause,
    };
    use crate::config::{GraphConfig, NodeMapping};

    fn test_config() -> GraphConfig {
        GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_node_label("Employee", "id")
            .with_node_label("Company", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap()
    }

    // Helper: analyze a query that only has a single RETURN expression
    fn analyze_return_expr(expr: ValueExpression) -> Result<SemanticResult> {
        let query = CypherQuery {
            match_clauses: vec![],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expression: expr,
                    alias: None,
                }],
            },
            limit: None,
            order_by: None,
            skip: None,
        };
        let mut analyzer = SemanticAnalyzer::new(test_config());
        analyzer.analyze(&query)
    }

    // Helper: analyze a query with a single MATCH (var:label) and a RETURN expression
    fn analyze_return_with_match(
        var: &str,
        label: &str,
        expr: ValueExpression,
    ) -> Result<SemanticResult> {
        let node = NodePattern::new(Some(var.to_string())).with_label(label);
        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Node(node)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![ReturnItem {
                    expression: expr,
                    alias: None,
                }],
            },
            limit: None,
            order_by: None,
            skip: None,
        };
        let mut analyzer = SemanticAnalyzer::new(test_config());
        analyzer.analyze(&query)
    }

    #[test]
    fn test_merge_node_variable_metadata() {
        // MATCH (n:Person {age: 30}), (n:Employee {dept: "X"})
        let node1 = NodePattern::new(Some("n".to_string()))
            .with_label("Person")
            .with_property("age", PropertyValue::Integer(30));
        let node2 = NodePattern::new(Some("n".to_string()))
            .with_label("Employee")
            .with_property("dept", PropertyValue::String("X".to_string()));

        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Node(node1), GraphPattern::Node(node2)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result.errors.is_empty());
        let n = result.variables.get("n").expect("variable n present");
        // Labels merged
        assert!(n.labels.contains(&"Person".to_string()));
        assert!(n.labels.contains(&"Employee".to_string()));
        // Properties unioned
        assert!(n.properties.contains("age"));
        assert!(n.properties.contains("dept"));
    }

    #[test]
    fn test_invalid_length_range_collects_error() {
        let start = NodePattern::new(Some("a".to_string())).with_label("Person");
        let end = NodePattern::new(Some("b".to_string())).with_label("Person");
        let mut rel = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("r")
            .with_type("KNOWS");
        rel.length = Some(LengthRange {
            min: Some(3),
            max: Some(2),
        });

        let path = PathPattern {
            start_node: start,
            segments: vec![PathSegment {
                relationship: rel,
                end_node: end,
            }],
        };

        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Path(path)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Invalid path length range")));
    }

    #[test]
    fn test_undefined_variable_in_where() {
        // MATCH (n:Person) WHERE EXISTS(m.name)
        let node = NodePattern::new(Some("n".to_string())).with_label("Person");
        let where_clause = WhereClause {
            expression: BooleanExpression::Exists(PropertyRef::new("m", "name")),
        };
        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Node(node)],
            }],
            where_clause: Some(where_clause),
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Undefined variable: 'm'")));
    }

    #[test]
    fn test_variable_redefinition_between_node_and_relationship() {
        // MATCH (n:Person)-[n:KNOWS]->(m:Person)
        let start = NodePattern::new(Some("n".to_string())).with_label("Person");
        let end = NodePattern::new(Some("m".to_string())).with_label("Person");
        let rel = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("n")
            .with_type("KNOWS");

        let path = PathPattern {
            start_node: start,
            segments: vec![PathSegment {
                relationship: rel,
                end_node: end,
            }],
        };

        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Path(path)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("redefined with different type")));
    }

    #[test]
    fn test_unknown_node_label_warns() {
        // MATCH (x:Unknown)
        let node = NodePattern::new(Some("x".to_string())).with_label("Unknown");
        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Node(node)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Node label 'Unknown' not found in schema")));
    }

    #[test]
    fn test_property_not_in_schema_reports_error() {
        // Configure Person with allowed property 'name' only
        let custom_config = GraphConfig::builder()
            .with_node_mapping(
                NodeMapping::new("Person", "id").with_properties(vec!["name".to_string()]),
            )
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();

        // MATCH (n:Person {age: 30})
        let node = NodePattern::new(Some("n".to_string()))
            .with_label("Person")
            .with_property("age", PropertyValue::Integer(30));
        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Node(node)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(custom_config);
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Property 'age' not found on labels [\"Person\"]")));
    }

    #[test]
    fn test_valid_length_range_ok() {
        let start = NodePattern::new(Some("a".to_string())).with_label("Person");
        let end = NodePattern::new(Some("b".to_string())).with_label("Person");
        let mut rel = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("r")
            .with_type("KNOWS");
        rel.length = Some(LengthRange {
            min: Some(2),
            max: Some(3),
        });

        let path = PathPattern {
            start_node: start,
            segments: vec![PathSegment {
                relationship: rel,
                end_node: end,
            }],
        };

        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Path(path)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(test_config());
        let result = analyzer.analyze(&query).unwrap();
        assert!(result
            .errors
            .iter()
            .all(|e| !e.contains("Invalid path length range")));
    }

    #[test]
    fn test_relationship_variable_metadata_merge_across_segments() {
        // Path with two segments sharing the same relationship variable 'r'
        // (a:Person)-[r:KNOWS {since: 2020}]->(b:Person)-[r:FRIEND {level: 1}]->(c:Person)
        let start = NodePattern::new(Some("a".to_string())).with_label("Person");
        let mid = NodePattern::new(Some("b".to_string())).with_label("Person");
        let end = NodePattern::new(Some("c".to_string())).with_label("Person");

        let mut rel1 = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("r")
            .with_type("KNOWS")
            .with_property("since", PropertyValue::Integer(2020));
        rel1.length = None;

        let mut rel2 = RelationshipPattern::new(RelationshipDirection::Outgoing)
            .with_variable("r")
            .with_type("FRIEND")
            .with_property("level", PropertyValue::Integer(1));
        rel2.length = None;

        let path = PathPattern {
            start_node: start,
            segments: vec![
                PathSegment {
                    relationship: rel1,
                    end_node: mid,
                },
                PathSegment {
                    relationship: rel2,
                    end_node: end,
                },
            ],
        };

        // Custom config that knows both relationship types to avoid warnings muddying the assertion
        let custom_config = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .with_relationship("FRIEND", "src_id", "dst_id")
            .build()
            .unwrap();

        let query = CypherQuery {
            match_clauses: vec![MatchClause {
                patterns: vec![GraphPattern::Path(path)],
            }],
            where_clause: None,
            return_clause: ReturnClause {
                distinct: false,
                items: vec![],
            },
            limit: None,
            order_by: None,
            skip: None,
        };

        let mut analyzer = SemanticAnalyzer::new(custom_config);
        let result = analyzer.analyze(&query).unwrap();
        let r = result.variables.get("r").expect("variable r present");
        // Types merged
        assert!(r.labels.contains(&"KNOWS".to_string()));
        assert!(r.labels.contains(&"FRIEND".to_string()));
        // Properties unioned
        assert!(r.properties.contains("since"));
        assert!(r.properties.contains("level"));
    }

    #[test]
    fn test_function_argument_undefined_variable_in_return() {
        // RETURN toUpper(m.name)
        let expr = ValueExpression::Function {
            name: "toUpper".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("m", "name"))],
        };
        let result = analyze_return_expr(expr).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Undefined variable: 'm'")));
    }

    #[test]
    fn test_function_argument_valid_variable_ok() {
        // MATCH (n:Person) RETURN toUpper(n.name)
        let expr = ValueExpression::Function {
            name: "toUpper".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("n", "name"))],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_arithmetic_with_undefined_variable_in_return() {
        // RETURN x + 1
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Variable("x".to_string())),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(1))),
        };
        let result = analyze_return_expr(expr).unwrap();
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Undefined variable: 'x'")));
    }

    #[test]
    fn test_arithmetic_with_defined_property_ok() {
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Literal(PropertyValue::Integer(1))),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Property(PropertyRef::new("n", "age"))),
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        // Should not report undefined variable 'n'
        assert!(result
            .errors
            .iter()
            .all(|e| !e.contains("Undefined variable: 'n'")));
    }

    #[test]
    fn test_count_with_multiple_args_fails_validation() {
        // COUNT(n.age, n.name) should fail semantic validation
        let expr = ValueExpression::Function {
            name: "count".to_string(),
            args: vec![
                ValueExpression::Property(PropertyRef::new("n", "age")),
                ValueExpression::Property(PropertyRef::new("n", "name")),
            ],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("COUNT requires exactly 1 argument")),
            "Expected error about COUNT arity, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_count_with_zero_args_fails_validation() {
        // COUNT() with no arguments should fail
        let expr = ValueExpression::Function {
            name: "count".to_string(),
            args: vec![],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.contains("COUNT requires exactly 1 argument")),
            "Expected error about COUNT arity, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_count_with_one_arg_passes_validation() {
        // COUNT(n.age) should pass validation
        let expr = ValueExpression::Function {
            name: "count".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("n", "age"))],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result
                .errors
                .iter()
                .all(|e| !e.contains("COUNT requires exactly 1 argument")),
            "COUNT with 1 arg should not produce arity error, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_sum_with_variable_fails_validation() {
        let expr = ValueExpression::Function {
            name: "sum".to_string(),
            args: vec![ValueExpression::Variable("n".to_string())],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            !result.errors.is_empty(),
            "Expected SUM(variable) to produce validation errors"
        );
        let has_sum_error = result
            .errors
            .iter()
            .any(|e| e.contains("SUM(n) is invalid") && e.contains("requires a property"));
        assert!(
            has_sum_error,
            "Expected error about SUM requiring property, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_avg_with_variable_fails_validation() {
        let expr = ValueExpression::Function {
            name: "avg".to_string(),
            args: vec![ValueExpression::Variable("n".to_string())],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            !result.errors.is_empty(),
            "Expected AVG(variable) to produce validation errors"
        );
        let has_avg_error = result
            .errors
            .iter()
            .any(|e| e.contains("AVG(n) is invalid") && e.contains("requires a property"));
        assert!(
            has_avg_error,
            "Expected error about AVG requiring property, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_sum_with_property_passes_validation() {
        let expr = ValueExpression::Function {
            name: "sum".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("n", "age"))],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result.errors.is_empty(),
            "SUM with property should pass validation, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_min_with_variable_fails_validation() {
        let expr = ValueExpression::Function {
            name: "min".to_string(),
            args: vec![ValueExpression::Variable("n".to_string())],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            !result.errors.is_empty(),
            "Expected MIN(variable) to produce validation errors"
        );
        let has_min_error = result
            .errors
            .iter()
            .any(|e| e.contains("MIN(n) is invalid") && e.contains("requires a property"));
        assert!(
            has_min_error,
            "Expected error about MIN requiring property, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_max_with_variable_fails_validation() {
        let expr = ValueExpression::Function {
            name: "max".to_string(),
            args: vec![ValueExpression::Variable("n".to_string())],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            !result.errors.is_empty(),
            "Expected MAX(variable) to produce validation errors"
        );
        let has_max_error = result
            .errors
            .iter()
            .any(|e| e.contains("MAX(n) is invalid") && e.contains("requires a property"));
        assert!(
            has_max_error,
            "Expected error about MAX requiring property, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_min_with_property_passes_validation() {
        let expr = ValueExpression::Function {
            name: "min".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("n", "age"))],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result.errors.is_empty(),
            "MIN with property should pass validation, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_max_with_property_passes_validation() {
        let expr = ValueExpression::Function {
            name: "max".to_string(),
            args: vec![ValueExpression::Property(PropertyRef::new("n", "age"))],
        };
        let result = analyze_return_with_match("n", "Person", expr).unwrap();
        assert!(
            result.errors.is_empty(),
            "MAX with property should pass validation, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_arithmetic_with_non_numeric_literal_error() {
        // RETURN "x" + 1
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Literal(PropertyValue::String(
                "x".to_string(),
            ))),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Integer(1))),
        };
        let result = analyze_return_expr(expr).unwrap();
        // The semantic analyzer returns Ok with errors collected in the result
        assert!(result
            .errors
            .iter()
            .any(|e| e.contains("Arithmetic requires numeric literal operands")));
    }

    #[test]
    fn test_arithmetic_with_numeric_literals_ok() {
        // RETURN 1 + 2.0
        let expr = ValueExpression::Arithmetic {
            left: Box::new(ValueExpression::Literal(PropertyValue::Integer(1))),
            operator: ArithmeticOperator::Add,
            right: Box::new(ValueExpression::Literal(PropertyValue::Float(2.0))),
        };
        let result = analyze_return_expr(expr);
        assert!(result.is_ok(), "Expected Ok but got {:?}", result);
        assert!(result.unwrap().errors.is_empty());
    }
}
