// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! High-level Cypher query interface for Lance datasets

use crate::ast::CypherQuery as CypherAST;
use crate::config::GraphConfig;
use crate::error::{GraphError, Result};
use crate::logical_plan::LogicalPlanner;
use crate::parser::parse_cypher_query;
use std::collections::HashMap;

mod path_executor;
use self::path_executor::PathExecutor;
mod aliases;
mod clauses;
mod expr;
use crate::query::expr::{to_df_boolean_expr_with_vars, to_df_literal};

/// A Cypher query that can be executed against Lance datasets
#[derive(Debug, Clone)]
pub struct CypherQuery {
    /// The original Cypher query string
    query_text: String,
    /// Parsed AST representation
    ast: CypherAST,
    /// Graph configuration for mapping
    config: Option<GraphConfig>,
    /// Query parameters
    parameters: HashMap<String, serde_json::Value>,
}
impl CypherQuery {
    /// Create a new Cypher query from a query string
    pub fn new(query: &str) -> Result<Self> {
        let ast = parse_cypher_query(query)?;

        Ok(Self {
            query_text: query.to_string(),
            ast,
            config: None,
            parameters: HashMap::new(),
        })
    }

    /// Set the graph configuration for this query
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a parameter to the query
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<serde_json::Value>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Add multiple parameters to the query
    pub fn with_parameters(mut self, params: HashMap<String, serde_json::Value>) -> Self {
        self.parameters.extend(params);
        self
    }

    /// Get the original query text
    pub fn query_text(&self) -> &str {
        &self.query_text
    }

    /// Get the parsed AST
    pub fn ast(&self) -> &CypherAST {
        &self.ast
    }

    /// Get the graph configuration
    pub fn config(&self) -> Option<&GraphConfig> {
        self.config.as_ref()
    }

    /// Get query parameters
    pub fn parameters(&self) -> &HashMap<String, serde_json::Value> {
        &self.parameters
    }

    /// Execute using the DataFusion planner with in-memory datasets
    ///
    /// # Overview
    /// This convenience method creates both a catalog and session context from the provided
    /// in-memory RecordBatches. It's ideal for testing and small datasets that fit in memory.
    ///
    /// For production use with external data sources (CSV, Parquet, databases), use
    /// `execute_with_datafusion_context` instead, which automatically builds the catalog
    /// from the SessionContext.
    ///
    /// # Arguments
    /// * `datasets` - HashMap of table name to RecordBatch (nodes and relationships)
    ///
    /// # Returns
    /// A single RecordBatch containing the query results
    ///
    /// # Example
    /// ```ignore
    /// use std::collections::HashMap;
    /// use arrow::record_batch::RecordBatch;
    /// use lance_graph::query::CypherQuery;
    ///
    /// // Create in-memory datasets
    /// let mut datasets = HashMap::new();
    /// datasets.insert("Person".to_string(), person_batch);
    /// datasets.insert("KNOWS".to_string(), knows_batch);
    ///
    /// // Parse and execute query
    /// let query = CypherQuery::parse("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name, f.name")?
    ///     .with_config(config);
    /// let result = query.execute_datafusion(datasets).await?;
    /// ```
    pub async fn execute_datafusion(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use crate::source_catalog::InMemoryCatalog;
        use datafusion::datasource::{DefaultTableSource, MemTable};
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        if datasets.is_empty() {
            return Err(GraphError::ConfigError {
                message: "No input datasets provided".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Create session context and catalog, register tables in both
        let ctx = SessionContext::new();
        let mut catalog: InMemoryCatalog = InMemoryCatalog::new();

        for (name, batch) in &datasets {
            let mem_table = Arc::new(
                MemTable::try_new(batch.schema(), vec![vec![batch.clone()]]).map_err(|e| {
                    GraphError::PlanError {
                        message: format!("Failed to create MemTable for {}: {}", name, e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?,
            );

            let table_source = Arc::new(DefaultTableSource::new(mem_table.clone()));

            // Register as both node and relationship source (planner will use whichever is appropriate)
            catalog = catalog
                .with_node_source(name, table_source.clone())
                .with_relationship_source(name, table_source.clone());

            // Register in session context for execution (using the same MemTable instance)
            ctx.register_table(name, mem_table)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to register table {}: {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Delegate to common execution logic
        self.execute_with_catalog_and_context(Arc::new(catalog), ctx)
            .await
    }

    /// Execute query with a DataFusion SessionContext, automatically building the catalog
    ///
    /// This is a convenience method that builds the graph catalog by querying the
    /// SessionContext for table schemas. The GraphConfig determines which tables to
    /// look up (node labels and relationship types).
    ///
    /// This method is ideal for integrating with DataFusion's rich data source ecosystem
    /// (CSV, Parquet, Delta Lake, Iceberg, etc.) without manually building a catalog.
    ///
    /// # Arguments
    /// * `ctx` - DataFusion SessionContext with pre-registered tables
    ///
    /// # Returns
    /// Query results as an Arrow RecordBatch
    ///
    /// # Errors
    /// Returns error if:
    /// - GraphConfig is not set (use `.with_config()` first)
    /// - Required tables are not registered in the SessionContext
    /// - Query execution fails
    ///
    /// # Example
    /// ```ignore
    /// use datafusion::execution::context::SessionContext;
    /// use datafusion::prelude::CsvReadOptions;
    /// use lance_graph::{CypherQuery, GraphConfig};
    ///
    /// // Step 1: Create GraphConfig
    /// let config = GraphConfig::builder()
    ///     .with_node_label("Person", "person_id")
    ///     .with_relationship("KNOWS", "src_id", "dst_id")
    ///     .build()?;
    ///
    /// // Step 2: Register data sources in DataFusion
    /// let ctx = SessionContext::new();
    /// ctx.register_csv("Person", "data/persons.csv", CsvReadOptions::default()).await?;
    /// ctx.register_parquet("KNOWS", "s3://bucket/knows.parquet", Default::default()).await?;
    ///
    /// // Step 3: Execute query (catalog is built automatically)
    /// let query = CypherQuery::parse("MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name")?
    ///     .with_config(config);
    /// let result = query.execute_with_datafusion_context(ctx).await?;
    /// ```
    ///
    /// # Note
    /// The catalog is built by querying the SessionContext for schemas of tables
    /// mentioned in the GraphConfig. Table names must match between GraphConfig
    /// (node labels/relationship types) and SessionContext (registered table names).
    pub async fn execute_with_datafusion_context(
        &self,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use crate::source_catalog::InMemoryCatalog;
        use datafusion::datasource::DefaultTableSource;
        use std::sync::Arc;

        // Require a config
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| GraphError::ConfigError {
                message: "Graph configuration is required for query execution".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Build catalog by querying SessionContext for table providers
        let mut catalog = InMemoryCatalog::new();

        // Register node sources
        for label in config.node_mappings.keys() {
            let table_provider =
                ctx.table_provider(label)
                    .await
                    .map_err(|e| GraphError::ConfigError {
                        message: format!(
                            "Node label '{}' not found in SessionContext: {}",
                            label, e
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            let table_source = Arc::new(DefaultTableSource::new(table_provider));
            catalog = catalog.with_node_source(label, table_source);
        }

        // Register relationship sources
        for rel_type in config.relationship_mappings.keys() {
            let table_provider =
                ctx.table_provider(rel_type)
                    .await
                    .map_err(|e| GraphError::ConfigError {
                        message: format!(
                            "Relationship type '{}' not found in SessionContext: {}",
                            rel_type, e
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            let table_source = Arc::new(DefaultTableSource::new(table_provider));
            catalog = catalog.with_relationship_source(rel_type, table_source);
        }

        // Execute using the built catalog
        self.execute_with_catalog_and_context(Arc::new(catalog), ctx)
            .await
    }

    /// Execute query with an explicit catalog and session context
    ///
    /// This is the most flexible API for advanced users who want to provide their own
    /// catalog implementation or have fine-grained control over both the catalog and
    /// session context.
    ///
    /// # Arguments
    /// * `catalog` - Graph catalog containing node and relationship schemas for planning
    /// * `ctx` - DataFusion SessionContext with registered data sources for execution
    ///
    /// # Returns
    /// Query results as an Arrow RecordBatch
    ///
    /// # Errors
    /// Returns error if query parsing, planning, or execution fails
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use datafusion::execution::context::SessionContext;
    /// use lance_graph::source_catalog::InMemoryCatalog;
    /// use lance_graph::query::CypherQuery;
    ///
    /// // Create custom catalog
    /// let catalog = InMemoryCatalog::new()
    ///     .with_node_source("Person", custom_table_source);
    ///
    /// // Create SessionContext
    /// let ctx = SessionContext::new();
    /// ctx.register_table("Person", custom_table).unwrap();
    ///
    /// // Execute with explicit catalog and context
    /// let query = CypherQuery::parse("MATCH (p:Person) RETURN p.name")?
    ///     .with_config(config);
    /// let result = query.execute_with_catalog_and_context(Arc::new(catalog), ctx).await?;
    /// ```
    pub async fn execute_with_catalog_and_context(
        &self,
        catalog: std::sync::Arc<dyn crate::source_catalog::GraphSourceCatalog>,
        ctx: datafusion::execution::context::SessionContext,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use crate::datafusion_planner::{DataFusionPlanner, GraphPhysicalPlanner};
        use crate::semantic::SemanticAnalyzer;
        use arrow::compute::concat_batches;

        // Require a config for DataFusion execution
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| GraphError::ConfigError {
                message: "Graph configuration is required for DataFusion execution".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Phase 1: Semantic Analysis
        let mut analyzer = SemanticAnalyzer::new(config.clone());
        analyzer.analyze(&self.ast)?;

        // Phase 2: Logical Planning
        let mut logical_planner = LogicalPlanner::new();
        let logical_plan = logical_planner.plan(&self.ast)?;

        // Phase 3: DataFusion Logical Planning
        // Convert graph logical plan to DataFusion logical plan
        let df_planner = DataFusionPlanner::with_catalog(config.clone(), catalog);
        let df_logical_plan = df_planner.plan(&logical_plan)?;

        // Phase 4: Physical Planning and Execution
        // DataFusion optimizes the logical plan, creates a physical execution plan,
        // and executes it against the pre-configured SessionContext
        let df = ctx
            .execute_logical_plan(df_logical_plan)
            .await
            .map_err(|e| GraphError::ExecutionError {
                message: format!("Failed to execute DataFusion plan: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Get schema before collecting (in case result is empty)
        let result_schema = df.schema().inner().clone();

        // Collect results
        let batches = df.collect().await.map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to collect query results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if batches.is_empty() {
            // Return empty batch with the schema from the DataFrame
            // This preserves column structure even when there are no rows
            return Ok(arrow::record_batch::RecordBatch::new_empty(result_schema));
        }

        // Combine all batches
        let schema = batches[0].schema();
        concat_batches(&schema, &batches).map_err(|e| GraphError::ExecutionError {
            message: format!("Failed to concatenate result batches: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })
    }

    /// Execute this Cypher query against Lance datasets
    ///
    /// Note: This initial implementation supports a single-table projection/filter/limit
    /// workflow to enable basic end-to-end execution. Multi-table/path execution will be
    /// wired up via the DataFusion planner in a follow-up.
    pub async fn execute(
        &self,
        datasets: HashMap<String, arrow::record_batch::RecordBatch>,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::compute::concat_batches;
        use datafusion::datasource::MemTable;
        use datafusion::prelude::*;
        use std::sync::Arc;

        // Require a config for now, even if we don't fully exploit it yet
        let _config = self
            .config
            .as_ref()
            .ok_or_else(|| GraphError::ConfigError {
                message: "Graph configuration is required for query execution".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        if datasets.is_empty() {
            return Err(GraphError::PlanError {
                message: "No input datasets provided".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        // Create DataFusion context and register all provided tables
        let ctx = SessionContext::new();
        for (name, batch) in &datasets {
            let table =
                MemTable::try_new(batch.schema(), vec![vec![batch.clone()]]).map_err(|e| {
                    GraphError::PlanError {
                        message: format!("Failed to create DataFusion table: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    }
                })?;
            ctx.register_table(name, Arc::new(table))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to register table '{}': {}", name, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        // Try to execute a path (1+ hops) if the query is a simple pattern
        if let Some(df) = self.try_execute_path_generic(&ctx).await? {
            let batches = df.collect().await.map_err(|e| GraphError::PlanError {
                message: format!("Failed to collect results: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
            if batches.is_empty() {
                let schema = datasets.values().next().unwrap().schema();
                return Ok(arrow_array::RecordBatch::new_empty(schema));
            }
            let merged = concat_batches(&batches[0].schema(), &batches).map_err(|e| {
                GraphError::PlanError {
                    message: format!("Failed to concatenate result batches: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                }
            })?;
            return Ok(merged);
        }

        // Fallback: single-table style query on the first provided table
        let (table_name, batch) = datasets.iter().next().unwrap();
        let schema = batch.schema();

        // Start a DataFrame from the registered table
        let mut df = ctx
            .table(table_name)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to create DataFrame for '{}': {}", table_name, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Apply WHERE if present (limited support: simple comparisons on a single column)
        if let Some(where_clause) = &self.ast.where_clause {
            if let Some(filter_expr) = to_df_boolean_expr_simple(&where_clause.expression) {
                df = df.filter(filter_expr).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
        }

        // Build projection from RETURN clause
        let proj_exprs: Vec<Expr> = self
            .ast
            .return_clause
            .items
            .iter()
            .map(|item| to_df_value_expr_simple(&item.expression))
            .collect();
        if !proj_exprs.is_empty() {
            df = df.select(proj_exprs).map_err(|e| GraphError::PlanError {
                message: format!("Failed to project: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Apply DISTINCT
        if self.ast.return_clause.distinct {
            df = df.distinct().map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply DISTINCT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Apply ORDER BY if present
        if let Some(order_by) = &self.ast.order_by {
            let sort_expr = to_df_order_by_expr_simple(&order_by.items);
            df = df.sort(sort_expr).map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply ORDER BY: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Apply SKIP/OFFSET and LIMIT if present
        if self.ast.skip.is_some() || self.ast.limit.is_some() {
            let offset = self.ast.skip.unwrap_or(0) as usize;
            let fetch = self.ast.limit.map(|l| l as usize);
            df = df.limit(offset, fetch).map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply SKIP/LIMIT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // Collect results and concat into a single RecordBatch
        let batches = df.collect().await.map_err(|e| GraphError::PlanError {
            message: format!("Failed to collect results: {}", e),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        if batches.is_empty() {
            // Return an empty batch with the source schema
            return Ok(arrow_array::RecordBatch::new_empty(schema));
        }

        let merged =
            concat_batches(&batches[0].schema(), &batches).map_err(|e| GraphError::PlanError {
                message: format!("Failed to concatenate result batches: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        Ok(merged)
    }

    /// Validate the query against the provided configuration
    pub fn validate(&self) -> Result<()> {
        // Check that all referenced labels exist in configuration
        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.validate_pattern(pattern)?;
            }
        }

        // Validate WHERE clause if present
        if let Some(where_clause) = &self.ast.where_clause {
            self.validate_boolean_expression(&where_clause.expression)?;
        }

        // Validate RETURN clause
        for item in &self.ast.return_clause.items {
            self.validate_value_expression(&item.expression)?;
        }

        Ok(())
    }

    /// Get all node labels referenced in this query
    pub fn referenced_node_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_node_labels_from_pattern(pattern, &mut labels);
            }
        }

        labels.sort();
        labels.dedup();
        labels
    }

    /// Get all relationship types referenced in this query
    pub fn referenced_relationship_types(&self) -> Vec<String> {
        let mut types = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_relationship_types_from_pattern(pattern, &mut types);
            }
        }

        types.sort();
        types.dedup();
        types
    }

    /// Get all variables used in this query
    pub fn variables(&self) -> Vec<String> {
        let mut variables = Vec::new();

        for match_clause in &self.ast.match_clauses {
            for pattern in &match_clause.patterns {
                self.collect_variables_from_pattern(pattern, &mut variables);
            }
        }

        variables.sort();
        variables.dedup();
        variables
    }

    // Validation helper methods

    fn validate_pattern(&self, pattern: &crate::ast::GraphPattern) -> Result<()> {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                for label in &node.labels {
                    if let Some(config) = &self.config {
                        if config.get_node_mapping(label).is_none() {
                            return Err(GraphError::PlanError {
                                message: format!("No mapping found for node label '{}'", label),
                                location: snafu::Location::new(file!(), line!(), column!()),
                            });
                        }
                    }
                }
                Ok(())
            }
            crate::ast::GraphPattern::Path(path) => {
                self.validate_pattern(&crate::ast::GraphPattern::Node(path.start_node.clone()))?;
                for segment in &path.segments {
                    for rel_type in &segment.relationship.types {
                        if let Some(config) = &self.config {
                            if config.get_relationship_mapping(rel_type).is_none() {
                                return Err(GraphError::PlanError {
                                    message: format!(
                                        "No mapping found for relationship type '{}'",
                                        rel_type
                                    ),
                                    location: snafu::Location::new(file!(), line!(), column!()),
                                });
                            }
                        }
                    }
                    self.validate_pattern(&crate::ast::GraphPattern::Node(
                        segment.end_node.clone(),
                    ))?;
                }
                Ok(())
            }
        }
    }

    fn validate_boolean_expression(&self, _expr: &crate::ast::BooleanExpression) -> Result<()> {
        // TODO: Implement validation of boolean expressions
        Ok(())
    }

    fn validate_value_expression(&self, _expr: &crate::ast::ValueExpression) -> Result<()> {
        // TODO: Implement validation of value expressions
        Ok(())
    }

    // Collection helper methods

    fn collect_node_labels_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        labels: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                labels.extend(node.labels.clone());
            }
            crate::ast::GraphPattern::Path(path) => {
                labels.extend(path.start_node.labels.clone());
                for segment in &path.segments {
                    labels.extend(segment.end_node.labels.clone());
                }
            }
        }
    }

    fn collect_relationship_types_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        types: &mut Vec<String>,
    ) {
        if let crate::ast::GraphPattern::Path(path) = pattern {
            for segment in &path.segments {
                types.extend(segment.relationship.types.clone());
            }
        }
    }

    fn collect_variables_from_pattern(
        &self,
        pattern: &crate::ast::GraphPattern,
        variables: &mut Vec<String>,
    ) {
        match pattern {
            crate::ast::GraphPattern::Node(node) => {
                if let Some(var) = &node.variable {
                    variables.push(var.clone());
                }
            }
            crate::ast::GraphPattern::Path(path) => {
                if let Some(var) = &path.start_node.variable {
                    variables.push(var.clone());
                }
                for segment in &path.segments {
                    if let Some(var) = &segment.relationship.variable {
                        variables.push(var.clone());
                    }
                    if let Some(var) = &segment.end_node.variable {
                        variables.push(var.clone());
                    }
                }
            }
        }
    }
}

impl CypherQuery {
    // Generic path executor (N-hop) entrypoint.
    async fn try_execute_path_generic(
        &self,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Result<Option<datafusion::dataframe::DataFrame>> {
        use crate::ast::GraphPattern;
        let [mc] = self.ast.match_clauses.as_slice() else {
            return Ok(None);
        };
        let match_clause = mc;
        let path = match match_clause.patterns.as_slice() {
            [GraphPattern::Path(p)] if !p.segments.is_empty() => p,
            _ => return Ok(None),
        };
        let cfg = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Graph configuration is required for execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;

        // Handle single-segment variable-length paths by unrolling ranges (*1..N, capped)
        if path.segments.len() == 1 {
            if let Some(length_range) = &path.segments[0].relationship.length {
                let cap: u32 = crate::MAX_VARIABLE_LENGTH_HOPS;
                let min_len = length_range.min.unwrap_or(1).max(1);
                let max_len = length_range.max.unwrap_or(cap);

                if min_len > max_len {
                    return Err(GraphError::InvalidPattern {
                        message: format!(
                            "Invalid variable-length range: min {:?} greater than max {:?}",
                            length_range.min, length_range.max
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }

                if max_len > cap {
                    return Err(GraphError::UnsupportedFeature {
                        feature: format!(
                            "Variable-length paths with length > {} are not supported (got {:?}..{:?})",
                            cap, length_range.min, length_range.max
                        ),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    });
                }

                use datafusion::dataframe::DataFrame;
                let mut union_df: Option<DataFrame> = None;

                for hops in min_len..=max_len {
                    // Build a fixed-length synthetic path by repeating the single segment
                    let mut synthetic = crate::ast::PathPattern {
                        start_node: path.start_node.clone(),
                        segments: Vec::with_capacity(hops as usize),
                    };

                    for i in 0..hops {
                        let mut seg = path.segments[0].clone();
                        // Drop variables to avoid alias collisions on repeated hops
                        seg.relationship.variable = None;
                        if (i + 1) < hops {
                            seg.end_node.variable = None; // intermediate hop
                        }
                        // Clear length spec for this fixed hop
                        seg.relationship.length = None;
                        synthetic.segments.push(seg);
                    }

                    let exec = PathExecutor::new(ctx, cfg, &synthetic)?;
                    let mut df = exec.build_chain().await?;
                    df = exec.apply_where(df, &self.ast)?;
                    df = exec.apply_return(df, &self.ast)?;

                    union_df = Some(match union_df {
                        Some(acc) => acc.union(df).map_err(|e| GraphError::PlanError {
                            message: format!("Failed to UNION variable-length paths: {}", e),
                            location: snafu::Location::new(file!(), line!(), column!()),
                        })?,
                        None => df,
                    });
                }

                return Ok(union_df);
            }
        }

        let exec = PathExecutor::new(ctx, cfg, path)?;
        let df = exec.build_chain().await?;
        let df = exec.apply_where(df, &self.ast)?;
        let df = exec.apply_return(df, &self.ast)?;
        Ok(Some(df))
    }

    // Attempt execution for a single-path pattern using joins.
    // Supports single-hop and two-hop expansions
    #[allow(dead_code)]
    async fn try_execute_single_hop_path(
        &self,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Result<Option<datafusion::dataframe::DataFrame>> {
        use crate::ast::{GraphPattern, RelationshipDirection, ValueExpression};
        use datafusion::prelude::*;

        // Only handle a single MATCH with a single path and exactly one segment
        let [mc] = self.ast.match_clauses.as_slice() else {
            return Ok(None);
        };
        let match_clause = mc;
        let path = match match_clause.patterns.as_slice() {
            [GraphPattern::Path(p)] if (p.segments.len() == 1 || p.segments.len() == 2) => p,
            _ => return Ok(None),
        };
        let seg = &path.segments[0];
        let rel_type = match seg.relationship.types.first() {
            Some(t) => t.as_str(),
            None => return Ok(None),
        };
        let start_label = match path.start_node.labels.first() {
            Some(l) => l.as_str(),
            None => return Ok(None),
        };
        let end_label = match seg.end_node.labels.first() {
            Some(l) => l.as_str(),
            None => return Ok(None),
        };

        let start_alias = path.start_node.variable.as_deref().unwrap_or(start_label);
        let rel_alias = seg.relationship.variable.as_deref().unwrap_or(rel_type);
        let end_alias = seg.end_node.variable.as_deref().unwrap_or(end_label);

        // Validate mappings
        let cfg = self.config.as_ref().ok_or_else(|| GraphError::PlanError {
            message: "Graph configuration is required for execution".to_string(),
            location: snafu::Location::new(file!(), line!(), column!()),
        })?;
        let start_map = cfg
            .get_node_mapping(start_label)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No node mapping for '{}'", start_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let end_map = cfg
            .get_node_mapping(end_label)
            .ok_or_else(|| GraphError::PlanError {
                message: format!("No node mapping for '{}'", end_label),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let rel_map =
            cfg.get_relationship_mapping(rel_type)
                .ok_or_else(|| GraphError::PlanError {
                    message: format!("No relationship mapping for '{}'", rel_type),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

        // Read tables and alias
        let mut left = ctx
            .table(start_label)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", start_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        // Alias and flatten columns to '<alias>__<col>' to avoid ambiguity
        let left_schema = left.schema();
        let left_proj: Vec<datafusion::logical_expr::Expr> = left_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    start_alias,
                    f.name()
                ))
            })
            .collect();
        left = left
            .alias(start_alias)?
            .select(left_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", start_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        for (k, v) in &path.start_node.properties {
            let expr = to_df_literal(v);
            left = left
                .filter(datafusion::logical_expr::Expr::BinaryExpr(
                    datafusion::logical_expr::BinaryExpr {
                        left: Box::new(datafusion::logical_expr::col(format!(
                            "{}__{}",
                            start_alias, k
                        ))),
                        op: datafusion::logical_expr::Operator::Eq,
                        right: Box::new(expr),
                    },
                ))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply filter: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        let mut rel_df = ctx
            .table(rel_type)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", rel_type, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let rel_schema = rel_df.schema();
        let rel_proj: Vec<datafusion::logical_expr::Expr> = rel_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    rel_alias,
                    f.name()
                ))
            })
            .collect();
        rel_df = rel_df
            .alias(rel_alias)?
            .select(rel_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", rel_type, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Join start -> relationship
        let (left_key, right_key) = match seg.relationship.direction {
            RelationshipDirection::Outgoing | RelationshipDirection::Undirected => (
                format!("{}__{}", start_alias, start_map.id_field),
                format!("{}__{}", rel_alias, rel_map.source_id_field),
            ),
            RelationshipDirection::Incoming => (
                format!("{}__{}", start_alias, start_map.id_field),
                format!("{}__{}", rel_alias, rel_map.target_id_field),
            ),
        };
        let mut joined = left
            .join(
                rel_df,
                JoinType::Inner,
                &[left_key.as_str()],
                &[right_key.as_str()],
                None,
            )
            .map_err(|e| GraphError::PlanError {
                message: format!("Join failed (node->rel): {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // Join relationship -> end (or mid, for 2-hop)
        let mut right = ctx
            .table(end_label)
            .await
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to read table '{}': {}", end_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let right_schema = right.schema();
        let right_proj: Vec<datafusion::logical_expr::Expr> = right_schema
            .fields()
            .iter()
            .map(|f| {
                datafusion::logical_expr::col(f.name()).alias(format!(
                    "{}__{}",
                    end_alias,
                    f.name()
                ))
            })
            .collect();
        right = right
            .alias(end_alias)?
            .select(right_proj)
            .map_err(|e| GraphError::PlanError {
                message: format!("Failed to alias/select '{}': {}", end_label, e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        let (left_key2, right_key2) = match seg.relationship.direction {
            RelationshipDirection::Outgoing | RelationshipDirection::Undirected => (
                format!("{}__{}", rel_alias, rel_map.target_id_field),
                format!("{}__{}", end_alias, end_map.id_field),
            ),
            RelationshipDirection::Incoming => (
                format!("{}__{}", rel_alias, rel_map.source_id_field),
                format!("{}__{}", end_alias, end_map.id_field),
            ),
        };
        joined = joined
            .join(
                right,
                JoinType::Inner,
                &[left_key2.as_str()],
                &[right_key2.as_str()],
                None,
            )
            .map_err(|e| GraphError::PlanError {
                message: format!("Join failed (rel->node): {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;

        // If there is a second segment (two-hop), continue chaining joins
        if path.segments.len() == 2 {
            let seg2 = &path.segments[1];
            let rel_type2 = match seg2.relationship.types.first() {
                Some(t) => t.as_str(),
                None => return Ok(None),
            };
            let end2_label = match seg2.end_node.labels.first() {
                Some(l) => l.as_str(),
                None => return Ok(None),
            };

            let mid_alias = end_alias; // end of seg1 is the mid node
            let mut rel2_alias = seg2
                .relationship
                .variable
                .as_deref()
                .unwrap_or(rel_type2)
                .to_string();
            let mut end2_alias = seg2
                .end_node
                .variable
                .as_deref()
                .unwrap_or(end2_label)
                .to_string();
            // Ensure unique aliases to avoid duplicate-qualified column names
            use std::collections::HashSet;
            let mut used_aliases: HashSet<String> = [
                start_alias.to_string(),
                rel_alias.to_string(),
                end_alias.to_string(),
            ]
            .into_iter()
            .collect();
            let mut uniquify = |alias: &mut String| {
                if used_aliases.insert(alias.clone()) {
                    return;
                }
                let base = alias.clone();
                let mut i = 2usize;
                loop {
                    let cand = format!("{}_{}", base, i);
                    if used_aliases.insert(cand.clone()) {
                        *alias = cand;
                        break;
                    }
                    i += 1;
                }
            };
            uniquify(&mut rel2_alias);
            uniquify(&mut end2_alias);

            // Validate mappings
            let _mid_map = end_map; // end of seg1
            let rel2_map =
                cfg.get_relationship_mapping(rel_type2)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No relationship mapping for '{}'", rel_type2),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
            let end2_map =
                cfg.get_node_mapping(end2_label)
                    .ok_or_else(|| GraphError::PlanError {
                        message: format!("No node mapping for '{}'", end2_label),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;

            // Read rel2 and end2
            let mut rel2_df = ctx
                .table(rel_type2)
                .await
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to read table '{}': {}", rel_type2, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            rel2_df = rel2_df.alias(&rel2_alias)?;

            // Determine the mid-equivalent column from rel1 to avoid ambiguous mid.id on the left
            let mid_equiv_from_rel1 = match seg.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    rel_map.target_id_field.as_str()
                }
                RelationshipDirection::Incoming => rel_map.source_id_field.as_str(),
            };

            // Join mid -> rel2 using mid-equivalent column from rel1
            let (left_key3, right_key3) = match seg2.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    (mid_equiv_from_rel1, rel2_map.source_id_field.as_str())
                }
                RelationshipDirection::Incoming => {
                    (mid_equiv_from_rel1, rel2_map.target_id_field.as_str())
                }
            };
            joined = joined
                .join(rel2_df, JoinType::Inner, &[left_key3], &[right_key3], None)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (mid->rel2): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            // Join rel2 -> end2
            let mut end2_df = ctx
                .table(end2_label)
                .await
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to read table '{}': {}", end2_label, e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            end2_df = end2_df.alias(&end2_alias)?;
            // If left side already contains a column with the same name as the right join key,
            // rename the right key to avoid ambiguous unqualified field references.
            let mut right_join_key = end2_map.id_field.clone();
            {
                let left_schema = joined.schema();
                if left_schema
                    .fields()
                    .iter()
                    .any(|f| f.name() == &right_join_key)
                {
                    use datafusion::logical_expr::{col, Expr};
                    let new_key = format!("{}__rhs", right_join_key);
                    let schema = end2_df.schema();
                    let mut proj: Vec<Expr> = Vec::with_capacity(schema.fields().len());
                    for f in schema.fields() {
                        if f.name() == &right_join_key {
                            proj.push(col(f.name()).alias(&new_key));
                        } else {
                            proj.push(col(f.name()));
                        }
                    }
                    end2_df = end2_df.select(proj).map_err(|e| GraphError::PlanError {
                        message: format!("Failed to prepare right join side: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
                    right_join_key = new_key;
                }
            }

            let (left_key4, right_key4) = match seg2.relationship.direction {
                RelationshipDirection::Outgoing | RelationshipDirection::Undirected => {
                    (rel2_map.target_id_field.as_str(), right_join_key.as_str())
                }
                RelationshipDirection::Incoming => {
                    (rel2_map.source_id_field.as_str(), right_join_key.as_str())
                }
            };
            joined = joined
                .join(end2_df, JoinType::Inner, &[left_key4], &[right_key4], None)
                .map_err(|e| GraphError::PlanError {
                    message: format!("Join failed (rel2->end2): {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;

            // Update end_alias to refer to final node for projection/WHERE qualification below
            let end_alias = end2_alias.as_str();

            // WHERE (qualified across all known aliases)
            if let Some(where_clause) = &self.ast.where_clause {
                if let Some(expr) =
                    to_df_boolean_expr_with_vars(&where_clause.expression, &|var, prop| {
                        let alias = if Some(var) == path.start_node.variable.as_deref() {
                            start_alias
                        } else if Some(var) == seg.relationship.variable.as_deref() {
                            rel_alias
                        } else if Some(var) == seg.end_node.variable.as_deref() {
                            mid_alias
                        } else if Some(var) == seg2.relationship.variable.as_deref() {
                            &rel2_alias
                        } else if Some(var) == seg2.end_node.variable.as_deref() {
                            end_alias
                        } else {
                            var
                        };
                        format!("{}.{}", alias, prop)
                    })
                {
                    joined = joined.filter(expr).map_err(|e| GraphError::PlanError {
                        message: format!("Failed to apply WHERE: {}", e),
                        location: snafu::Location::new(file!(), line!(), column!()),
                    })?;
                }
            }

            // Project RETURN across aliases
            let mut proj: Vec<datafusion::logical_expr::Expr> = Vec::new();
            for item in &self.ast.return_clause.items {
                if let ValueExpression::Property(prop) = &item.expression {
                    let col_name = if Some(prop.variable.as_str())
                        == path.start_node.variable.as_deref()
                    {
                        format!("{}.{}", start_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.relationship.variable.as_deref() {
                        format!("{}.{}", rel_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.end_node.variable.as_deref() {
                        format!("{}.{}", mid_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg2.relationship.variable.as_deref()
                    {
                        format!("{}.{}", rel2_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg2.end_node.variable.as_deref() {
                        format!("{}.{}", end_alias, prop.property)
                    } else {
                        format!("{}.{}", prop.variable, prop.property)
                    };
                    let mut e = datafusion::logical_expr::col(&col_name);
                    if let Some(a) = &item.alias {
                        e = e.alias(a);
                    }
                    proj.push(e);
                }
            }
            if !proj.is_empty() {
                joined = joined.select(proj).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to project: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }

            // DISTINCT and LIMIT
            if self.ast.return_clause.distinct {
                joined = joined.distinct().map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply DISTINCT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
            if let Some(limit) = self.ast.limit {
                joined =
                    joined
                        .limit(0, Some(limit as usize))
                        .map_err(|e| GraphError::PlanError {
                            message: format!("Failed to apply LIMIT: {}", e),
                            location: snafu::Location::new(file!(), line!(), column!()),
                        })?;
            }

            return Ok(Some(joined));
        }

        // WHERE (qualified)
        if let Some(where_clause) = &self.ast.where_clause {
            if let Some(expr) =
                to_df_boolean_expr_with_vars(&where_clause.expression, &|var, prop| {
                    let alias = if Some(var) == path.start_node.variable.as_deref() {
                        start_alias
                    } else if Some(var) == seg.relationship.variable.as_deref() {
                        rel_alias
                    } else if Some(var) == seg.end_node.variable.as_deref() {
                        end_alias
                    } else {
                        var
                    };
                    format!("{}.{}", alias, prop)
                })
            {
                joined = joined.filter(expr).map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply WHERE: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
            }
        }

        // Project RETURN
        let mut proj: Vec<datafusion::logical_expr::Expr> = Vec::new();
        for item in &self.ast.return_clause.items {
            if let ValueExpression::Property(prop) = &item.expression {
                let col_name =
                    if Some(prop.variable.as_str()) == path.start_node.variable.as_deref() {
                        format!("{}.{}", start_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.relationship.variable.as_deref() {
                        format!("{}.{}", rel_alias, prop.property)
                    } else if Some(prop.variable.as_str()) == seg.end_node.variable.as_deref() {
                        format!("{}.{}", end_alias, prop.property)
                    } else {
                        format!("{}.{}", prop.variable, prop.property)
                    };
                let mut e = datafusion::logical_expr::col(&col_name);
                if let Some(a) = &item.alias {
                    e = e.alias(a);
                }
                proj.push(e);
            }
        }
        if !proj.is_empty() {
            joined = joined.select(proj).map_err(|e| GraphError::PlanError {
                message: format!("Failed to project: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }

        // DISTINCT and LIMIT
        if self.ast.return_clause.distinct {
            joined = joined.distinct().map_err(|e| GraphError::PlanError {
                message: format!("Failed to apply DISTINCT: {}", e),
                location: snafu::Location::new(file!(), line!(), column!()),
            })?;
        }
        if let Some(limit) = self.ast.limit {
            joined = joined
                .limit(0, Some(limit as usize))
                .map_err(|e| GraphError::PlanError {
                    message: format!("Failed to apply LIMIT: {}", e),
                    location: snafu::Location::new(file!(), line!(), column!()),
                })?;
        }

        Ok(Some(joined))
    }
}

/// Builder for constructing Cypher queries programmatically
#[derive(Debug, Default)]
pub struct CypherQueryBuilder {
    match_clauses: Vec<crate::ast::MatchClause>,
    where_expression: Option<crate::ast::BooleanExpression>,
    return_items: Vec<crate::ast::ReturnItem>,
    order_by_items: Vec<crate::ast::OrderByItem>,
    limit: Option<u64>,
    distinct: bool,
    skip: Option<u64>,
    config: Option<GraphConfig>,
    parameters: HashMap<String, serde_json::Value>,
}

impl CypherQueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a MATCH clause for a node pattern
    pub fn match_node(mut self, variable: &str, label: &str) -> Self {
        let node = crate::ast::NodePattern {
            variable: Some(variable.to_string()),
            labels: vec![label.to_string()],
            properties: HashMap::new(),
        };

        let match_clause = crate::ast::MatchClause {
            patterns: vec![crate::ast::GraphPattern::Node(node)],
        };

        self.match_clauses.push(match_clause);
        self
    }

    /// Set the graph configuration
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a RETURN item
    pub fn return_property(mut self, variable: &str, property: &str) -> Self {
        let prop_ref = crate::ast::PropertyRef::new(variable, property);
        let return_item = crate::ast::ReturnItem {
            expression: crate::ast::ValueExpression::Property(prop_ref),
            alias: None,
        };

        self.return_items.push(return_item);
        self
    }

    /// Set DISTINCT flag
    pub fn distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    /// Add a LIMIT clause
    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add a SKIP clause
    pub fn skip(mut self, skip: u64) -> Self {
        self.skip = Some(skip);
        self
    }

    /// Build the final CypherQuery
    pub fn build(self) -> Result<CypherQuery> {
        if self.match_clauses.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one MATCH clause".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        if self.return_items.is_empty() {
            return Err(GraphError::PlanError {
                message: "Query must have at least one RETURN item".to_string(),
                location: snafu::Location::new(file!(), line!(), column!()),
            });
        }

        let ast = crate::ast::CypherQuery {
            match_clauses: self.match_clauses,
            where_clause: self
                .where_expression
                .map(|expr| crate::ast::WhereClause { expression: expr }),
            return_clause: crate::ast::ReturnClause {
                distinct: self.distinct,
                items: self.return_items,
            },
            order_by: if self.order_by_items.is_empty() {
                None
            } else {
                Some(crate::ast::OrderByClause {
                    items: self.order_by_items,
                })
            },
            limit: self.limit,
            skip: self.skip,
        };

        // Generate query text from AST (simplified)
        let query_text = "MATCH ... RETURN ...".to_string(); // TODO: Implement AST->text conversion

        let query = CypherQuery {
            query_text,
            ast,
            config: self.config,
            parameters: self.parameters,
        };

        query.validate()?;
        Ok(query)
    }
}

/// Minimal translator for simple boolean expressions into DataFusion Expr
fn to_df_boolean_expr_simple(
    expr: &crate::ast::BooleanExpression,
) -> Option<datafusion::logical_expr::Expr> {
    use crate::ast::{BooleanExpression as BE, ComparisonOperator as CO, ValueExpression as VE};
    use datafusion::logical_expr::{col, Expr, Operator};
    match expr {
        BE::Comparison {
            left,
            operator,
            right,
        } => {
            // Only support property <op> literal
            let (col_name, lit_expr) = match (left, right) {
                (VE::Property(prop), VE::Literal(val)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                (VE::Literal(val), VE::Property(prop)) => {
                    (prop.property.clone(), to_df_literal(val))
                }
                _ => return None,
            };
            let op = match operator {
                CO::Equal => Operator::Eq,
                CO::NotEqual => Operator::NotEq,
                CO::LessThan => Operator::Lt,
                CO::LessThanOrEqual => Operator::LtEq,
                CO::GreaterThan => Operator::Gt,
                CO::GreaterThanOrEqual => Operator::GtEq,
            };
            Some(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(col(col_name)),
                op,
                right: Box::new(lit_expr),
            }))
        }
        BE::And(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::And,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Or(l, r) => Some(datafusion::logical_expr::Expr::BinaryExpr(
            datafusion::logical_expr::BinaryExpr {
                left: Box::new(to_df_boolean_expr_simple(l)?),
                op: Operator::Or,
                right: Box::new(to_df_boolean_expr_simple(r)?),
            },
        )),
        BE::Not(inner) => Some(datafusion::logical_expr::Expr::Not(Box::new(
            to_df_boolean_expr_simple(inner)?,
        ))),
        BE::Exists(prop) => Some(datafusion::logical_expr::Expr::IsNotNull(Box::new(
            datafusion::logical_expr::Expr::Column(datafusion::common::Column::from_name(
                prop.property.clone(),
            )),
        ))),
        _ => None,
    }
}

/// Build ORDER BY expressions for simple queries (single table)
fn to_df_order_by_expr_simple(
    items: &[crate::ast::OrderByItem],
) -> Vec<datafusion::logical_expr::SortExpr> {
    use datafusion::logical_expr::SortExpr;

    items
        .iter()
        .map(|item| {
            let expr = to_df_value_expr_simple(&item.expression);
            let asc = matches!(item.direction, crate::ast::SortDirection::Ascending);
            SortExpr {
                expr,
                asc,
                nulls_first: false,
            }
        })
        .collect()
}

fn to_df_value_expr_simple(expr: &crate::ast::ValueExpression) -> datafusion::logical_expr::Expr {
    use crate::ast::ValueExpression as VE;
    use datafusion::logical_expr::{col, lit};
    match expr {
        VE::Property(prop) => col(&prop.property),
        VE::Variable(v) => col(v),
        VE::Literal(v) => crate::query::expr::to_df_literal(v),
        VE::Function { .. } | VE::Arithmetic { .. } => lit(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GraphConfig;

    #[test]
    fn test_parse_simple_cypher_query() {
        let query = CypherQuery::new("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(query.query_text(), "MATCH (n:Person) RETURN n.name");
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_query_with_parameters() {
        let mut params = HashMap::new();
        params.insert("minAge".to_string(), serde_json::Value::Number(30.into()));

        let query = CypherQuery::new("MATCH (n:Person) WHERE n.age > $minAge RETURN n.name")
            .unwrap()
            .with_parameters(params);

        assert!(query.parameters().contains_key("minAge"));
    }

    #[test]
    fn test_query_builder() {
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .build()
            .unwrap();

        let query = CypherQueryBuilder::new()
            .with_config(config)
            .match_node("n", "Person")
            .return_property("n", "name")
            .limit(10)
            .build()
            .unwrap();

        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.variables(), vec!["n"]);
    }

    #[test]
    fn test_relationship_query_parsing() {
        let query =
            CypherQuery::new("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name")
                .unwrap();
        assert_eq!(query.referenced_node_labels(), vec!["Person"]);
        assert_eq!(query.referenced_relationship_types(), vec!["KNOWS"]);
        assert_eq!(query.variables(), vec!["a", "b", "r"]);
    }

    #[tokio::test]
    async fn test_execute_basic_projection_and_filter() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Build a simple batch: name (Utf8), age (Int64)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let q = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("people".to_string(), batch);

        let out = q.execute(data).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // Expect Bob (34) and David (42)
        let result: Vec<(String, i64)> = (0..out.num_rows())
            .map(|i| (names.value(i).to_string(), ages.value(i)))
            .collect();
        assert!(result.contains(&("Bob".to_string(), 34)));
        assert!(result.contains(&("David".to_string(), 42)));
    }

    #[tokio::test]
    async fn test_execute_single_hop_path_join_projection() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // People table: id, name, age
        let person_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let people = RecordBatch::try_new(
            person_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![28, 34, 29])),
            ],
        )
        .unwrap();

        // KNOWS relationship: src_person_id -> dst_person_id
        let rel_schema = Arc::new(Schema::new(vec![
            Field::new("src_person_id", DataType::Int64, false),
            Field::new("dst_person_id", DataType::Int64, false),
        ]));
        let knows = RecordBatch::try_new(
            rel_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])), // Alice -> Bob, Bob -> Carol
                Arc::new(Int64Array::from(vec![2, 3])),
            ],
        )
        .unwrap();

        // Config: Person(id) and KNOWS(src_person_id -> dst_person_id)
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_person_id", "dst_person_id")
            .build()
            .unwrap();

        // Query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name
        let q = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        // Register tables using labels / rel types as names
        data.insert("Person".to_string(), people);
        data.insert("KNOWS".to_string(), knows);

        let out = q.execute(data).await.unwrap();
        // Expect two rows: Bob, Carol (the targets)
        let names = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let got: Vec<String> = (0..out.num_rows())
            .map(|i| names.value(i).to_string())
            .collect();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&"Bob".to_string()));
        assert!(got.contains(&"Carol".to_string()));
    }

    #[tokio::test]
    async fn test_execute_order_by_asc() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // name, age (int)
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Bob", "Alice", "David", "Carol"])),
                Arc::new(Int64Array::from(vec![34, 28, 42, 29])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Order ascending by age
        let q = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age ASC")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("people".to_string(), batch);

        let out = q.execute(data).await.unwrap();
        let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![28, 29, 34, 42]);
    }

    #[tokio::test]
    async fn test_execute_order_by_desc_with_skip_limit() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Bob", "Alice", "David", "Carol"])),
                Arc::new(Int64Array::from(vec![34, 28, 42, 29])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Desc by age, skip 1 (drop 42), take 2 -> [34, 29]
        let q =
            CypherQuery::new("MATCH (p:Person) RETURN p.age ORDER BY p.age DESC SKIP 1 LIMIT 2")
                .unwrap()
                .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("people".to_string(), batch);

        let out = q.execute(data).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let ages = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![34, 29]);
    }

    #[tokio::test]
    async fn test_execute_skip_without_limit() {
        use arrow_array::{Int64Array, RecordBatch};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("age", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![10, 20, 30, 40]))],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let q = CypherQuery::new("MATCH (p:Person) RETURN p.age ORDER BY p.age ASC SKIP 2")
            .unwrap()
            .with_config(cfg);

        let mut data = HashMap::new();
        data.insert("people".to_string(), batch);

        let out = q.execute(data).await.unwrap();
        assert_eq!(out.num_rows(), 2);
        let ages = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let collected: Vec<i64> = (0..out.num_rows()).map(|i| ages.value(i)).collect();
        assert_eq!(collected, vec![30, 40]);
    }

    #[tokio::test]
    async fn test_execute_datafusion_pipeline() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
                Arc::new(Int64Array::from(vec![25, 35, 30])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Test simple node query with DataFusion pipeline
        let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")
            .unwrap()
            .with_config(cfg);

        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), batch);

        // Execute using the new DataFusion pipeline
        let result = query.execute_datafusion(datasets.clone()).await;

        match &result {
            Ok(batch) => {
                println!(
                    "DataFusion result: {} rows, {} columns",
                    batch.num_rows(),
                    batch.num_columns()
                );
                if batch.num_rows() > 0 {
                    println!("First row data: {:?}", batch.slice(0, 1));
                }
            }
            Err(e) => {
                println!("DataFusion execution failed: {:?}", e);
            }
        }

        // For comparison, try legacy execution
        let legacy_result = query.execute(datasets).await.unwrap();
        println!(
            "Legacy result: {} rows, {} columns",
            legacy_result.num_rows(),
            legacy_result.num_columns()
        );

        let result = result.unwrap();

        // Verify correct filtering: should return 1 row (Bob with age > 30)
        assert_eq!(
            result.num_rows(),
            1,
            "Expected 1 row after filtering WHERE p.age > 30"
        );

        // Verify correct projection: should return 1 column (name)
        assert_eq!(
            result.num_columns(),
            1,
            "Expected 1 column after projection RETURN p.name"
        );

        // Verify correct data: should contain "Bob"
        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            names.value(0),
            "Bob",
            "Expected filtered result to contain Bob"
        );
    }

    #[tokio::test]
    async fn test_execute_datafusion_simple_scan() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            ],
        )
        .unwrap();

        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        // Test simple scan without filters
        let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .with_config(cfg);

        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), batch);

        // Execute using DataFusion pipeline
        let result = query.execute_datafusion(datasets).await.unwrap();

        // Should return all rows
        assert_eq!(
            result.num_rows(),
            2,
            "Should return all 2 rows without filtering"
        );
        assert_eq!(result.num_columns(), 1, "Should return 1 column (name)");

        // Verify data
        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let name_set: std::collections::HashSet<String> = (0..result.num_rows())
            .map(|i| names.value(i).to_string())
            .collect();
        let expected: std::collections::HashSet<String> =
            ["Alice", "Bob"].iter().map(|s| s.to_string()).collect();
        assert_eq!(name_set, expected, "Should return Alice and Bob");
    }

    #[tokio::test]
    async fn test_execute_with_context_simple_scan() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
                Arc::new(Int64Array::from(vec![28, 34, 29])),
            ],
        )
        .unwrap();

        // Create SessionContext and register data source
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Person", mem_table).unwrap();

        // Create query
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .with_config(cfg);

        // Execute with context (catalog built automatically)
        let result = query.execute_with_datafusion_context(ctx).await.unwrap();

        // Verify results
        assert_eq!(result.num_rows(), 3);
        assert_eq!(result.num_columns(), 1);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "Alice");
        assert_eq!(names.value(1), "Bob");
        assert_eq!(names.value(2), "Carol");
    }

    #[tokio::test]
    async fn test_execute_with_context_with_filter() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![28, 34, 29, 42])),
            ],
        )
        .unwrap();

        // Create SessionContext
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Person", mem_table).unwrap();

        // Create query with filter
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
            .unwrap()
            .with_config(cfg);

        // Execute with context
        let result = query.execute_with_datafusion_context(ctx).await.unwrap();

        // Verify: should return Bob (34) and David (42)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ages = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let results: Vec<(String, i64)> = (0..result.num_rows())
            .map(|i| (names.value(i).to_string(), ages.value(i)))
            .collect();

        assert!(results.contains(&("Bob".to_string(), 34)));
        assert!(results.contains(&("David".to_string(), 42)));
    }

    #[tokio::test]
    async fn test_execute_with_context_relationship_traversal() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create Person nodes
        let person_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let person_batch = RecordBatch::try_new(
            person_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol"])),
            ],
        )
        .unwrap();

        // Create KNOWS relationships
        let knows_schema = Arc::new(Schema::new(vec![
            Field::new("src_id", DataType::Int64, false),
            Field::new("dst_id", DataType::Int64, false),
            Field::new("since", DataType::Int64, false),
        ]));
        let knows_batch = RecordBatch::try_new(
            knows_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Int64Array::from(vec![2, 3])),
                Arc::new(Int64Array::from(vec![2020, 2021])),
            ],
        )
        .unwrap();

        // Create SessionContext and register tables
        let person_table = Arc::new(
            MemTable::try_new(person_schema.clone(), vec![vec![person_batch.clone()]]).unwrap(),
        );
        let knows_table = Arc::new(
            MemTable::try_new(knows_schema.clone(), vec![vec![knows_batch.clone()]]).unwrap(),
        );

        let ctx = SessionContext::new();
        ctx.register_table("Person", person_table).unwrap();
        ctx.register_table("KNOWS", knows_table).unwrap();

        // Create query
        let cfg = GraphConfig::builder()
            .with_node_label("Person", "id")
            .with_relationship("KNOWS", "src_id", "dst_id")
            .build()
            .unwrap();

        let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
            .unwrap()
            .with_config(cfg);

        // Execute with context
        let result = query.execute_with_datafusion_context(ctx).await.unwrap();

        // Verify: should return 2 relationships (Alice->Bob, Bob->Carol)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let src_names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let dst_names = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let relationships: Vec<(String, String)> = (0..result.num_rows())
            .map(|i| {
                (
                    src_names.value(i).to_string(),
                    dst_names.value(i).to_string(),
                )
            })
            .collect();

        assert!(relationships.contains(&("Alice".to_string(), "Bob".to_string())));
        assert!(relationships.contains(&("Bob".to_string(), "Carol".to_string())));
    }

    #[tokio::test]
    async fn test_execute_with_context_order_by_limit() {
        use arrow_array::{Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;

        // Create test data
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StringArray::from(vec!["Alice", "Bob", "Carol", "David"])),
                Arc::new(Int64Array::from(vec![85, 92, 78, 95])),
            ],
        )
        .unwrap();

        // Create SessionContext
        let mem_table =
            Arc::new(MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap());
        let ctx = SessionContext::new();
        ctx.register_table("Student", mem_table).unwrap();

        // Create query with ORDER BY and LIMIT
        let cfg = GraphConfig::builder()
            .with_node_label("Student", "id")
            .build()
            .unwrap();

        let query = CypherQuery::new(
            "MATCH (s:Student) RETURN s.name, s.score ORDER BY s.score DESC LIMIT 2",
        )
        .unwrap()
        .with_config(cfg);

        // Execute with context
        let result = query.execute_with_datafusion_context(ctx).await.unwrap();

        // Verify: should return top 2 scores (David: 95, Bob: 92)
        assert_eq!(result.num_rows(), 2);
        assert_eq!(result.num_columns(), 2);

        let names = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // First row should be David (95)
        assert_eq!(names.value(0), "David");
        assert_eq!(scores.value(0), 95);

        // Second row should be Bob (92)
        assert_eq!(names.value(1), "Bob");
        assert_eq!(scores.value(1), 92);
    }
}
