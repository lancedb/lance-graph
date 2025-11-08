// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared test fixtures and utilities for datafusion_planner tests

use crate::config::GraphConfig;
use crate::logical_plan::LogicalOperator;
use crate::source_catalog::{InMemoryCatalog, SimpleTableSource};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub fn person_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int64, true),
    ]))
}

pub fn make_catalog() -> Arc<dyn crate::source_catalog::GraphSourceCatalog> {
    let person_src = Arc::new(SimpleTableSource::new(person_schema()));
    let knows_schema = Arc::new(Schema::new(vec![
        Field::new("src_person_id", DataType::Int64, false),
        Field::new("dst_person_id", DataType::Int64, false),
    ]));
    let knows_src = Arc::new(SimpleTableSource::new(knows_schema));
    Arc::new(
        InMemoryCatalog::new()
            .with_node_source("Person", person_src)
            .with_relationship_source("KNOWS", knows_src),
    )
}

pub fn person_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap()
}

pub fn person_knows_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
        .unwrap()
}

pub fn person_scan(variable: &str) -> LogicalOperator {
    LogicalOperator::ScanByLabel {
        variable: variable.to_string(),
        label: "Person".to_string(),
        properties: Default::default(),
    }
}
