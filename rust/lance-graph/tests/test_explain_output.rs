use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a Person dataset
fn create_person_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new("city", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Alice", "Bob", "Carol", "David", "Eve",
            ])),
            Arc::new(Int64Array::from(vec![28, 34, 29, 42, 31])),
            Arc::new(StringArray::from(vec!["NYC", "SF", "LA", "NYC", "SF"])),
        ],
    )
    .unwrap()
}

/// Helper to create a KNOWS relationship dataset
fn create_knows_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
        Field::new("since", DataType::Int64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![2, 3, 3, 4, 5])),
            Arc::new(Int64Array::from(vec![2020, 2019, 2021, 2018, 2022])),
        ],
    )
    .unwrap()
}

/// Helper to create a WORKS_AT relationship dataset
fn create_works_at_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("person_id", DataType::Int64, false),
        Field::new("company_id", DataType::Int64, false),
        Field::new("role", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30])),
            Arc::new(StringArray::from(vec![
                "Engineer", "Manager", "Engineer", "Director", "Engineer",
            ])),
        ],
    )
    .unwrap()
}

/// Helper to create a Company dataset
fn create_company_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("industry", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec!["TechCorp", "DataInc", "AILabs"])),
            Arc::new(StringArray::from(vec!["Technology", "Data", "AI"])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_explain_simple_node_scan() {
    println!("\n=== EXAMPLE 1: Simple Node Scan with Filter ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("Cypher Query:"));
    assert!(plan.contains("MATCH (p:Person)"));
    assert!(!plan.contains("| cypher_query")); // Query should not be in table
}

#[tokio::test]
async fn test_explain_one_hop_relationship() {
    println!("\n=== EXAMPLE 2: One-Hop Relationship Traversal ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WHERE a.age > 25 \
         RETURN a.name, b.name, b.city \
         ORDER BY a.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());
    datasets.insert("KNOWS".to_string(), create_knows_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("| graph_logical_plan"));
    assert!(plan.contains("Expand"));
    assert!(plan.contains("HashJoin") || plan.contains("Join"));
}

#[tokio::test]
async fn test_explain_two_hop_path() {
    println!("\n=== EXAMPLE 3: Two-Hop Path (Friends of Friends) ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         WHERE a.city = 'NYC' AND c.age < 35 \
         RETURN a.name AS source, b.name AS intermediate, c.name AS target \
         ORDER BY source, target \
         LIMIT 10",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());
    datasets.insert("KNOWS".to_string(), create_knows_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("| physical_plan"));
    // Should have multiple joins for two-hop path
    let join_count = plan.matches("Join").count();
    assert!(
        join_count >= 2,
        "Expected at least 2 joins for two-hop path"
    );
}

#[tokio::test]
async fn test_explain_multi_relationship_types() {
    println!("\n=== EXAMPLE 4: Multiple Relationship Types ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_node_label("Company", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .with_relationship("WORKS_AT", "person_id", "company_id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) \
         WHERE c.industry = 'Technology' \
         RETURN p.name, c.name, p.age \
         ORDER BY p.age DESC",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());
    datasets.insert("Company".to_string(), create_company_dataset());
    datasets.insert("WORKS_AT".to_string(), create_works_at_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("Company"));
    assert!(plan.contains("WORKS_AT"));
}

#[tokio::test]
async fn test_explain_distinct() {
    println!("\n=== EXAMPLE 5: DISTINCT Query ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (p:Person)-[:KNOWS]->(f:Person) \
         RETURN DISTINCT p.city",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());
    datasets.insert("KNOWS".to_string(), create_knows_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("Distinct") || plan.contains("DISTINCT"));
}

#[tokio::test]
async fn test_explain_complex_filter() {
    println!("\n=== EXAMPLE 6: Complex Boolean Filter ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (p:Person) \
         WHERE (p.age > 30 AND p.city = 'NYC') OR (p.age < 30 AND p.city = 'SF') \
         RETURN p.name, p.age, p.city",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("Filter"));
    assert!(plan.contains("OR") || plan.contains("And"));
}

#[tokio::test]
async fn test_explain_with_skip_and_limit() {
    println!("\n=== EXAMPLE 7: Pagination with SKIP and LIMIT ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (p:Person) \
         RETURN p.name, p.age \
         ORDER BY p.age DESC \
         SKIP 2 \
         LIMIT 3",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("Limit") || plan.contains("LIMIT"));
}

#[tokio::test]
async fn test_explain_relationship_properties() {
    println!("\n=== EXAMPLE 8: Filter on Relationship Properties ===\n");

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    let query = CypherQuery::new(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) \
         WHERE r.since > 2020 \
         RETURN a.name, b.name, r.since \
         ORDER BY r.since",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), create_person_dataset());
    datasets.insert("KNOWS".to_string(), create_knows_dataset());

    let plan = query.explain_datafusion(datasets).await.unwrap();
    println!("{}", plan);

    assert!(plan.contains("since") || plan.contains("Filter"));
}
