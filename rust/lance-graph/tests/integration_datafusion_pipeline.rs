use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper function to create a Person dataset
fn create_person_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new("city", DataType::Utf8, true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Alice", "Bob", "Charlie", "David", "Eve",
            ])),
            Arc::new(Int64Array::from(vec![25, 35, 30, 40, 28])),
            Arc::new(StringArray::from(vec![
                Some("New York"),
                Some("San Francisco"),
                Some("Chicago"),
                None,
                Some("Seattle"),
            ])),
        ],
    )
    .unwrap()
}

/// Helper function to create a KNOWS relationship dataset
fn create_knows_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_person_id", DataType::Int64, false),
        Field::new("dst_person_id", DataType::Int64, false),
        Field::new("since_year", DataType::Int64, true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 1])),
            Arc::new(Int64Array::from(vec![2, 3, 4, 5, 3])),
            Arc::new(Int64Array::from(vec![
                Some(2020),
                Some(2019),
                Some(2021),
                None,
                Some(2018),
            ])),
        ],
    )
    .unwrap()
}

/// Helper function to create graph config
fn create_graph_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_datafusion_simple_node_scan() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return all 5 people
    assert_eq!(result.num_rows(), 5);
    assert_eq!(result.num_columns(), 1);

    // Verify all names are present
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let name_set: std::collections::HashSet<String> = (0..result.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    let expected: std::collections::HashSet<String> = ["Alice", "Bob", "Charlie", "David", "Eve"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(name_set, expected);
}

#[tokio::test]
async fn test_datafusion_node_filtering() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return 3 people (Bob:35, David:40, Charlie:30 is not > 30)
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_columns(), 2);

    // Verify the filtered results
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

    let mut results = Vec::new();
    for i in 0..result.num_rows() {
        results.push((names.value(i).to_string(), ages.value(i)));
    }

    // Sort for consistent comparison
    results.sort();
    assert_eq!(
        results,
        vec![("Bob".to_string(), 35), ("David".to_string(), 40)]
    );
}

#[tokio::test]
async fn test_datafusion_multiple_conditions() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age >= 30 RETURN p.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return people with age >= 30
    // Bob:35, Charlie:30, David:40
    assert_eq!(result.num_rows(), 3);

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let name_set: std::collections::HashSet<String> = (0..result.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    let expected: std::collections::HashSet<String> = ["Bob", "Charlie", "David"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(name_set, expected);
}

#[tokio::test]
async fn test_datafusion_relationship_traversal() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Test basic relationship traversal with strict assertions
    let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return source names for all relationships
    assert_eq!(result.num_rows(), 5); // 5 relationships in the dataset
    assert_eq!(result.num_columns(), 1);

    // Verify exact source name counts
    let source_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for i in 0..result.num_rows() {
        *counts.entry(source_names.value(i).to_string()).or_insert(0) += 1;
    }

    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    // Source name counts: Alice:2, Bob:1, Charlie:1, David:1
    assert_eq!(counts.get("Alice"), Some(&2));
    assert_eq!(counts.get("Bob"), Some(&1));
    assert_eq!(counts.get("Charlie"), Some(&1));
    assert_eq!(counts.get("David"), Some(&1));
    assert!(
        !counts.contains_key("Eve"),
        "Eve has no outgoing KNOWS relationships"
    );
}

#[tokio::test]
async fn test_datafusion_relationship_with_variable() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Test relationship traversal with strict count verification
    let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_columns(), 1);
    assert_eq!(result.num_rows(), 5);

    // Verify exact counts
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for i in 0..result.num_rows() {
        *counts.entry(names.value(i).to_string()).or_insert(0) += 1;
    }

    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    assert_eq!(counts.get("Alice"), Some(&2));
    assert_eq!(counts.get("Bob"), Some(&1));
    assert_eq!(counts.get("Charlie"), Some(&1));
    assert_eq!(counts.get("David"), Some(&1));
    assert!(!counts.contains_key("Eve"));
}

#[tokio::test]
async fn test_datafusion_complex_filtering() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    let query =
        CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_columns(), 1);
    // Bob (35) has 1 edge: 2->3, David (40) has 1 edge: 4->5
    assert_eq!(result.num_rows(), 2);

    // Verify exact results
    let source_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let name_set: std::collections::HashSet<String> = (0..result.num_rows())
        .map(|i| source_names.value(i).to_string())
        .collect();
    let expected: std::collections::HashSet<String> = ["Bob", "David"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(name_set, expected);
}

#[tokio::test]
async fn test_datafusion_projection_multiple_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age >= 28 RETURN p.name, p.age")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return people with age >= 28 (Bob:35, Charlie:30, Eve:28, David:40)
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 2);

    // Verify column types and data
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

    for i in 0..result.num_rows() {
        let age = ages.value(i);
        assert!(age >= 28);

        let name = names.value(i);
        assert!(["Bob", "Charlie", "Eve", "David"].contains(&name));
    }
}

#[tokio::test]
async fn test_datafusion_error_handling_missing_config() {
    let person_batch = create_person_dataset();

    // Query without config should fail
    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name").unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await;
    assert!(result.is_err());

    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(error_msg.contains("Graph configuration is required"));
}

#[tokio::test]
async fn test_datafusion_error_handling_empty_datasets() {
    let config = create_graph_config();

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let datasets = HashMap::new(); // Empty datasets

    let result = query.execute_datafusion(datasets).await;
    assert!(result.is_err());

    let error_msg = format!("{:?}", result.unwrap_err());
    assert!(error_msg.contains("No input datasets provided"));
}

#[tokio::test]
async fn test_datafusion_performance_large_dataset() {
    let config = create_graph_config();

    // Create a larger dataset for performance testing
    let large_size = 1000;
    let ids: Vec<i64> = (1..=large_size).collect();
    let names: Vec<String> = (1..=large_size).map(|i| format!("Person{}", i)).collect();
    let ages: Vec<i64> = (1..=large_size).map(|i| 20 + (i % 50)).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
    ]));

    let large_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(ages)),
        ],
    )
    .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 40 RETURN p.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), large_batch);

    let start = std::time::Instant::now();
    let result = query.execute_datafusion(datasets).await.unwrap();
    let duration = start.elapsed();

    // Should complete reasonably quickly (adjust threshold as needed)
    assert!(
        duration.as_millis() < 1000,
        "Query took too long: {:?}",
        duration
    );

    // Verify correct filtering (ages 41-69 out of 20-69 range)
    let actual_count = result.num_rows();

    // Each age appears 20 times (1000 people, ages 20-69, so 50 different ages)
    // Ages 41-69 = 29 ages * 20 people each = 580 people
    assert_eq!(actual_count, 580);
}

#[tokio::test]
async fn test_datafusion_empty_result_set() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query that should return no results
    let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 100 RETURN p.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return empty result set
    assert_eq!(result.num_rows(), 0);
    // Note: Even with 0 rows, DataFusion still returns the expected column structure
    assert!(result.num_columns() >= 1);
}

#[tokio::test]
async fn test_datafusion_all_columns_projection() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query that returns all columns
    let query =
        CypherQuery::new("MATCH (p:Person) WHERE p.id = 1 RETURN p.id, p.name, p.age, p.city")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return Alice's data
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.num_columns(), 4);

    // Verify Alice's data
    let ids = result
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ages = result
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let cities = result
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(names.value(0), "Alice");
    assert_eq!(ages.value(0), 25);
    assert_eq!(cities.value(0), "New York");
}

#[tokio::test]
async fn test_datafusion_relationship_count() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Count relationships with strict verification
    let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return 5 relationships (as per create_knows_dataset)
    assert_eq!(result.num_rows(), 5);

    // Verify exact source name counts
    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut name_counts = std::collections::HashMap::new();

    for i in 0..result.num_rows() {
        let name = names.value(i);
        *name_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    // Source name counts: Alice:2, Bob:1, Charlie:1, David:1
    assert_eq!(name_counts.get("Alice"), Some(&2));
    assert_eq!(name_counts.get("Bob"), Some(&1));
    assert_eq!(name_counts.get("Charlie"), Some(&1));
    assert_eq!(name_counts.get("David"), Some(&1));
    assert!(!name_counts.contains_key("Eve"));

    // Verify total
    let total_relationships: usize = name_counts.values().sum();
    assert_eq!(total_relationships, 5);
}

#[tokio::test]
async fn test_datafusion_one_hop_source_names_strict() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 1);
    assert_eq!(out.num_rows(), 5);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for i in 0..out.num_rows() {
        *counts.entry(names.value(i).to_string()).or_insert(0) += 1;
    }
    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    // Source name counts: Alice:2, Bob:1, Charlie:1, David:1
    assert_eq!(counts.get("Alice"), Some(&2));
    assert_eq!(counts.get("Bob"), Some(&1));
    assert_eq!(counts.get("Charlie"), Some(&1));
    assert_eq!(counts.get("David"), Some(&1));
    assert!(!counts.contains_key("Eve"));
}

#[tokio::test]
async fn test_datafusion_one_hop_filtered_source_age_strict() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    let query =
        CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN a.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 1);
    // Bob (35): 2->3, David (40): 4->5
    assert_eq!(out.num_rows(), 2);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let set: std::collections::HashSet<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    let expected: std::collections::HashSet<String> = ["Bob", "David"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(set, expected);
}
