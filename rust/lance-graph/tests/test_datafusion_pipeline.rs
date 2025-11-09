use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Test Data Structure
// ============================================================================
//
// Person Dataset (5 nodes):
// | ID | Name    | Age | City          |
// |----|---------|-----|---------------|
// | 1  | Alice   | 25  | New York      |
// | 2  | Bob     | 35  | San Francisco |
// | 3  | Charlie | 30  | Chicago       |
// | 4  | David   | 40  | NULL          |
// | 5  | Eve     | 28  | Seattle       |
//
// KNOWS Relationship Dataset (5 edges):
// | src_person_id | dst_person_id | since_year |
// |---------------|---------------|------------|
// | 1             | 2             | 2020       |
// | 2             | 3             | 2019       |
// | 3             | 4             | 2021       |
// | 4             | 5             | NULL       |
// | 1             | 3             | 2018       |
//
// Visual Graph Structure:
//
//     Alice(1) ──2020──> Bob(2) ──2019──> Charlie(3) ──2021──> David(4) ──NULL──> Eve(5)
//        │                                    ▲
//        └──────────────2018──────────────────┘
//
// Single-hop paths (5 edges):
//   1. Alice → Bob
//   2. Bob → Charlie
//   3. Charlie → David
//   4. David → Eve
//   5. Alice → Charlie (shortcut)
//
// Two-hop paths (4 paths):
//   1. Alice → Bob → Charlie
//   2. Bob → Charlie → David
//   3. Charlie → David → Eve
//   4. Alice → Charlie → David
//
// Key characteristics:
//   - Eve (5): Has no outgoing edges (dead end)
//   - Alice (1): Has 2 outgoing edges (most connections)
//   - David (4): Has NULL since_year and NULL city values
// ============================================================================

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

// Helper function to execute a query and return results
async fn execute_test_query(cypher: &str) -> RecordBatch {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    let query = CypherQuery::new(cypher).unwrap().with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    query.execute_datafusion(datasets).await.unwrap()
}

// Helper function to extract string column values
fn get_string_column(batch: &RecordBatch, col_idx: usize) -> Vec<String> {
    let array = batch
        .column(col_idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (0..array.len())
        .map(|i| array.value(i).to_string())
        .collect()
}

// ============================================================================
// Basic Node Query Tests
// ============================================================================

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

// ============================================================================
// Basic Relationship Query Tests
// ============================================================================

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

    // WHERE a.age > 30 filters source, {age: 30} filters target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person {age: 30}) WHERE a.age > 30 RETURN a.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_columns(), 1);
    // Only Bob (35) -> Charlie (30), David doesn't connect to anyone age 30
    assert_eq!(result.num_rows(), 1);

    // Verify exact results
    let source_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    // Should only be Bob
    assert_eq!(source_names.value(0), "Bob");
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

#[tokio::test]
async fn test_datafusion_one_hop_with_city_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Filter targets by city using inline property filter
    // Tests inline property filter instead of WHERE clause
    let query =
        CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b:Person {city: 'Seattle'}) RETURN b.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Only Eve has city = 'Seattle' and is reachable (David->Eve)
    assert_eq!(out.num_rows(), 1);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "Eve");
}

#[tokio::test]
async fn test_datafusion_one_hop_multiple_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return multiple properties from both source and target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.name, a.age, b.name, b.age",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_columns(), 4);
    assert_eq!(out.num_rows(), 5);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let a_ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let b_names = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_ages = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify at least one row has correct data
    let mut found_alice_bob = false;
    for i in 0..out.num_rows() {
        if a_names.value(i) == "Alice" && b_names.value(i) == "Bob" {
            assert_eq!(a_ages.value(i), 25);
            assert_eq!(b_ages.value(i), 35);
            found_alice_bob = true;
        }
    }
    assert!(found_alice_bob);
}

#[tokio::test]
async fn test_datafusion_one_hop_return_relationship_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return both node and relationship properties in projection
    // This validates qualified relationship columns and aliasing
    let query = CypherQuery::new(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) \
         RETURN a.name, r.since_year, b.name \
         ORDER BY a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should return 3 columns: a.name, r.since_year, b.name
    assert_eq!(out.num_columns(), 3);
    // Should return 5 edges
    assert_eq!(out.num_rows(), 5);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let since_years = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let b_names = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify first row: Alice -> Bob (2020)
    assert_eq!(a_names.value(0), "Alice");
    assert_eq!(since_years.value(0), 2020);
    assert_eq!(b_names.value(0), "Bob");

    // Verify second row: Alice -> Charlie (2018)
    assert_eq!(a_names.value(1), "Alice");
    assert_eq!(since_years.value(1), 2018);
    assert_eq!(b_names.value(1), "Charlie");

    // Verify third row: Bob -> Charlie (2019)
    assert_eq!(a_names.value(2), "Bob");
    assert_eq!(since_years.value(2), 2019);
    assert_eq!(b_names.value(2), "Charlie");

    // Verify fourth row: Charlie -> David (2021)
    assert_eq!(a_names.value(3), "Charlie");
    assert_eq!(since_years.value(3), 2021);
    assert_eq!(b_names.value(3), "David");

    // Verify fifth row: David -> Eve (NULL since_year)
    assert_eq!(a_names.value(4), "David");
    assert!(since_years.is_null(4)); // NULL value
    assert_eq!(b_names.value(4), "Eve");
}

// ============================================================================
// Two-Hop Path Query Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_two_hop_basic() {
    // Query: Find friends of friends
    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    // Two-hop paths: 1->2->3, 2->3->4, 3->4->5, 1->3->4
    let out = execute_test_query(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c.name",
    )
    .await;

    // Should return: Charlie (from 1->2->3), David (from 2->3->4 and 1->3->4), Eve (from 3->4->5)
    assert_eq!(out.num_columns(), 1);
    assert_eq!(out.num_rows(), 4); // 4 two-hop paths

    let names = get_string_column(&out, 0);

    let mut counts = HashMap::<String, usize>::new();
    for name in names {
        *counts.entry(name).or_insert(0) += 1;
    }

    // Verify counts: Charlie:1, David:2, Eve:1
    assert_eq!(counts.get("Charlie"), Some(&1));
    assert_eq!(counts.get("David"), Some(&2));
    assert_eq!(counts.get("Eve"), Some(&1));
    assert!(!counts.contains_key("Alice"));
    assert!(!counts.contains_key("Bob"));
}

#[tokio::test]
async fn test_datafusion_two_hop_return_intermediate() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return the intermediate node in two-hop paths
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 1);
    assert_eq!(out.num_rows(), 4);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut counts = HashMap::<String, usize>::new();
    for i in 0..out.num_rows() {
        *counts.entry(names.value(i).to_string()).or_insert(0) += 1;
    }

    // Intermediate nodes: Bob (1->2->3), Charlie (2->3->4 and 1->3->4), David (3->4->5)
    assert_eq!(counts.get("Bob"), Some(&1));
    assert_eq!(counts.get("Charlie"), Some(&2));
    assert_eq!(counts.get("David"), Some(&1));
}

#[tokio::test]
async fn test_datafusion_two_hop_return_all_three() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return all three nodes in the path
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN a.name, b.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 3);
    assert_eq!(out.num_rows(), 4);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let c_names = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Collect all paths
    let mut paths = Vec::new();
    for i in 0..out.num_rows() {
        paths.push((
            a_names.value(i).to_string(),
            b_names.value(i).to_string(),
            c_names.value(i).to_string(),
        ));
    }

    // Expected paths: Alice->Bob->Charlie, Bob->Charlie->David, Charlie->David->Eve, Alice->Charlie->David
    assert!(paths.contains(&(
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string()
    )));
    assert!(paths.contains(&(
        "Bob".to_string(),
        "Charlie".to_string(),
        "David".to_string()
    )));
    assert!(paths.contains(&(
        "Charlie".to_string(),
        "David".to_string(),
        "Eve".to_string()
    )));
    assert!(paths.contains(&(
        "Alice".to_string(),
        "Charlie".to_string(),
        "David".to_string()
    )));
}

#[tokio::test]
async fn test_datafusion_two_hop_with_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop with filter on intermediate node
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE b.age > 30 RETURN c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Filter: b.age > 30 means b can be Bob(35), David(40)
    // Paths with Bob as intermediate: 1->2->3 (Alice->Bob->Charlie)
    // Paths with David as intermediate: 3->4->5 (Charlie->David->Eve)
    // No paths with Charlie(30) as intermediate
    assert_eq!(out.num_rows(), 2);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let result_names: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    assert!(result_names.contains(&"Charlie".to_string()));
    assert!(result_names.contains(&"Eve".to_string()));
}

#[tokio::test]
async fn test_datafusion_two_hop_with_relationship_variable() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop with relationship variables
    let query = CypherQuery::new(
        "MATCH (a:Person)-[r1:KNOWS]->(b:Person)-[r2:KNOWS]->(c:Person) RETURN a.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 2);
    assert_eq!(out.num_rows(), 4);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let c_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify we get the correct source->target pairs
    let mut pairs = Vec::new();
    for i in 0..out.num_rows() {
        pairs.push((a_names.value(i).to_string(), c_names.value(i).to_string()));
    }

    assert!(pairs.contains(&("Alice".to_string(), "Charlie".to_string())));
    assert!(pairs.contains(&("Bob".to_string(), "David".to_string())));
    assert!(pairs.contains(&("Charlie".to_string(), "Eve".to_string())));
    assert!(pairs.contains(&("Alice".to_string(), "David".to_string())));
}

#[tokio::test]
async fn test_datafusion_two_hop_distinct() {
    // Query: Get distinct final destinations in two-hop paths
    let out = execute_test_query(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN DISTINCT c.name",
    )
    .await;

    assert_eq!(out.num_columns(), 1);
    // Three distinct targets: Charlie, David, Eve
    assert_eq!(out.num_rows(), 3);

    let mut names = get_string_column(&out, 0);
    names.sort();

    assert_eq!(names, vec!["Charlie", "David", "Eve"]);
}

#[tokio::test]
async fn test_datafusion_two_hop_no_results() {
    // Query: Two-hop starting from Eve (who has no outgoing edges)
    let out = execute_test_query(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE a.name = 'Eve' RETURN c.name"
    )
    .await;

    // Eve has no outgoing edges, so no two-hop paths
    assert_eq!(out.num_rows(), 0);
}

// ============================================================================
// Complex Query Tests (Advanced Filtering & Multi-Condition)
// ============================================================================

#[tokio::test]
async fn test_datafusion_two_hop_with_multiple_filters() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop with filters on source, intermediate, and target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         WHERE a.age < 30 AND b.age >= 30 AND c.age > 25 \
         RETURN a.name, b.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // a.age < 30: Alice(25), Eve(28)
    // b.age >= 30: Bob(35), Charlie(30), David(40)
    // c.age > 25: Bob(35), Charlie(30), David(40), Eve(28)
    // Paths from Alice: Alice->Bob->Charlie, Alice->Charlie->David
    // Valid: Alice(25)->Bob(35)->Charlie(30), Alice(25)->Charlie(30)->David(40)
    assert_eq!(out.num_rows(), 2);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let c_names = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut paths = Vec::new();
    for i in 0..out.num_rows() {
        paths.push((
            a_names.value(i).to_string(),
            b_names.value(i).to_string(),
            c_names.value(i).to_string(),
        ));
    }

    assert!(paths.contains(&(
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string()
    )));
    assert!(paths.contains(&(
        "Alice".to_string(),
        "Charlie".to_string(),
        "David".to_string()
    )));
}

#[tokio::test]
async fn test_datafusion_two_hop_return_relationship_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Filter two-hop paths by relationship property on first hop
    // Only paths where first relationship has since_year = 2020
    // Alice-[2020]->Bob-[2019]->Charlie is the only match
    let query = CypherQuery::new(
        "MATCH (a:Person)-[r1:KNOWS {since_year: 2020}]->(b:Person)-[r2:KNOWS]->(c:Person) \
         RETURN a.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 2);
    // Only Alice->Bob->Charlie (Alice-[2020]->Bob-[2019]->Charlie)
    assert_eq!(out.num_rows(), 1);

    let sources = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let targets = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(sources.value(0), "Alice");
    assert_eq!(targets.value(0), "Charlie");
}

#[tokio::test]
async fn test_datafusion_two_hop_return_both_relationship_properties() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return properties from both relationships in a two-hop path
    // This validates qualified relationship columns for r1 and r2, and proper aliasing
    let query = CypherQuery::new(
        "MATCH (a:Person)-[r1:KNOWS]->(b:Person)-[r2:KNOWS]->(c:Person) \
         RETURN a.name, r1.since_year, b.name, r2.since_year, c.name \
         ORDER BY a.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should return 5 columns: a.name, r1.since_year, b.name, r2.since_year, c.name
    assert_eq!(out.num_columns(), 5);
    // Should return 4 two-hop paths
    assert_eq!(out.num_rows(), 4);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let r1_years = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let b_names = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let r2_years = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
    let c_names = out
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify first path: Alice -[2020]-> Bob -[2019]-> Charlie
    assert_eq!(a_names.value(0), "Alice");
    assert_eq!(r1_years.value(0), 2020);
    assert_eq!(b_names.value(0), "Bob");
    assert_eq!(r2_years.value(0), 2019);
    assert_eq!(c_names.value(0), "Charlie");

    // Verify second path: Alice -[2018]-> Charlie -[2021]-> David
    assert_eq!(a_names.value(1), "Alice");
    assert_eq!(r1_years.value(1), 2018);
    assert_eq!(b_names.value(1), "Charlie");
    assert_eq!(r2_years.value(1), 2021);
    assert_eq!(c_names.value(1), "David");

    // Verify third path: Bob -[2019]-> Charlie -[2021]-> David
    assert_eq!(a_names.value(2), "Bob");
    assert_eq!(r1_years.value(2), 2019);
    assert_eq!(b_names.value(2), "Charlie");
    assert_eq!(r2_years.value(2), 2021);
    assert_eq!(c_names.value(2), "David");

    // Verify fourth path: Charlie -[2021]-> David -[NULL]-> Eve
    assert_eq!(a_names.value(3), "Charlie");
    assert_eq!(r1_years.value(3), 2021);
    assert_eq!(b_names.value(3), "David");
    assert!(r2_years.is_null(3)); // NULL value for David -> Eve
    assert_eq!(c_names.value(3), "Eve");
}

#[tokio::test]
async fn test_datafusion_two_hop_with_limit() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop with LIMIT
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN c.name LIMIT 2",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should return only 2 rows (limited from 4 total paths)
    assert_eq!(out.num_rows(), 2);
}

#[tokio::test]
async fn test_datafusion_complex_boolean_expression() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Complex boolean expression with AND/OR
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WHERE (a.age > 30 AND b.age < 35) OR (a.name = 'Alice' AND b.name = 'Bob') \
         RETURN a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Matches:
    // - Bob(35)->Charlie(30): age > 30 AND age < 35
    // - David(40)->Eve(28): age > 30 AND age < 35
    // - Alice(25)->Bob(35): name = 'Alice' AND name = 'Bob'
    assert_eq!(out.num_rows(), 3);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut pairs = Vec::new();
    for i in 0..out.num_rows() {
        pairs.push((a_names.value(i).to_string(), b_names.value(i).to_string()));
    }

    assert!(pairs.contains(&("Alice".to_string(), "Bob".to_string())));
    assert!(pairs.contains(&("Bob".to_string(), "Charlie".to_string())));
    assert!(pairs.contains(&("David".to_string(), "Eve".to_string())));
}

#[tokio::test]
async fn test_datafusion_two_hop_same_intermediate_node() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Find paths through Charlie specifically
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         WHERE b.name = 'Charlie' \
         RETURN a.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Paths through Charlie: Bob->Charlie->David, Alice->Charlie->David
    assert_eq!(out.num_rows(), 2);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let c_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut pairs = Vec::new();
    for i in 0..out.num_rows() {
        pairs.push((a_names.value(i).to_string(), c_names.value(i).to_string()));
    }

    assert!(pairs.contains(&("Bob".to_string(), "David".to_string())));
    assert!(pairs.contains(&("Alice".to_string(), "David".to_string())));
}

#[tokio::test]
async fn test_datafusion_varlength_projection_correctness() {
    // Test that variable-length path projection correctly handles qualified column names
    // and doesn't accidentally include intermediate node columns
    let out = execute_test_query(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) RETURN b.name",
    )
    .await;

    // Alice can reach: Bob (1-hop), Charlie (1-hop and 2-hop via Bob), David (2-hop via Charlie)
    // Total: 4 results (Bob, Charlie, Charlie, David)
    assert_eq!(out.num_rows(), 4);

    // Verify schema only contains the requested column (now in Cypher dot notation)
    let schema = out.schema();
    let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Should only have the 'b.name' column (Cypher dot notation)
    assert_eq!(column_names.len(), 1);
    assert_eq!(
        column_names[0], "b.name",
        "Expected Cypher dot notation 'b.name' column"
    );

    // Verify no DataFusion qualified names remain (no __)
    for name in &column_names {
        assert!(
            !name.contains("__"),
            "Column name should not contain DataFusion qualifiers: {}",
            name
        );
    }
}

#[tokio::test]
async fn test_datafusion_two_hop_count_paths_per_source() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Count two-hop paths from Alice
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         WHERE a.name = 'Alice' \
         RETURN c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice's two-hop paths: Alice->Bob->Charlie, Alice->Charlie->David
    assert_eq!(out.num_rows(), 2);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let mut counts = HashMap::<String, usize>::new();
    for i in 0..out.num_rows() {
        *counts.entry(names.value(i).to_string()).or_insert(0) += 1;
    }

    assert_eq!(counts.get("Charlie"), Some(&1));
    assert_eq!(counts.get("David"), Some(&1));
}

#[tokio::test]
async fn test_datafusion_filter_on_both_nodes_and_edges() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Filter on both node properties and relationship existence
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         WHERE a.age >= 25 AND a.age <= 30 AND b.age > 30 \
         RETURN a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // a: age 25-30 = Alice(25), Charlie(30), Eve(28)
    // b: age > 30 = Bob(35), David(40)
    // Edges: Alice->Bob, Charlie->David
    assert_eq!(out.num_rows(), 2);

    let a_names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut pairs = Vec::new();
    for i in 0..out.num_rows() {
        pairs.push((a_names.value(i).to_string(), b_names.value(i).to_string()));
    }

    assert!(pairs.contains(&("Alice".to_string(), "Bob".to_string())));
    assert!(pairs.contains(&("Charlie".to_string(), "David".to_string())));
}

#[tokio::test]
async fn test_datafusion_distinct_with_two_hop() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Get distinct source nodes that have two-hop paths
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN DISTINCT a.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Sources with two-hop paths: Alice, Bob, Charlie
    assert_eq!(out.num_rows(), 3);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let result_set: std::collections::HashSet<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    let expected: std::collections::HashSet<String> = ["Alice", "Bob", "Charlie"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    assert_eq!(result_set, expected);
}

#[tokio::test]
async fn test_datafusion_expand_with_both_relationship_and_target_filters() {
    // Query: Find people Alice knows since 2018 who are age 30
    // Alice-[2020]->Bob(35), Alice-[2018]->Charlie(30)
    // Only Charlie matches both filters
    let out = execute_test_query(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS {since_year: 2018}]->(b:Person {age: 30}) \
         RETURN b.name",
    )
    .await;

    assert_eq!(out.num_rows(), 1);
    let names = get_string_column(&out, 0);
    assert_eq!(names[0], "Charlie");
}

// ============================================================================
// ORDER BY Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_order_by_single_column_asc() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: ORDER BY name ascending
    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 5);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify alphabetical order: Alice, Bob, Charlie, David, Eve
    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(names.value(3), "David");
    assert_eq!(names.value(4), "Eve");
}

#[tokio::test]
async fn test_datafusion_order_by_single_column_desc() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: ORDER BY age descending
    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 5);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify descending age order: David(40), Bob(35), Charlie(30), Eve(28), Alice(25)
    assert_eq!(names.value(0), "David");
    assert_eq!(ages.value(0), 40);
    assert_eq!(names.value(1), "Bob");
    assert_eq!(ages.value(1), 35);
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(ages.value(2), 30);
    assert_eq!(names.value(3), "Eve");
    assert_eq!(ages.value(3), 28);
    assert_eq!(names.value(4), "Alice");
    assert_eq!(ages.value(4), 25);
}

#[tokio::test]
async fn test_datafusion_order_by_multiple_columns() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: ORDER BY age DESC, name ASC (secondary sort by name)
    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC, p.name ASC")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 5);

    let _names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // First by age DESC, then by name ASC
    assert_eq!(ages.value(0), 40); // David
    assert_eq!(ages.value(1), 35); // Bob
    assert_eq!(ages.value(2), 30); // Charlie
    assert_eq!(ages.value(3), 28); // Eve
    assert_eq!(ages.value(4), 25); // Alice
}

#[tokio::test]
async fn test_datafusion_order_by_with_limit() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: ORDER BY age DESC LIMIT 3 (top 3 oldest)
    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age DESC LIMIT 3")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should only return 3 rows
    assert_eq!(out.num_rows(), 3);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Top 3 oldest: David(40), Bob(35), Charlie(30)
    assert_eq!(names.value(0), "David");
    assert_eq!(ages.value(0), 40);
    assert_eq!(names.value(1), "Bob");
    assert_eq!(ages.value(1), 35);
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(ages.value(2), 30);
}

#[tokio::test]
async fn test_datafusion_order_by_with_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: Filter then order
    let query =
        CypherQuery::new("MATCH (p:Person) WHERE p.age >= 30 RETURN p.name ORDER BY p.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Age >= 30: Bob(35), Charlie(30), David(40)
    assert_eq!(out.num_rows(), 3);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Alphabetical: Bob, Charlie, David
    assert_eq!(names.value(0), "Bob");
    assert_eq!(names.value(1), "Charlie");
    assert_eq!(names.value(2), "David");
}

#[tokio::test]
async fn test_datafusion_order_by_relationship_query() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Order relationship results by target name
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 5);

    let b_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Targets ordered: Bob, Charlie(x2), David, Eve
    assert_eq!(b_names.value(0), "Bob");
    assert_eq!(b_names.value(1), "Charlie");
    assert_eq!(b_names.value(2), "Charlie");
    assert_eq!(b_names.value(3), "David");
    assert_eq!(b_names.value(4), "Eve");
}

#[tokio::test]
async fn test_datafusion_order_by_two_hop_query() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop with ORDER BY on final target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN a.name, c.name ORDER BY c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 4);

    let c_names = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Final targets ordered: Charlie, David(x2), Eve
    assert_eq!(c_names.value(0), "Charlie");
    assert_eq!(c_names.value(1), "David");
    assert_eq!(c_names.value(2), "David");
    assert_eq!(c_names.value(3), "Eve");
}

#[tokio::test]
async fn test_datafusion_order_by_with_distinct() {
    // Query: DISTINCT with ORDER BY
    let out = execute_test_query(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN DISTINCT b.name ORDER BY b.name",
    )
    .await;

    // Distinct targets: Bob, Charlie, David, Eve
    assert_eq!(out.num_rows(), 4);

    let names = get_string_column(&out, 0);

    // Alphabetical order
    assert_eq!(names, vec!["Bob", "Charlie", "David", "Eve"]);
}

// ============================================================================
// Column Alias Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_return_with_single_alias() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: RETURN with alias
    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name AS person_name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 5);

    // Check that the column is named "person_name" not "p__name"
    let schema = out.schema();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "person_name");

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(!names.value(0).is_empty()); // Has data
}

#[tokio::test]
async fn test_datafusion_return_with_multiple_aliases() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: Multiple columns with aliases
    let query =
        CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name, p.age AS age")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Age > 30: Bob(35), Charlie(30 - excluded), David(40)
    assert_eq!(out.num_rows(), 2);

    // Check column names are aliased
    let schema = out.schema();
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "name");
    assert_eq!(schema.field(1).name(), "age");

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify data
    let mut results: Vec<(String, i64)> = (0..out.num_rows())
        .map(|i| (names.value(i).to_string(), ages.value(i)))
        .collect();
    results.sort_by_key(|r| r.1);

    assert_eq!(results[0], ("Bob".to_string(), 35));
    assert_eq!(results[1], ("David".to_string(), 40));
}

#[tokio::test]
async fn test_datafusion_return_mixed_with_and_without_alias() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: Mix of aliased and non-aliased columns
    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name AS full_name, p.age LIMIT 3")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 3);

    // Check column names
    let schema = out.schema();
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "full_name"); // Aliased
    assert_eq!(schema.field(1).name(), "p.age"); // Not aliased - Cypher dot notation
}

#[tokio::test]
async fn test_datafusion_return_alias_with_relationship() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Alias in relationship query
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.name AS source, b.name AS target \
         ORDER BY source, target \
         LIMIT 3",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 3);

    // Check column names are aliased
    let schema = out.schema();
    assert_eq!(schema.field(0).name(), "source");
    assert_eq!(schema.field(1).name(), "target");

    let sources = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let targets = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // First 3 ordered by source, target
    assert_eq!(sources.value(0), "Alice");
    assert_eq!(targets.value(0), "Bob");
}

#[tokio::test]
async fn test_datafusion_return_alias_with_order_by() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    // Query: Alias with ORDER BY (ORDER BY uses original property reference)
    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.name AS name ORDER BY p.age DESC LIMIT 2")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 2);

    // Check column name is aliased
    let schema = out.schema();
    assert_eq!(schema.field(0).name(), "name");

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Ordered by age DESC: David(40), Bob(35)
    assert_eq!(names.value(0), "David");
    assert_eq!(names.value(1), "Bob");
}

// ============================================================================
// Variable-Length Path Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_varlength_single_hop() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: MATCH (a:Person)-[:KNOWS*1..1]->(b:Person) - equivalent to single hop
    let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS*1..1]->(b:Person) RETURN b.name")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Same as single-hop: Alice→Bob, Alice→Charlie, Bob→Charlie, Charlie→David, David→Eve
    assert_eq!(out.num_rows(), 5);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Collect all target names
    let mut targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    targets.sort();

    // Should have: Bob, Charlie(x2), David, Eve
    assert_eq!(targets, vec!["Bob", "Charlie", "Charlie", "David", "Eve"]);
}

#[tokio::test]
async fn test_datafusion_varlength_two_hops() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: MATCH (a:Person)-[:KNOWS*2..2]->(b:Person) - exactly 2 hops
    let query =
        CypherQuery::new("MATCH (a:Person)-[:KNOWS*2..2]->(b:Person) RETURN a.name, b.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // 2-hop paths: Alice→Bob→Charlie, Alice→Charlie→David, Bob→Charlie→David, Charlie→David→Eve
    assert_eq!(out.num_rows(), 4);

    let sources = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let targets = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Collect all paths
    let mut paths: Vec<(String, String)> = (0..out.num_rows())
        .map(|i| (sources.value(i).to_string(), targets.value(i).to_string()))
        .collect();
    paths.sort();

    assert_eq!(
        paths,
        vec![
            ("Alice".to_string(), "Charlie".to_string()),
            ("Alice".to_string(), "David".to_string()),
            ("Bob".to_string(), "David".to_string()),
            ("Charlie".to_string(), "Eve".to_string()),
        ]
    );
}

#[tokio::test]
async fn test_datafusion_varlength_one_to_two_hops() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) - 1 or 2 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice 1-hop: Bob, Charlie
    // Alice 2-hop: Charlie (via Bob), David (via Charlie)
    // Total: 4 paths (Bob, Charlie x2, David)
    assert_eq!(out.num_rows(), 4);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    targets.sort();

    assert_eq!(targets, vec!["Bob", "Charlie", "Charlie", "David"]);
}

#[tokio::test]
async fn test_datafusion_varlength_with_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length with filter on target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) \
         WHERE b.age > 35 \
         RETURN a.name, b.name, b.age",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Only paths ending at David (age 40)
    // Alice→Bob→David, Bob→David
    let ages = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();

    for i in 0..out.num_rows() {
        assert!(ages.value(i) > 35);
    }
}

#[tokio::test]
async fn test_datafusion_varlength_with_order_by() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length with ORDER BY
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         RETURN b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 4);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Should be ordered alphabetically: Bob, Charlie (x2), David
    assert_eq!(names.value(0), "Bob");
    assert_eq!(names.value(1), "Charlie");
    assert_eq!(names.value(2), "Charlie");
    assert_eq!(names.value(3), "David");
}

#[tokio::test]
async fn test_datafusion_varlength_with_limit() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length with LIMIT
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) \
         RETURN b.name \
         LIMIT 3",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should limit to 3 results
    assert_eq!(out.num_rows(), 3);
}

#[tokio::test]
async fn test_datafusion_varlength_with_distinct() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length with DISTINCT
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         RETURN DISTINCT b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice reaches: Bob, Charlie, David (3 distinct people within 2 hops)
    assert_eq!(out.num_rows(), 3);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    targets.sort();

    assert_eq!(targets, vec!["Bob", "Charlie", "David"]);
}

#[tokio::test]
async fn test_datafusion_varlength_three_hops() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: MATCH (a:Person)-[:KNOWS*3..3]->(b:Person) - exactly 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*3..3]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice 3-hop: Alice→Bob→Charlie→David, Alice→Charlie→David→Eve
    assert_eq!(out.num_rows(), 2);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();
    targets.sort();

    assert_eq!(targets, vec!["David", "Eve"]);
}

#[tokio::test]
async fn test_datafusion_varlength_no_results() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length from Eve (who knows nobody)
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Eve'})-[:KNOWS*1..2]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Eve has no outgoing KNOWS relationships
    assert_eq!(out.num_rows(), 0);
}

#[tokio::test]
async fn test_datafusion_varlength_with_source_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Variable-length with filter on source
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..2]->(b:Person) \
         WHERE a.age > 30 \
         RETURN a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let sources = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // All sources should have age > 30 (Bob: 35, David: 40)
    for i in 0..out.num_rows() {
        let source = sources.value(i);
        assert!(source == "Bob" || source == "David");
    }
}

#[tokio::test]
async fn test_datafusion_varlength_return_source_and_target() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Return both source and target
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*2..2]->(b:Person) \
         RETURN a.name AS source, b.name AS target \
         ORDER BY source, target",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // 2-hop paths: Alice→Bob→Charlie, Alice→Charlie→David, Bob→Charlie→David, Charlie→David→Eve
    assert_eq!(out.num_rows(), 4);

    let sources = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let targets = out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Ordered by source, target
    assert_eq!(sources.value(0), "Alice");
    assert_eq!(targets.value(0), "Charlie");

    assert_eq!(sources.value(1), "Alice");
    assert_eq!(targets.value(1), "David");

    assert_eq!(sources.value(2), "Bob");
    assert_eq!(targets.value(2), "David");

    assert_eq!(sources.value(3), "Charlie");
    assert_eq!(targets.value(3), "Eve");
}

#[tokio::test]
async fn test_datafusion_varlength_count() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Count variable-length paths
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice can reach 4 people within 2 hops
    assert_eq!(out.num_rows(), 4);
}

// ============================================================================
// Aggregation Function Tests
// ============================================================================

#[tokio::test]
async fn test_count_star_all_nodes() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (a:Person) RETURN count(*) AS total")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    let count_col = result
        .column_by_name("total")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(count_col.value(0), 5);
}

#[tokio::test]
async fn test_count_with_filter() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query =
        CypherQuery::new("MATCH (a:Person) WHERE a.age > 30 RETURN count(*) AS older_than_30")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    let count_col = result
        .column_by_name("older_than_30")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    // Bob (35) and David (40) are older than 30
    assert_eq!(count_col.value(0), 2);
}

#[tokio::test]
async fn test_count_property() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) RETURN count(p.name) AS person_count")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    let count_col = result
        .column_by_name("person_count")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(count_col.value(0), 5);
}

#[tokio::test]
async fn test_count_with_grouping() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.city, count(*) AS count ORDER BY p.city")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should have 4 groups: NULL (David), Chicago, New York, San Francisco, Seattle
    assert_eq!(result.num_rows(), 5);

    let city_col = result
        .column_by_name("p.city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let count_col = result
        .column_by_name("count")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    // NULL city: 1 person (David)
    assert!(city_col.is_null(0));
    assert_eq!(count_col.value(0), 1);

    // Chicago: 1 person (Charlie)
    assert_eq!(city_col.value(1), "Chicago");
    assert_eq!(count_col.value(1), 1);

    // New York: 1 person (Alice)
    assert_eq!(city_col.value(2), "New York");
    assert_eq!(count_col.value(2), 1);

    // San Francisco: 1 person (Bob)
    assert_eq!(city_col.value(3), "San Francisco");
    assert_eq!(count_col.value(3), 1);

    // Seattle: 1 person (Eve)
    assert_eq!(city_col.value(4), "Seattle");
    assert_eq!(count_col.value(4), 1);
}

#[tokio::test]
async fn test_count_without_alias_has_descriptive_name() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) RETURN count(*)")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    // Should have column named "count(*)" not "expr" or "count"
    let count_col = result.column_by_name("count(*)");
    assert!(
        count_col.is_some(),
        "Expected column named 'count(*)' but schema is: {:?}",
        result.schema()
    );
}

#[tokio::test]
async fn test_count_property_without_alias_has_descriptive_name() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) RETURN count(p.name)")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    // Should have column named "count(p.name)" not "expr"
    let count_col = result.column_by_name("count(p.name)");
    assert!(
        count_col.is_some(),
        "Expected column named 'count(p.name)' but schema is: {:?}",
        result.schema()
    );
}

#[tokio::test]
async fn test_sum_property() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) RETURN sum(p.age) AS total_age")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    let sum_col = result
        .column_by_name("total_age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    // Sum of ages: 25 + 35 + 30 + 40 + 28 = 158
    assert_eq!(sum_col.value(0), 158);
}

#[tokio::test]
async fn test_sum_with_filter() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query =
        CypherQuery::new("MATCH (p:Person) WHERE p.age >= 30 RETURN sum(p.age) AS total_age")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    let sum_col = result
        .column_by_name("total_age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    // Sum of ages >= 30: 35 + 30 + 40 = 105
    assert_eq!(sum_col.value(0), 105);
}

#[tokio::test]
async fn test_sum_with_grouping() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query =
        CypherQuery::new("MATCH (p:Person) RETURN p.city, sum(p.age) AS total_age ORDER BY p.city")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should have 5 groups: NULL, Chicago, New York, San Francisco, Seattle
    assert_eq!(result.num_rows(), 5);

    let city_col = result
        .column_by_name("p.city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let sum_col = result
        .column_by_name("total_age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    // Verify grouping results (ordered by city, NULL comes first)
    assert!(city_col.is_null(0)); // David: 40 (NULL city)
    assert_eq!(sum_col.value(0), 40);

    assert_eq!(city_col.value(1), "Chicago"); // Charlie: 30
    assert_eq!(sum_col.value(1), 30);

    assert_eq!(city_col.value(2), "New York"); // Alice: 25
    assert_eq!(sum_col.value(2), 25);

    assert_eq!(city_col.value(3), "San Francisco"); // Bob: 35
    assert_eq!(sum_col.value(3), 35);

    assert_eq!(city_col.value(4), "Seattle"); // Eve: 28
    assert_eq!(sum_col.value(4), 28);
}

#[tokio::test]
async fn test_sum_without_alias_has_descriptive_name() {
    let person_batch = create_person_dataset();
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let query = CypherQuery::new("MATCH (p:Person) RETURN sum(p.age)")
        .unwrap()
        .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(result.num_rows(), 1);
    // Should have column named "sum(p.age)" not "expr"
    let sum_col = result.column_by_name("sum(p.age)");
    assert!(
        sum_col.is_some(),
        "Expected column named 'sum(p.age)' but schema is: {:?}",
        result.schema()
    );
}

// ============================================================================
// Disconnected Pattern (Join) Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_disconnected_patterns_cross_join() {
    // Test: MATCH (a:Person), (b:Person) - Cartesian product
    // This creates a cross join between two disconnected patterns
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new(
        "MATCH (a:Person), (b:Person) WHERE a.id = 1 AND b.id = 2 RETURN a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // Should return Alice and Bob
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.num_columns(), 2);

    let a_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(a_names.value(0), "Alice");
    assert_eq!(b_names.value(0), "Bob");
}

#[tokio::test]
async fn test_datafusion_disconnected_patterns_multiple_results() {
    // Test: Multiple disconnected patterns with filtering
    // MATCH (a:Person), (b:Person) WHERE a.age > 30 AND b.age < 30
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query = CypherQuery::new(
        "MATCH (a:Person), (b:Person) WHERE a.age > 30 AND b.age < 30 RETURN a.name, b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // a.age > 30: Bob(35), David(40) = 2 people
    // b.age < 30: Alice(25), Eve(28) = 2 people
    // Cross product: 2 * 2 = 4 combinations
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 2);

    let a_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let b_names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify all combinations exist
    let mut combinations = std::collections::HashSet::new();
    for i in 0..result.num_rows() {
        combinations.insert((a_names.value(i).to_string(), b_names.value(i).to_string()));
    }

    assert!(combinations.contains(&("Bob".to_string(), "Alice".to_string())));
    assert!(combinations.contains(&("Bob".to_string(), "Eve".to_string())));
    assert!(combinations.contains(&("David".to_string(), "Alice".to_string())));
    assert!(combinations.contains(&("David".to_string(), "Eve".to_string())));
}

#[tokio::test]
async fn test_datafusion_mixed_connected_and_disconnected() {
    // Test: Mix of connected pattern and disconnected pattern
    // MATCH (a:Person)-[:KNOWS]->(b:Person), (c:Person) WHERE c.age = 25
    // This should join the relationship traversal with a separate node scan
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person), (c:Person) \
         WHERE c.age = 25 \
         RETURN a.name, b.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // 5 KNOWS relationships * 1 person with age=25 (Alice) = 5 rows
    assert_eq!(result.num_rows(), 5);
    assert_eq!(result.num_columns(), 3);

    let c_names = result
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // All c.name values should be "Alice" (age 25)
    for i in 0..result.num_rows() {
        assert_eq!(c_names.value(i), "Alice");
    }
}

#[tokio::test]
async fn test_datafusion_disconnected_with_distinct() {
    // Test: Disconnected patterns with DISTINCT
    // MATCH (a:Person), (b:Person) WHERE a.id < b.id RETURN DISTINCT a.name
    let config = create_graph_config();
    let person_batch = create_person_dataset();

    let query =
        CypherQuery::new("MATCH (a:Person), (b:Person) WHERE a.id < b.id RETURN DISTINCT a.name")
            .unwrap()
            .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);

    let result = query.execute_datafusion(datasets).await.unwrap();

    // IDs: 1,2,3,4,5
    // Pairs where a.id < b.id: (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)
    // Distinct a names: Alice(1), Bob(2), Charlie(3), David(4)
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 1);

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let name_set: std::collections::HashSet<String> = (0..result.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    let expected: std::collections::HashSet<String> = ["Alice", "Bob", "Charlie", "David"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(name_set, expected);
}
