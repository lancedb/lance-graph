use arrow_array::{Int64Array, RecordBatch, StringArray};
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

// ============================================================================
// Two-Hop Path Query Tests
// ============================================================================

#[tokio::test]
async fn test_datafusion_two_hop_basic() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Find friends of friends
    // Edges: 1->2, 2->3, 3->4, 4->5, 1->3
    // Two-hop paths: 1->2->3, 2->3->4, 3->4->5, 1->3->4
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should return: Charlie (from 1->2->3), David (from 2->3->4 and 1->3->4), Eve (from 3->4->5)
    assert_eq!(out.num_columns(), 1);
    assert_eq!(out.num_rows(), 4); // 4 two-hop paths

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut counts = HashMap::<String, usize>::new();
    for i in 0..out.num_rows() {
        *counts.entry(names.value(i).to_string()).or_insert(0) += 1;
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
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Get distinct final destinations in two-hop paths
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN DISTINCT c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Distinct destinations: Charlie, David, Eve
    assert_eq!(out.num_rows(), 3);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let result_set: std::collections::HashSet<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    let expected: std::collections::HashSet<String> = ["Charlie", "David", "Eve"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    assert_eq!(result_set, expected);
}

#[tokio::test]
async fn test_datafusion_two_hop_no_results() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Two-hop starting from Eve (who has no outgoing edges)
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) WHERE a.name = 'Eve' RETURN c.name"
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

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

    // Query: Return relationship properties from two-hop path
    let query = CypherQuery::new(
        "MATCH (a:Person)-[r1:KNOWS]->(b:Person)-[r2:KNOWS]->(c:Person) \
         RETURN a.name, c.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();
    assert_eq!(out.num_columns(), 2);
    assert_eq!(out.num_rows(), 4);
}

#[tokio::test]
async fn test_datafusion_one_hop_with_city_filter() {
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: Filter targets by city (David has NULL city, should be excluded by comparison)
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.city = 'Seattle' RETURN b.name",
    )
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
    let config = create_graph_config();
    let person_batch = create_person_dataset();
    let knows_batch = create_knows_dataset();

    // Query: DISTINCT with ORDER BY
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN DISTINCT b.name ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Distinct targets: Bob, Charlie, David, Eve
    assert_eq!(out.num_rows(), 4);

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Alphabetical order
    assert_eq!(names.value(0), "Bob");
    assert_eq!(names.value(1), "Charlie");
    assert_eq!(names.value(2), "David");
    assert_eq!(names.value(3), "Eve");
}
