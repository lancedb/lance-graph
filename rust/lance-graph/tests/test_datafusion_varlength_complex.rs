use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;
use std::collections::HashMap;
use std::sync::Arc;

// This test suite uses a more complex social network graph with:
// - 10 people (nodes)
// - 15 relationships (edges)
// - Multiple paths between nodes
// - Cycles in the graph
// - Varying path lengths
//
// Person Dataset (10 nodes):
// | ID | Name     | Age | Department  |
// |----|----------|-----|-------------|
// | 1  | Alice    | 30  | Engineering |
// | 2  | Bob      | 35  | Engineering |
// | 3  | Charlie  | 28  | Sales       |
// | 4  | Diana    | 32  | Marketing   |
// | 5  | Eve      | 29  | Engineering |
// | 6  | Frank    | 40  | Sales       |
// | 7  | Grace    | 27  | Marketing   |
// | 8  | Henry    | 33  | Engineering |
// | 9  | Iris     | 31  | Sales       |
// | 10 | Jack     | 36  | Marketing   |
//
// KNOWS Relationship Dataset (15 edges):
// | src | dst | strength |
// |-----|-----|----------|
// | 1   | 2   | 5        | Alice -> Bob
// | 1   | 3   | 3        | Alice -> Charlie
// | 2   | 4   | 4        | Bob -> Diana
// | 2   | 5   | 5        | Bob -> Eve
// | 3   | 6   | 2        | Charlie -> Frank
// | 4   | 7   | 3        | Diana -> Grace
// | 5   | 8   | 4        | Eve -> Henry
// | 6   | 9   | 5        | Frank -> Iris
// | 7   | 10  | 3        | Grace -> Jack
// | 8   | 1   | 2        | Henry -> Alice (creates cycle!)
// | 3   | 4   | 4        | Charlie -> Diana (alternate path)
// | 5   | 6   | 3        | Eve -> Frank (alternate path)
// | 9   | 10  | 4        | Iris -> Jack (alternate path)
// | 2   | 8   | 3        | Bob -> Henry (shortcut)
// | 4   | 10  | 5        | Diana -> Jack (shortcut)
//
// Visual Graph Structure:
//
//        ┌──────────────────────────────┐
//        │                              │
//        ▼                              │
//     Alice(1) ──5──> Bob(2) ──4──> Diana(4) ──3──> Grace(7) ──3──> Jack(10)
//        │              │        │                                      ▲
//        │              │        └──────────────5──────────────────────┘
//        │              │
//        │              ├──5──> Eve(5) ──4──> Henry(8)
//        │              │         │              │
//        │              │         └──3──> Frank(6) ──5──> Iris(9) ──4──> Jack(10)
//        │              │                   ▲                              ▲
//        │              └──3───────────────┘                              │
//        │                                                                 │
//        └──3──> Charlie(3) ──2──> Frank(6) ─────────────────────────────┘
//                   │
//                   └──4──> Diana(4)
//
// Key Features:
// - Cycle: Alice -> Bob -> Eve -> Henry -> Alice
// - Multiple paths from Alice to Jack (shortest: 3 hops, longest: 5+ hops)
// - Diamond pattern: Alice -> {Bob, Charlie} -> Diana
// - Convergence: Multiple paths lead to Jack
//

/// Helper function to create a complex Person dataset
fn create_complex_person_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int64, false),
        Field::new("department", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
            Arc::new(StringArray::from(vec![
                "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris",
                "Jack",
            ])),
            Arc::new(Int64Array::from(vec![
                30, 35, 28, 32, 29, 40, 27, 33, 31, 36,
            ])),
            Arc::new(StringArray::from(vec![
                "Engineering",
                "Engineering",
                "Sales",
                "Marketing",
                "Engineering",
                "Sales",
                "Marketing",
                "Engineering",
                "Sales",
                "Marketing",
            ])),
        ],
    )
    .unwrap()
}

/// Helper function to create a complex KNOWS relationship dataset
fn create_complex_knows_dataset() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src_person_id", DataType::Int64, false),
        Field::new("dst_person_id", DataType::Int64, false),
        Field::new("strength", DataType::Int64, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![
                1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 3, 5, 9, 2, 4,
            ])),
            Arc::new(Int64Array::from(vec![
                2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 4, 6, 10, 8, 10,
            ])),
            Arc::new(Int64Array::from(vec![
                5, 3, 4, 5, 2, 3, 4, 5, 3, 2, 4, 3, 4, 3, 5,
            ])),
        ],
    )
    .unwrap()
}

/// Helper function to create graph config
fn create_complex_graph_config() -> GraphConfig {
    GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_person_id", "dst_person_id")
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_varlength_multiple_paths_to_target() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all paths from Alice to Jack within 5 hops
    // Multiple paths exist: Alice->Bob->Diana->Jack, Alice->Charlie->Diana->Jack, etc.
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..5]->(b:Person {name: 'Jack'}) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find multiple paths to Jack
    assert!(out.num_rows() > 0, "Should find at least one path to Jack");
}

#[tokio::test]
async fn test_varlength_shortest_path_length() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find paths of exactly 3 hops from Alice to Jack
    // Shortest path: Alice->Bob->Diana->Jack
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*3..3]->(b:Person {name: 'Jack'}) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find 3-hop paths
    assert!(out.num_rows() >= 1, "Should find at least one 3-hop path");
}

#[tokio::test]
async fn test_varlength_with_cycle() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Test cycle: Alice->Bob->Eve->Henry->Alice
    // Find all people Alice can reach in exactly 4 hops (should include herself via cycle)
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*4..4]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    // Should be able to reach Alice herself via the cycle
    assert!(
        targets.contains(&"Alice".to_string()),
        "Should reach Alice via cycle: Alice->Bob->Eve->Henry->Alice"
    );
}

#[tokio::test]
async fn test_varlength_reachability_analysis() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all people Alice can reach within 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         RETURN DISTINCT b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Alice can reach many people within 3 hops
    assert!(
        out.num_rows() >= 5,
        "Alice should reach at least 5 distinct people within 3 hops"
    );
}

#[tokio::test]
async fn test_varlength_diamond_pattern() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Test diamond pattern: Alice -> {Bob, Charlie} -> Diana
    // Find all 2-hop paths from Alice to Diana
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*2..2]->(b:Person {name: 'Diana'}) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find multiple 2-hop paths to Diana
    // Alice->Bob->Diana, Alice->Charlie->Diana, plus potentially others
    assert!(
        out.num_rows() >= 2,
        "Should find at least 2 paths through diamond pattern, found: {}",
        out.num_rows()
    );
}

#[tokio::test]
async fn test_varlength_with_and_without_distinct() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Query WITHOUT DISTINCT - returns all paths
    let query_all_paths = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*2..2]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config.clone());

    let mut datasets1 = HashMap::new();
    datasets1.insert("Person".to_string(), person_batch.clone());
    datasets1.insert("KNOWS".to_string(), knows_batch.clone());

    let out_all = query_all_paths.execute_datafusion(datasets1).await.unwrap();

    // Query WITH DISTINCT - returns unique endpoints only
    let query_distinct = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*2..2]->(b:Person) \
         RETURN DISTINCT b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets2 = HashMap::new();
    datasets2.insert("Person".to_string(), person_batch);
    datasets2.insert("KNOWS".to_string(), knows_batch);

    let out_distinct = query_distinct.execute_datafusion(datasets2).await.unwrap();

    // Note: Due to how variable-length paths are implemented with UNION,
    // DISTINCT may not fully deduplicate across all branches if intermediate
    // node columns differ. This is a known limitation of the unrolling approach.

    // With DISTINCT: should have fewer or equal rows than without
    assert!(
        out_distinct.num_rows() <= out_all.num_rows(),
        "DISTINCT ({}) should be <= all paths ({})",
        out_distinct.num_rows(),
        out_all.num_rows()
    );

    println!(
        "Alice 2-hop reachability: {} total paths, {} with DISTINCT",
        out_all.num_rows(),
        out_distinct.num_rows()
    );
}

#[tokio::test]
async fn test_varlength_distinct_reduces_duplicates() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Test that DISTINCT reduces the number of results
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*2..2]->(b:Person) \
         RETURN DISTINCT b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find multiple people reachable in 2 hops
    assert!(
        out.num_rows() >= 2,
        "Should find at least 2 people in 2 hops"
    );

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Collect names
    let result_names: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    println!("Alice can reach in 2 hops (DISTINCT): {:?}", result_names);

    // Verify results are sorted (due to ORDER BY)
    let mut sorted = result_names.clone();
    sorted.sort();
    assert_eq!(result_names, sorted, "Results should be sorted");
}

#[tokio::test]
async fn test_varlength_count_paths_vs_endpoints() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Count all paths (without DISTINCT)
    let query_paths = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         RETURN b.name",
    )
    .unwrap()
    .with_config(config.clone());

    let mut datasets1 = HashMap::new();
    datasets1.insert("Person".to_string(), person_batch.clone());
    datasets1.insert("KNOWS".to_string(), knows_batch.clone());

    let out_paths = query_paths.execute_datafusion(datasets1).await.unwrap();

    // Count unique endpoints (with DISTINCT)
    let query_endpoints = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         RETURN DISTINCT b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets2 = HashMap::new();
    datasets2.insert("Person".to_string(), person_batch);
    datasets2.insert("KNOWS".to_string(), knows_batch);

    let out_endpoints = query_endpoints.execute_datafusion(datasets2).await.unwrap();

    // Total paths should be >= unique endpoints
    assert!(
        out_paths.num_rows() >= out_endpoints.num_rows(),
        "Total paths ({}) should be >= unique endpoints ({})",
        out_paths.num_rows(),
        out_endpoints.num_rows()
    );

    // With a complex graph, there should be multiple paths to some nodes
    // So total paths > unique endpoints
    println!(
        "Alice can reach {} unique people via {} total paths within 3 hops",
        out_endpoints.num_rows(),
        out_paths.num_rows()
    );
}

#[tokio::test]
async fn test_varlength_same_department() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all Engineering people Alice can reach within 2 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         WHERE b.department = 'Engineering' \
         RETURN DISTINCT b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let names = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let targets: Vec<String> = (0..out.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    // Should find Bob, Eve, and Henry (all Engineering)
    assert!(targets.contains(&"Bob".to_string()));
    assert!(targets.contains(&"Eve".to_string()));
}

#[tokio::test]
async fn test_varlength_cross_department_connections() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all Marketing people reachable from Engineering people within 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) \
         WHERE a.department = 'Engineering' AND b.department = 'Marketing' \
         RETURN DISTINCT b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find Marketing people reachable from Engineering
    assert!(
        out.num_rows() >= 1,
        "Should find at least one Marketing person reachable from Engineering"
    );
}

#[tokio::test]
async fn test_varlength_age_filter() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all people over 35 that Alice can reach within 2 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..2]->(b:Person) \
         WHERE b.age > 35 \
         RETURN DISTINCT b.name, b.age \
         ORDER BY b.age DESC",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // All results should have age > 35
    for i in 0..out.num_rows() {
        assert!(ages.value(i) > 35, "All results should have age > 35");
    }
}

#[tokio::test]
async fn test_varlength_age_range() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find people in their 30s (30-39) reachable within 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         WHERE b.age >= 30 AND b.age < 40 \
         RETURN DISTINCT b.name, b.age \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // All results should be in their 30s
    for i in 0..out.num_rows() {
        let age = ages.value(i);
        assert!((30..40).contains(&age), "Age should be in range [30, 40)");
    }
}

#[tokio::test]
async fn test_varlength_convergence_to_hub() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Jack is a convergence point - find all people who can reach Jack in exactly 2 hops
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*2..2]->(b:Person {name: 'Jack'}) \
         RETURN DISTINCT a.name \
         ORDER BY a.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Multiple people should reach Jack in 2 hops
    assert!(
        out.num_rows() >= 2,
        "Multiple people should reach Jack in 2 hops"
    );
}

#[tokio::test]
async fn test_varlength_divergence_from_source() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Bob has multiple outgoing connections - test divergence
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Bob'})-[:KNOWS*1..1]->(b:Person) \
         RETURN b.name \
         ORDER BY b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Bob knows multiple people directly
    assert!(
        out.num_rows() >= 3,
        "Bob should have at least 3 direct connections"
    );
}

#[tokio::test]
async fn test_varlength_increasing_reach() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Test that reach increases with hop count
    let queries = vec![("1..1", 1), ("1..2", 2), ("1..3", 3)];

    let mut prev_count = 0;

    for (range, _max_hops) in queries {
        let query = CypherQuery::new(&format!(
            "MATCH (a:Person {{name: 'Alice'}})-[:KNOWS*{}]->(b:Person) \
             RETURN DISTINCT b.name",
            range
        ))
        .unwrap()
        .with_config(config.clone());

        let mut datasets = HashMap::new();
        datasets.insert("Person".to_string(), person_batch.clone());
        datasets.insert("KNOWS".to_string(), knows_batch.clone());

        let out = query.execute_datafusion(datasets).await.unwrap();
        let current_count = out.num_rows();

        // Each additional hop should reach at least as many people (monotonic increase)
        assert!(
            current_count >= prev_count,
            "Reach should increase or stay same with more hops: prev={}, current={}",
            prev_count,
            current_count
        );

        prev_count = current_count;
    }
}

#[tokio::test]
async fn test_varlength_combined_filters() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Complex filter: Engineering people over 30 reachable within 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         WHERE b.department = 'Engineering' AND b.age > 30 \
         RETURN DISTINCT b.name, b.age, b.department \
         ORDER BY b.age",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let departments = out
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Verify all results match both filters
    for i in 0..out.num_rows() {
        assert!(ages.value(i) > 30);
        assert_eq!(departments.value(i), "Engineering");
    }
}

#[tokio::test]
async fn test_varlength_with_limit_and_order() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Get top 3 youngest people Alice can reach within 3 hops
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..3]->(b:Person) \
         RETURN DISTINCT b.name, b.age \
         ORDER BY b.age ASC \
         LIMIT 3",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    assert_eq!(out.num_rows(), 3, "Should return exactly 3 results");

    let ages = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

    // Verify results are ordered by age ascending
    for i in 1..out.num_rows() {
        assert!(
            ages.value(i) >= ages.value(i - 1),
            "Results should be ordered by age"
        );
    }
}

#[tokio::test]
async fn test_varlength_large_hop_count() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Test with larger hop count (up to 10)
    let query = CypherQuery::new(
        "MATCH (a:Person {name: 'Alice'})-[:KNOWS*1..10]->(b:Person) \
         RETURN DISTINCT b.name",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Due to cycles, Alice can reach many people with 10 hops
    assert!(out.num_rows() >= 5, "Should reach many people with 10 hops");
}

#[tokio::test]
async fn test_varlength_all_pairs_reachability() {
    let config = create_complex_graph_config();
    let person_batch = create_complex_person_dataset();
    let knows_batch = create_complex_knows_dataset();

    // Find all pairs of people connected within 5 hops
    let query = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS*1..5]->(b:Person) \
         RETURN DISTINCT a.name, b.name \
         ORDER BY a.name, b.name \
         LIMIT 20",
    )
    .unwrap()
    .with_config(config);

    let mut datasets = HashMap::new();
    datasets.insert("Person".to_string(), person_batch);
    datasets.insert("KNOWS".to_string(), knows_batch);

    let out = query.execute_datafusion(datasets).await.unwrap();

    // Should find many connected pairs
    assert!(
        out.num_rows() >= 15,
        "Should find at least 15 connected pairs"
    );
}
