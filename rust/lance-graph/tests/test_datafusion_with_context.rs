use datafusion::execution::context::SessionContext;
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;

#[tokio::test]
async fn test_execute_with_context_csv_simple() {
    // Create temporary CSV files for testing
    let temp_dir = tempfile::tempdir().unwrap();
    let person_csv_path = temp_dir.path().join("persons.csv");
    let knows_csv_path = temp_dir.path().join("knows.csv");

    // Write Person CSV
    std::fs::write(
        &person_csv_path,
        "id,name,age\n\
         1,Alice,28\n\
         2,Bob,34\n\
         3,Carol,29\n\
         4,David,42\n",
    )
    .unwrap();

    // Write KNOWS relationship CSV
    std::fs::write(
        &knows_csv_path,
        "src_id,dst_id,since\n\
         1,2,2020\n\
         2,3,2021\n\
         1,3,2019\n",
    )
    .unwrap();

    // Create graph configuration
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .with_relationship("KNOWS", "src_id", "dst_id")
        .build()
        .unwrap();

    // Create SessionContext and register CSV files
    let ctx = SessionContext::new();

    ctx.register_csv(
        "Person",
        person_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    ctx.register_csv(
        "KNOWS",
        knows_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    // Test 1: Simple node scan with filter
    // Note: No need to manually build catalog - it's automatic!
    let query1 =
        CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age ORDER BY p.age")
            .unwrap()
            .with_config(config.clone());

    let result1 = query1
        .execute_with_datafusion_context(ctx.clone())
        .await
        .unwrap();

    // Should return Bob (34) and David (42)
    assert_eq!(result1.num_rows(), 2);
    assert_eq!(result1.num_columns(), 2);

    // Verify column names use Cypher dot notation
    assert_eq!(result1.schema().field(0).name(), "p.name");
    assert_eq!(result1.schema().field(1).name(), "p.age");

    let names = result1
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let ages = result1
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();

    // Verify first row is Bob (34)
    assert_eq!(names.value(0), "Bob");
    assert_eq!(ages.value(0), 34);

    // Verify second row is David (42)
    assert_eq!(names.value(1), "David");
    assert_eq!(ages.value(1), 42);

    // Test 2: Relationship traversal
    let query2 = CypherQuery::new(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name ORDER BY a.name",
    )
    .unwrap()
    .with_config(config);

    let result2 = query2.execute_with_datafusion_context(ctx).await.unwrap();

    // Should return 3 relationships: Alice->Bob, Alice->Carol, Bob->Carol
    assert_eq!(result2.num_rows(), 3);
    assert_eq!(result2.num_columns(), 2);

    // Verify column names
    assert_eq!(result2.schema().field(0).name(), "a.name");
    assert_eq!(result2.schema().field(1).name(), "b.name");

    let src_names = result2
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let dst_names = result2
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();

    // Collect all relationships
    let relationships: Vec<(String, String)> = (0..result2.num_rows())
        .map(|i| {
            (
                src_names.value(i).to_string(),
                dst_names.value(i).to_string(),
            )
        })
        .collect();

    // Verify we have the expected relationships
    assert!(relationships.contains(&("Alice".to_string(), "Bob".to_string())));
    assert!(relationships.contains(&("Alice".to_string(), "Carol".to_string())));
    assert!(relationships.contains(&("Bob".to_string(), "Carol".to_string())));
}

#[tokio::test]
async fn test_execute_with_context_complex_query() {
    // Create temporary CSV files
    let temp_dir = tempfile::tempdir().unwrap();
    let employee_csv_path = temp_dir.path().join("employees.csv");
    let department_csv_path = temp_dir.path().join("departments.csv");
    let works_in_csv_path = temp_dir.path().join("works_in.csv");

    // Write Employee CSV
    std::fs::write(
        &employee_csv_path,
        "emp_id,name,salary\n\
         101,Alice,75000\n\
         102,Bob,85000\n\
         103,Carol,65000\n\
         104,David,95000\n\
         105,Eve,72000\n",
    )
    .unwrap();

    // Write Department CSV
    std::fs::write(
        &department_csv_path,
        "dept_id,name,budget\n\
         1,Engineering,500000\n\
         2,Sales,300000\n\
         3,HR,200000\n",
    )
    .unwrap();

    // Write WORKS_IN relationship CSV
    std::fs::write(
        &works_in_csv_path,
        "employee_id,department_id,role\n\
         101,1,Engineer\n\
         102,1,Senior Engineer\n\
         103,2,Sales Rep\n\
         104,1,Manager\n\
         105,3,HR Specialist\n",
    )
    .unwrap();

    // Create graph configuration
    let config = GraphConfig::builder()
        .with_node_label("Employee", "emp_id")
        .with_node_label("Department", "dept_id")
        .with_relationship("WORKS_IN", "employee_id", "department_id")
        .build()
        .unwrap();

    // Create SessionContext and register CSV files
    let ctx = SessionContext::new();

    ctx.register_csv(
        "Employee",
        employee_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    ctx.register_csv(
        "Department",
        department_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    ctx.register_csv(
        "WORKS_IN",
        works_in_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    // Query: Find high-earning employees in Engineering department
    let query = CypherQuery::new(
        "MATCH (e:Employee)-[:WORKS_IN]->(d:Department) \
         WHERE d.name = 'Engineering' AND e.salary > 80000 \
         RETURN e.name, e.salary, d.name \
         ORDER BY e.salary DESC",
    )
    .unwrap()
    .with_config(config);

    let result = query.execute_with_datafusion_context(ctx).await.unwrap();

    // Should return David (95000) and Bob (85000) from Engineering
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_columns(), 3);

    // Verify column names use Cypher dot notation
    assert_eq!(result.schema().field(0).name(), "e.name");
    assert_eq!(result.schema().field(1).name(), "e.salary");
    assert_eq!(result.schema().field(2).name(), "d.name");

    let emp_names = result
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let salaries = result
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    let dept_names = result
        .column(2)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();

    // First row: David with highest salary
    assert_eq!(emp_names.value(0), "David");
    assert_eq!(salaries.value(0), 95000);
    assert_eq!(dept_names.value(0), "Engineering");

    // Second row: Bob
    assert_eq!(emp_names.value(1), "Bob");
    assert_eq!(salaries.value(1), 85000);
    assert_eq!(dept_names.value(1), "Engineering");
}

#[tokio::test]
async fn test_execute_with_context_missing_table() {
    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let ctx = SessionContext::new();
    // Note: Not registering any tables!

    let query = CypherQuery::new("MATCH (p:Person) RETURN p.name")
        .unwrap()
        .with_config(config);

    let result = query.execute_with_datafusion_context(ctx).await;

    // Should error because Person table is not registered
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Person") && err_msg.contains("not found"),
        "Error should mention missing Person table: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_execute_with_context_aliases() {
    let temp_dir = tempfile::tempdir().unwrap();
    let person_csv_path = temp_dir.path().join("persons.csv");

    std::fs::write(
        &person_csv_path,
        "id,name,age\n\
         1,Alice,28\n\
         2,Bob,34\n",
    )
    .unwrap();

    let config = GraphConfig::builder()
        .with_node_label("Person", "id")
        .build()
        .unwrap();

    let ctx = SessionContext::new();
    ctx.register_csv(
        "Person",
        person_csv_path.to_str().unwrap(),
        Default::default(),
    )
    .await
    .unwrap();

    // Query with explicit aliases
    let query = CypherQuery::new(
        "MATCH (p:Person) RETURN p.name AS person_name, p.age AS person_age ORDER BY p.age",
    )
    .unwrap()
    .with_config(config);

    let result = query.execute_with_datafusion_context(ctx).await.unwrap();

    assert_eq!(result.num_rows(), 2);

    // Verify explicit aliases are preserved
    assert_eq!(result.schema().field(0).name(), "person_name");
    assert_eq!(result.schema().field(1).name(), "person_age");

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();

    assert_eq!(names.value(0), "Alice");
    assert_eq!(names.value(1), "Bob");
}
