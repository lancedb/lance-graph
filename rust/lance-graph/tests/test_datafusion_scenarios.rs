use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use lance_graph::config::GraphConfig;
use lance_graph::query::CypherQuery;
use std::collections::HashMap;
use std::sync::Arc;

struct TestGraph {
    config: GraphConfig,
    datasets: HashMap<String, RecordBatch>,
}

async fn execute_query(graph: TestGraph, cypher: &str) -> RecordBatch {
    CypherQuery::new(cypher)
        .unwrap()
        .with_config(graph.config)
        .execute_datafusion(graph.datasets)
        .await
        .unwrap()
}

// ============================================================================
// Graph: Animals
// Nodes: Animal (id, name, species, legs)
// Data: Ant(6 legs), Bird(2), Cat(4), Dog(4), Elephant(4)
// ============================================================================
fn animals_graph() -> TestGraph {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("species", DataType::Utf8, false),
        Field::new("legs", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Ant", "Bird", "Cat", "Dog", "Elephant",
            ])),
            Arc::new(StringArray::from(vec![
                "Insect", "Avian", "Feline", "Canine", "Mammal",
            ])),
            Arc::new(Int64Array::from(vec![6, 2, 4, 4, 4])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Animal".to_string(), batch);

    let config = GraphConfig::builder()
        .with_node_label("Animal", "id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Planets
// Nodes: Planet (id, name, discovery_year, habitable)
// Data: Mercury(1631), Mars(1659), Kepler-22b(2011, habitable), Neptune(1846), TRAPPIST-1d(2017, habitable)
// ============================================================================
fn planets_graph() -> TestGraph {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("discovery_year", DataType::Int64, false),
        Field::new("habitable", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "Mercury",
                "Mars",
                "Kepler-22b",
                "Neptune",
                "TRAPPIST-1d",
            ])),
            Arc::new(Int64Array::from(vec![1631, 1659, 2011, 1846, 2017])),
            Arc::new(BooleanArray::from(vec![false, false, true, false, true])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Planet".to_string(), batch);

    let config = GraphConfig::builder()
        .with_node_label("Planet", "id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Books & Authors
// Nodes: Author (id, name, country), Book (id, title, year, genre)
// Edges: WROTE (author_id, book_id, rating)
// Data: Terry Pratchett(UK)->Good Omens(9), Neil Gaiman(UK)->American Gods(8),
//       Isaac Asimov(US)->Foundation(10), Terry Pratchett->The Last Continent(7)
// ============================================================================
fn books_graph() -> TestGraph {
    let author_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("country", DataType::Utf8, false),
    ]));
    let author_batch = RecordBatch::try_new(
        author_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                "Terry Pratchett",
                "Neil Gaiman",
                "Isaac Asimov",
            ])),
            Arc::new(StringArray::from(vec!["UK", "UK", "US"])),
        ],
    )
    .unwrap();

    let book_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("year", DataType::Int64, false),
        Field::new("genre", DataType::Utf8, false),
    ]));
    let book_batch = RecordBatch::try_new(
        book_schema,
        vec![
            Arc::new(Int64Array::from(vec![10, 11, 12, 13])),
            Arc::new(StringArray::from(vec![
                "Good Omens",
                "American Gods",
                "Foundation",
                "The Last Continent",
            ])),
            Arc::new(Int64Array::from(vec![1990, 2001, 1951, 1998])),
            Arc::new(StringArray::from(vec![
                "Fantasy", "Fantasy", "Sci-Fi", "Fantasy",
            ])),
        ],
    )
    .unwrap();

    let wrote_schema = Arc::new(Schema::new(vec![
        Field::new("author_id", DataType::Int64, false),
        Field::new("book_id", DataType::Int64, false),
        Field::new("rating", DataType::Int64, false),
    ]));
    let wrote_batch = RecordBatch::try_new(
        wrote_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 1])),
            Arc::new(Int64Array::from(vec![10, 11, 12, 13])),
            Arc::new(Int64Array::from(vec![9, 8, 10, 7])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Author".to_string(), author_batch);
    datasets.insert("Book".to_string(), book_batch);
    datasets.insert("WROTE".to_string(), wrote_batch);

    let config = GraphConfig::builder()
        .with_node_label("Author", "id")
        .with_node_label("Book", "id")
        .with_relationship("WROTE", "author_id", "book_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Airports & Routes
// Nodes: Airport (code, city)
// Edges: ROUTE (src_airport_code, dst_airport_code, stops)
// Data: SFO->SEA(1 stop), SFO->DEN(2), SEA->LAX(1), DEN->LAX(0)
// ============================================================================
fn airports_graph() -> TestGraph {
    let airport_schema = Arc::new(Schema::new(vec![
        Field::new("code", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
    ]));
    let airport_batch = RecordBatch::try_new(
        airport_schema,
        vec![
            Arc::new(StringArray::from(vec!["SFO", "LAX", "SEA", "DEN"])),
            Arc::new(StringArray::from(vec![
                "San Francisco",
                "Los Angeles",
                "Seattle",
                "Denver",
            ])),
        ],
    )
    .unwrap();

    let route_schema = Arc::new(Schema::new(vec![
        Field::new("src_airport_code", DataType::Utf8, false),
        Field::new("dst_airport_code", DataType::Utf8, false),
        Field::new("stops", DataType::Int64, false),
    ]));
    let route_batch = RecordBatch::try_new(
        route_schema,
        vec![
            Arc::new(StringArray::from(vec!["SFO", "SFO", "SEA", "DEN"])),
            Arc::new(StringArray::from(vec!["SEA", "DEN", "LAX", "LAX"])),
            Arc::new(Int64Array::from(vec![1, 2, 1, 0])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Airport".to_string(), airport_batch);
    datasets.insert("ROUTE".to_string(), route_batch);

    let config = GraphConfig::builder()
        .with_node_label("Airport", "code")
        .with_relationship("ROUTE", "src_airport_code", "dst_airport_code")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Courses & Professors
// Nodes: Course (id, name, level), Professor (id, name)
// Edges: TAUGHT_BY (course_id, professor_id)
// Data: Intro Rust(100)<-Dr. Ada, Distributed Systems(400)<-Dr. Hopper, Algorithms(200)<-Dr. Ada
// ============================================================================
fn courses_graph() -> TestGraph {
    let course_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("level", DataType::Int64, false),
    ]));
    let course_batch = RecordBatch::try_new(
        course_schema,
        vec![
            Arc::new(Int64Array::from(vec![101, 102, 103])),
            Arc::new(StringArray::from(vec![
                "Intro Rust",
                "Distributed Systems",
                "Algorithms",
            ])),
            Arc::new(Int64Array::from(vec![100, 400, 200])),
        ],
    )
    .unwrap();

    let professor_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let professor_batch = RecordBatch::try_new(
        professor_schema,
        vec![
            Arc::new(Int64Array::from(vec![201, 202])),
            Arc::new(StringArray::from(vec!["Dr. Ada", "Dr. Hopper"])),
        ],
    )
    .unwrap();

    let taught_schema = Arc::new(Schema::new(vec![
        Field::new("course_id", DataType::Int64, false),
        Field::new("professor_id", DataType::Int64, false),
        Field::new("term", DataType::Utf8, false),
    ]));
    let taught_batch = RecordBatch::try_new(
        taught_schema,
        vec![
            Arc::new(Int64Array::from(vec![101, 102, 103])),
            Arc::new(Int64Array::from(vec![201, 202, 201])),
            Arc::new(StringArray::from(vec!["Fall", "Spring", "Spring"])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Course".to_string(), course_batch);
    datasets.insert("Professor".to_string(), professor_batch);
    datasets.insert("TAUGHT_BY".to_string(), taught_batch);

    let config = GraphConfig::builder()
        .with_node_label("Course", "id")
        .with_node_label("Professor", "id")
        .with_relationship("TAUGHT_BY", "course_id", "professor_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Engineers, Teams & Tools
// Nodes: Engineer (id, name), Team (id, name), Tool (id, name)
// Edges: MEMBER_OF (engineer_id, team_id), USES (team_id, tool_id)
// Data: Lin->Storage->Rust, Mira->Infra->Scala, Mira->Infra->Kotlin, Zed->Infra->Scala, Zed->Infra->Kotlin
// ============================================================================
fn engineers_graph() -> TestGraph {
    let engineer_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let engineer_batch = RecordBatch::try_new(
        engineer_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Lin", "Mira", "Zed"])),
        ],
    )
    .unwrap();

    let team_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let team_batch = RecordBatch::try_new(
        team_schema,
        vec![
            Arc::new(Int64Array::from(vec![10, 11])),
            Arc::new(StringArray::from(vec!["Storage", "Infra"])),
        ],
    )
    .unwrap();

    let tool_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let tool_batch = RecordBatch::try_new(
        tool_schema,
        vec![
            Arc::new(Int64Array::from(vec![20, 21, 22])),
            Arc::new(StringArray::from(vec!["Rust", "Scala", "Kotlin"])),
        ],
    )
    .unwrap();

    let member_schema = Arc::new(Schema::new(vec![
        Field::new("engineer_id", DataType::Int64, false),
        Field::new("team_id", DataType::Int64, false),
    ]));
    let member_batch = RecordBatch::try_new(
        member_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 11, 11])),
        ],
    )
    .unwrap();

    let uses_schema = Arc::new(Schema::new(vec![
        Field::new("team_id", DataType::Int64, false),
        Field::new("tool_id", DataType::Int64, false),
    ]));
    let uses_batch = RecordBatch::try_new(
        uses_schema,
        vec![
            Arc::new(Int64Array::from(vec![10, 11, 11])),
            Arc::new(Int64Array::from(vec![20, 21, 22])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Engineer".to_string(), engineer_batch);
    datasets.insert("Team".to_string(), team_batch);
    datasets.insert("Tool".to_string(), tool_batch);
    datasets.insert("MEMBER_OF".to_string(), member_batch);
    datasets.insert("USES".to_string(), uses_batch);

    let config = GraphConfig::builder()
        .with_node_label("Engineer", "id")
        .with_node_label("Team", "id")
        .with_node_label("Tool", "id")
        .with_relationship("MEMBER_OF", "engineer_id", "team_id")
        .with_relationship("USES", "team_id", "tool_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Suppliers & Parts
// Nodes: Supplier (id, name), Part (id, name, category)
// Edges: SUPPLIES (supplier_id, part_id, lead_time)
// Data: Nordic->Rotor(Hardware, 4 days), Skyline->Bolt(Hardware, 5), Apex->Gasket(Consumable, 7)
// ============================================================================
fn suppliers_graph() -> TestGraph {
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let supplier_batch = RecordBatch::try_new(
        supplier_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![
                "Nordic Components",
                "Skyline Parts",
            ])),
        ],
    )
    .unwrap();

    let part_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let part_batch = RecordBatch::try_new(
        part_schema,
        vec![
            Arc::new(Int64Array::from(vec![101, 102, 103])),
            Arc::new(StringArray::from(vec!["Rotor", "Sensor", "Bolt"])),
            Arc::new(StringArray::from(vec![
                "Hardware",
                "Electronics",
                "Hardware",
            ])),
        ],
    )
    .unwrap();

    let supplies_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_id", DataType::Int64, false),
        Field::new("part_id", DataType::Int64, false),
        Field::new("lead_time", DataType::Int64, false),
    ]));
    let supplies_batch = RecordBatch::try_new(
        supplies_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2])),
            Arc::new(Int64Array::from(vec![101, 102, 103])),
            Arc::new(Int64Array::from(vec![4, 7, 5])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Supplier".to_string(), supplier_batch);
    datasets.insert("Part".to_string(), part_batch);
    datasets.insert("SUPPLIES".to_string(), supplies_batch);

    let config = GraphConfig::builder()
        .with_node_label("Supplier", "id")
        .with_node_label("Part", "id")
        .with_relationship("SUPPLIES", "supplier_id", "part_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Museums
// Nodes: Museum (id, name, established)
// Data: Uffizi(1581), Louvre(1793), MoMA(1929), Guggenheim(1959)
// ============================================================================
fn museums_graph() -> TestGraph {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("established", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                "Louvre",
                "Metropolitan Museum",
                "Prado",
                "Uffizi",
            ])),
            Arc::new(StringArray::from(vec![
                "Paris", "New York", "Madrid", "Florence",
            ])),
            Arc::new(Int64Array::from(vec![1793, 1870, 1819, 1581])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Museum".to_string(), batch);

    let config = GraphConfig::builder()
        .with_node_label("Museum", "id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Languages
// Nodes: Language (id, name, family, speakers_millions)
// Edges: RELATED_TO (src_language_id, dst_language_id, strength) - UNDIRECTED
// Data: English(Germanic, 400M)<->German(Germanic, 90M) strength=90,
//       German<->French(Romance, 80M) strength=70
// ============================================================================
fn languages_graph() -> TestGraph {
    let language_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("family", DataType::Utf8, false),
        Field::new("speakers_millions", DataType::Int64, false),
    ]));
    let language_batch = RecordBatch::try_new(
        language_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["English", "German", "French"])),
            Arc::new(StringArray::from(vec!["Germanic", "Germanic", "Romance"])),
            Arc::new(Int64Array::from(vec![400, 90, 80])),
        ],
    )
    .unwrap();

    let related_schema = Arc::new(Schema::new(vec![
        Field::new("src_language_id", DataType::Int64, false),
        Field::new("dst_language_id", DataType::Int64, false),
        Field::new("strength", DataType::Int64, false),
    ]));
    let related_batch = RecordBatch::try_new(
        related_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![2, 1, 3, 2])),
            Arc::new(Int64Array::from(vec![90, 90, 70, 70])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Language".to_string(), language_batch);
    datasets.insert("RELATED_TO".to_string(), related_batch);

    let config = GraphConfig::builder()
        .with_node_label("Language", "id")
        .with_relationship("RELATED_TO", "src_language_id", "dst_language_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

// ============================================================================
// Graph: Projects & Milestones
// Nodes: Project (id, name, status), Milestone (id, name, sequence, status)
// Edges: HAS (project_id, milestone_id)
// Data: Apollo(Active)->Design(1), Prototype(2), Launch(3); Gemini(Paused)->Design(1), Review(2)
// ============================================================================
fn projects_graph() -> TestGraph {
    let project_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let project_batch = RecordBatch::try_new(
        project_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Apollo", "Gemini"])),
            Arc::new(StringArray::from(vec!["Active", "Paused"])),
        ],
    )
    .unwrap();

    let milestone_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("sequence", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let milestone_batch = RecordBatch::try_new(
        milestone_schema,
        vec![
            Arc::new(Int64Array::from(vec![11, 12, 13, 14])),
            Arc::new(StringArray::from(vec![
                "Design",
                "Prototype",
                "Launch",
                "Review",
            ])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 2])),
            Arc::new(StringArray::from(vec![
                "Active", "Active", "Active", "Active",
            ])),
        ],
    )
    .unwrap();

    let has_schema = Arc::new(Schema::new(vec![
        Field::new("project_id", DataType::Int64, false),
        Field::new("milestone_id", DataType::Int64, false),
    ]));
    let has_batch = RecordBatch::try_new(
        has_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 1, 2, 2])),
            Arc::new(Int64Array::from(vec![11, 12, 13, 11, 14])),
        ],
    )
    .unwrap();

    let mut datasets = HashMap::new();
    datasets.insert("Project".to_string(), project_batch);
    datasets.insert("Milestone".to_string(), milestone_batch);
    datasets.insert("HAS".to_string(), has_batch);

    let config = GraphConfig::builder()
        .with_node_label("Project", "id")
        .with_node_label("Milestone", "id")
        .with_relationship("HAS", "project_id", "milestone_id")
        .build()
        .unwrap();

    TestGraph { config, datasets }
}

#[tokio::test]
async fn test_animals_filter_skip_limit() {
    let graph = animals_graph();
    let result = execute_query(
        graph,
        "MATCH (a:Animal) WHERE a.legs >= 4 RETURN a.name ORDER BY a.name SKIP 1 LIMIT 2",
    )
    .await;

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(names.value(0), "Cat");
    assert_eq!(names.value(1), "Dog");
}

#[tokio::test]
async fn test_planets_complex_boolean_expression() {
    let graph = planets_graph();
    let result = execute_query(
        graph,
        "MATCH (p:Planet) \
         WHERE ((p.discovery_year >= 1900 AND p.discovery_year < 2020) OR p.habitable = true) \
           AND NOT (p.name = 'Mars') \
         RETURN p.name ORDER BY p.discovery_year",
    )
    .await;

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(names.value(0), "Kepler-22b");
    assert_eq!(names.value(1), "TRAPPIST-1d");
}

#[tokio::test]
async fn test_books_relationship_filter() {
    let graph = books_graph();
    let result = execute_query(
        graph,
        "MATCH (a:Author)-[r:WROTE]->(b:Book) \
         WHERE r.rating > 8 \
         RETURN a.name, a.country, r.rating, b.title, b.genre ORDER BY b.title",
    )
    .await;

    assert_eq!(result.num_columns(), 5);
    assert_eq!(result.num_rows(), 2);

    let authors = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let countries = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ratings = result
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let titles = result
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let genres = result
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(authors.value(0), "Isaac Asimov");
    assert_eq!(countries.value(0), "US");
    assert_eq!(ratings.value(0), 10);
    assert_eq!(titles.value(0), "Foundation");
    assert_eq!(genres.value(0), "Sci-Fi");

    assert_eq!(authors.value(1), "Terry Pratchett");
    assert_eq!(countries.value(1), "UK");
    assert_eq!(ratings.value(1), 9);
    assert_eq!(titles.value(1), "Good Omens");
    assert_eq!(genres.value(1), "Fantasy");
}

#[tokio::test]
async fn test_airports_variable_length_distinct() {
    let graph = airports_graph();
    let result = execute_query(
        graph,
        "MATCH (a:Airport)-[:ROUTE*1..2]->(b:Airport) \
         WHERE a.code = 'SFO' \
         RETURN DISTINCT b.code, b.city ORDER BY b.code",
    )
    .await;

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 3);

    let codes = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let cities = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(codes.value(0), "DEN");
    assert_eq!(cities.value(0), "Denver");

    assert_eq!(codes.value(1), "LAX");
    assert_eq!(cities.value(1), "Los Angeles");

    assert_eq!(codes.value(2), "SEA");
    assert_eq!(cities.value(2), "Seattle");
}

#[tokio::test]
async fn test_courses_incoming_edges_with_distinct() {
    let graph = courses_graph();
    let result = execute_query(
        graph,
        "MATCH (p:Professor)<-[:TAUGHT_BY]-(c:Course) \
         WHERE c.level >= 200 \
         RETURN DISTINCT p.name ORDER BY p.name",
    )
    .await;

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let collected: Vec<String> = (0..result.num_rows())
        .map(|i| names.value(i).to_string())
        .collect();

    assert_eq!(collected, vec!["Dr. Ada", "Dr. Hopper"]);
}

#[tokio::test]
async fn test_engineers_two_hop_path_with_filter() {
    let graph = engineers_graph();
    let result = execute_query(
        graph,
        "MATCH (e:Engineer)-[:MEMBER_OF]->(t:Team)-[:USES]->(tool:Tool) \
         WHERE tool.name <> 'Scala' \
         RETURN DISTINCT e.name, t.name, tool.name ORDER BY e.name",
    )
    .await;

    assert_eq!(result.num_columns(), 3);
    assert_eq!(result.num_rows(), 3);

    let engineer_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let team_names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let tool_names = result
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(engineer_names.value(0), "Lin");
    assert_eq!(team_names.value(0), "Storage");
    assert_eq!(tool_names.value(0), "Rust");

    assert_eq!(engineer_names.value(1), "Mira");
    assert_eq!(team_names.value(1), "Infra");
    assert_eq!(tool_names.value(1), "Kotlin");

    assert_eq!(engineer_names.value(2), "Zed");
    assert_eq!(team_names.value(2), "Infra");
    assert_eq!(tool_names.value(2), "Kotlin");
}

#[tokio::test]
async fn test_suppliers_relationship_ordering() {
    let graph = suppliers_graph();
    let result = execute_query(
        graph,
        "MATCH (s:Supplier)-[r:SUPPLIES]->(p:Part) \
         WHERE p.category = 'Hardware' AND r.lead_time <= 5 \
         RETURN s.name, p.name, r.lead_time ORDER BY r.lead_time",
    )
    .await;

    let suppliers = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let parts = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let lead_times = result
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(
        (suppliers.value(0), parts.value(0), lead_times.value(0)),
        ("Nordic Components", "Rotor", 4)
    );
    assert_eq!(
        (suppliers.value(1), parts.value(1), lead_times.value(1)),
        ("Skyline Parts", "Bolt", 5)
    );
}

#[tokio::test]
async fn test_museums_order_by_limit() {
    let graph = museums_graph();
    let result = execute_query(
        graph,
        "MATCH (m:Museum) WHERE m.established <= 1900 RETURN m.name ORDER BY m.established LIMIT 2",
    )
    .await;

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(names.value(0), "Uffizi");
    assert_eq!(names.value(1), "Louvre");
}

#[tokio::test]
async fn test_languages_undirected_relationships() {
    let graph = languages_graph();
    let result = execute_query(
        graph,
        "MATCH (l:Language)-[r:RELATED_TO]-(other:Language) \
         WHERE l.family = 'Germanic' \
         RETURN DISTINCT other.name, other.family, other.speakers_millions, r.strength \
         ORDER BY other.name",
    )
    .await;

    assert_eq!(result.num_columns(), 4);
    assert_eq!(result.num_rows(), 3);

    let names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let families = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let speakers = result
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let strengths = result
        .column(3)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    assert_eq!(names.value(0), "English");
    assert_eq!(families.value(0), "Germanic");
    assert_eq!(speakers.value(0), 400);
    assert_eq!(strengths.value(0), 90);

    assert_eq!(names.value(1), "French");
    assert_eq!(families.value(1), "Romance");
    assert_eq!(speakers.value(1), 80);
    assert_eq!(strengths.value(1), 70);

    assert_eq!(names.value(2), "German");
    assert_eq!(families.value(2), "Germanic");
    assert_eq!(speakers.value(2), 90);
    assert_eq!(strengths.value(2), 90);
}

#[tokio::test]
async fn test_projects_active_milestones() {
    let graph = projects_graph();
    let result = execute_query(
        graph,
        "MATCH (p:Project)-[:HAS]->(m:Milestone) \
         WHERE p.status = 'Active' AND m.sequence >= 2 \
         RETURN p.name, m.name ORDER BY m.sequence",
    )
    .await;

    let project_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let milestone_names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(result.num_rows(), 2);
    assert_eq!(project_names.value(0), "Apollo");
    assert_eq!(milestone_names.value(0), "Prototype");
    assert_eq!(project_names.value(1), "Apollo");
    assert_eq!(milestone_names.value(1), "Launch");
}

#[tokio::test]
async fn test_distinct_composite_keys() {
    let graph = engineers_graph();

    // Without DISTINCT: Would return 4 rows (Mira->Infra->Scala, Mira->Infra->Kotlin,
    //                                         Zed->Infra->Scala, Zed->Infra->Kotlin)
    // With DISTINCT on (team, tool): Should return 3 unique combinations
    //   - (Storage, Rust)
    //   - (Infra, Scala)
    //   - (Infra, Kotlin)
    let result = execute_query(
        graph,
        "MATCH (e:Engineer)-[:MEMBER_OF]->(t:Team)-[:USES]->(tool:Tool) \
         RETURN DISTINCT t.name, tool.name \
         ORDER BY t.name, tool.name",
    )
    .await;

    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.num_rows(), 3);

    let team_names = result
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let tool_names = result
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    assert_eq!(team_names.value(0), "Infra");
    assert_eq!(tool_names.value(0), "Kotlin");

    assert_eq!(team_names.value(1), "Infra");
    assert_eq!(tool_names.value(1), "Scala");

    assert_eq!(team_names.value(2), "Storage");
    assert_eq!(tool_names.value(2), "Rust");

    // Key insight: Even though both Mira and Zed are in Infra team,
    // DISTINCT (t.name, tool.name) deduplicates to show each (team, tool) pair only once
}
