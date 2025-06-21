#[allow(dead_code)]
use crate::Graph;
use crate::GraphBuilder;

#[allow(dead_code)]
pub fn load_big_graph() -> Graph {
    let nodes_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/03_slicing/spoke_graph_genegene_augmented/nodes/part-0.parquet";
    let edges_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/03_slicing/spoke_graph_genegene_augmented/edges/part-0.parquet";
    load_graph(nodes_pq, edges_pq)
}

pub fn load_ppi_graph() -> Graph {
    let nodes_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/03_slicing/spoke_graph_ppigenereg_augmented/nodes/part-0.parquet";
    let edges_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/03_slicing/spoke_graph_ppigenereg_augmented/edges/part-0.parquet";
    load_graph(nodes_pq, edges_pq)
}

pub fn load_graph(nodes_pq: &str, edges_pq: &str) -> Graph {
    let graph = Graph::from_parquet(
        nodes_pq.to_string(),
        edges_pq.to_string(),
        "id".to_string(),
        "subject".to_string(),
        "object".to_string(),
        Some("category".to_string()),
        Some("predicate".to_string()),
        None, // edge_weight_col,
        Some(false),
        Some("ggg".to_string()),
    );
    graph
}

/// a graph with two connected components, to see it the walk teleports
/// one drug per component, rest diseases
///  A -[e1] - B
///  |         |
/// [e2]      [e2]       
///  |         |
///  C - [e1]- D
#[allow(dead_code)]
pub fn two_component_4nodes_each_graph() -> Graph {
    let mut gb = GraphBuilder::new(Some("name".to_string()), Some(false));

    gb.add_node("X0".to_string(), Some(vec!["drug".to_string()]))
        .unwrap();
    gb.add_node("X1".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();
    gb.add_node("X2".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();
    gb.add_node("X3".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();

    gb.add_node("Y4".to_string(), Some(vec!["drug".to_string()]))
        .unwrap();
    gb.add_node("Y5".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();
    gb.add_node("Y6".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();
    gb.add_node("Y7".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();

    // first component
    gb.add_edge(
        "X0".to_string(),
        "X1".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "X1".to_string(),
        "X2".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "X2".to_string(),
        "X3".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "X3".to_string(),
        "X0".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();

    // 2nd component
    gb.add_edge(
        "Y4".to_string(),
        "Y5".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "Y5".to_string(),
        "Y6".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "Y6".to_string(),
        "Y7".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "Y7".to_string(),
        "Y4".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();

    gb.build().unwrap()
}

/// a graph with two connected components, to see it the walk teleports
#[allow(dead_code)]
pub fn two_component_graph() -> Graph {
    let mut gb = GraphBuilder::new(Some("name".to_string()), Some(false));

    gb.add_node("0".to_string(), Some(vec!["drug".to_string()]))
        .unwrap();
    gb.add_node("1".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();
    gb.add_node("2".to_string(), Some(vec!["drug".to_string()]))
        .unwrap();
    gb.add_node("3".to_string(), Some(vec!["disease".to_string()]))
        .unwrap();

    gb.add_edge(
        "0".to_string(),
        "1".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "2".to_string(),
        "3".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.build().unwrap()
}
