use graph::ms_graphs::load_big_graph;
use graph::*;
#[test]
fn test_big() {
    let graph = load_big_graph();
    let walk_params = WalksParameters::new(100).unwrap();
    let res = graph
        // .par_iter_random_walks_singlenode(1000, &walk_params, 1)
        .par_iter_random_walks(1000, &walk_params)
        .unwrap();
    let walks: Vec<_> = res.enumerate().map(|(_i, walk)| walk).collect();

    let (ets, vvv) = walks_to_edgetype_frequencies(walks, &graph);

    let edgetypenames = ets
        .iter()
        .map(|x| graph.get_edge_type_name_from_edge_type_id(*x))
        .collect_vec();
    println!("{edgetypenames:?}");
    println!("{vvv}");

    let c = freqs_to_correlation(vvv.mapv(|x| x as f32));
    println!("{:?}", c.shape());
    println!("{:?}", c);
}

#[test]
fn test_from_walks() {
    let graph = load_big_graph();
    let walk_params = WalksParameters::new(100).unwrap();
    let res = graph.par_iter_random_walks(1000, &walk_params).unwrap();
    let walks: Vec<_> = res.enumerate().map(|(_i, walk)| walk).collect();

    let etm = EdgetypeTransitionMatrix::from_walks(walks, &graph);
    println!("{:?}", etm);
}

// #[cfg(test)]
// mod tests {
use super::*;
// use crate::transition_matrix::walks_to_edgetype_frequencies;
use ndarray::prelude::*;

/// just a triangle graph with two different edge types
fn get_dummy_graph() -> Graph {
    let mut gb = GraphBuilder::new(Some("name".to_string()), Some(false));
    gb.add_edge(
        "0".to_string(),
        "1".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "1".to_string(),
        "2".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "2".to_string(),
        "0".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.build().unwrap()
}

#[test]
fn test_learn_transition_matrix_from_graph() {
    // let graph = get_triangle_graph_three_types();
    let graph = load_big_graph();

    // let walk_params = WalksParameters::new(100).unwrap();
    let arr =
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
    let edgetypes: Vec<EdgeTypeT> = graph.get_unique_edge_type_ids().unwrap();
    let etm = EdgetypeTransitionMatrix::from_matrix(arr, edgetypes).unwrap();
    let walk_params = WalksParameters::new(10)
        .unwrap()
        .set_edgetype_transition_matrix(etm)
        .unwrap();

    let nwalks = 1000;
    let iterations = 10;
    let e = learn_transition_matrix_from_graph(&graph, walk_params, nwalks, iterations);
    println!("{e:?}");
}
#[test]
fn test_matrix_from_walks() {
    let graph = get_dummy_graph();
    let walks = vec![vec![0, 1, 2], vec![0, 1, 2, 0, 2]];
    let (_, vvv) = walks_to_edgetype_frequencies(walks, &graph);

    assert_eq!(Array::from_vec(vec![1, 1]), vvv.slice(s![0, ..]));
    assert_eq!(Array::from_vec(vec![1, 3]), vvv.slice(s![1, ..]));

    let c = freqs_to_correlation(vvv.mapv(|x| x as f32));
    println!("{c:?}")
}

#[test]
/// what happens if theres only a single edge type
fn test_walks_to_edgetype_frequencies() {
    let graph = get_dummy_graph();
    let walks = vec![vec![1, 2, 1], vec![2, 0, 2]];
    let (_, vvv) = walks_to_edgetype_frequencies(walks, &graph);

    assert_eq!(Array::from_vec(vec![0, 2]), vvv.slice(s![0, ..]));
    assert_eq!(Array::from_vec(vec![0, 2]), vvv.slice(s![1, ..]));
    println!("{vvv}");
}

#[test]
fn test_edgetype_matrix() {
    let edgetypes = vec![10_u16, 20_u16]; // something arbitray
    let mut e = EdgetypeTransitionMatrix::new(edgetypes);

    e.set_probability(10, 10, 1.0);
    e.set_probability(20, 10, 10.0);

    assert_eq!(e.get_probability(10, 20), 0.0);
    assert_eq!(e.get_probability(20, 20), 0.0);
    assert_eq!(e.get_probability(10, 10), 1.0);
    assert_eq!(e.get_probability(20, 10), 10.0);
}

#[test]
fn test_edgetype_matrix_all_probabilities() {
    let arr = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.4, 0.5]).unwrap();
    assert!(EdgetypeTransitionMatrix::from_matrix(arr, vec![10, 20]).is_ok());

    let arr = Array2::from_shape_vec((2, 2), vec![0.0, 0., 0.0, 1.1]).unwrap();
    assert!(EdgetypeTransitionMatrix::from_matrix(arr, vec![10, 20]).is_err());
}
/// transition matrix that cycles through all edge types
/// i.e. in the graph we go in a loop around the triangle
/// n0 -A->n1-B->n2-C->n0 ...
///
/// Note: hard to test, as sometimes we get in places where there's no allowed movde:
/// 1. the first edge is pretty much random (it can follow the ET pattern, since there is no prev edge)
/// 2. we might have taken edge C but goign from n0-C->n2; now we're stuck in node n2 (the et-matrix dicates the C needs to be followed by A, but theres no A edge out of n2)
#[test]
fn test_edgetype_random_walk() {
    // let graph = load_big_graph();
    let graph = get_triangle_graph_three_types();
    let all_edge_types = graph.get_unique_edge_type_ids().unwrap();

    let arr =
        Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
    let etm = EdgetypeTransitionMatrix::from_matrix(arr, all_edge_types).unwrap();
    let walk_params = WalksParameters::new(10)
        .unwrap()
        .set_edgetype_transition_matrix(etm)
        .unwrap();
    let walks: Vec<_> = graph
        .par_iter_random_walks(7, &walk_params)
        .unwrap()
        .collect();

    for (i, w) in walks.iter().enumerate() {
        let eseq = walk_to_edgetype_sequence(&w, &graph);
        println!("{:?}", w);
        let eseq_names = eseq
            .iter()
            .map(|x| graph.get_edge_type_name_from_edge_type_id(*x).unwrap())
            .collect_vec();
        println!("{i}{:?}", eseq_names);
    }
}
/// just a triangle graph with two different edge types
fn get_triangle_graph_three_types() -> Graph {
    let mut gb = GraphBuilder::new(Some("name".to_string()), Some(false));
    gb.add_edge(
        "0".to_string(),
        "1".to_string(),
        Some("A".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "1".to_string(),
        "2".to_string(),
        Some("B".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.add_edge(
        "2".to_string(),
        "0".to_string(),
        Some("C".to_string()),
        Some(1.0),
    )
    .unwrap();
    gb.build().unwrap()
}
