use itertools::Itertools;
use ndarray_stats::CorrelationExt;

use ndarray::Array2;
use num_traits::Float;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::{
    collections::HashMap,
};

use crate::{EdgeTypeT, Graph, GraphBuilder, NodeT, WalksParameters};

/// Edgetype Transition matrix
///
/// Represents the probability for a random walk to change from edgetype i to edgetype j as M_ij.
/// Automatically handles the conversion between explicit edgetype (`EdgeTypeT`) and indices in the matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgetypeTransitionMatrix {
    /// Each edgetype is a row/column in the matrix
    pub edgetypes: Vec<EdgeTypeT>,
    /// conversion between row/col index and edgetype
    type_to_index: HashMap<EdgeTypeT, usize>, // which row/col corresponds to the edgetype
    /// the actual matrix M
    matrix: Array2<f32>,
}

impl EdgetypeTransitionMatrix {
    pub fn new(edgetypes: Vec<EdgeTypeT>) -> Self {
        let n = edgetypes.len();

        let default_value = 0.0;
        let matrix = Array2::from_elem((n, n), default_value);
        Self::from_matrix(matrix, edgetypes)
    }

    pub fn from_matrix(matrix: Array2<f32>, edgetypes: Vec<EdgeTypeT>) -> Self {
        let n = edgetypes.len();

        assert_eq!(n, matrix.shape()[0]);
        assert_eq!(n, matrix.shape()[1]);

        let type_to_index: HashMap<EdgeTypeT, usize> = edgetypes
            .iter()
            .enumerate()
            .map(|(i, et)| (*et, i))
            .collect();

        Self {
            edgetypes,
            type_to_index,
            matrix,
        }
    }

    /// return the row/column indices corresponding to the edgetypes
    /// TODO: return Result, in case the edgetypes dont exist
    fn get_indices(&self, src_edgetype: EdgeTypeT, dst_edgetype: EdgeTypeT) -> (usize, usize) {
        (
            self.type_to_index[&src_edgetype],
            self.type_to_index[&dst_edgetype],
        )
    }

    /// get the transition probability from one edgtype to another
    /// TODO: return Result, in case the edgetypes dont exist
    pub fn get_probability(&self, src_edgetype: EdgeTypeT, dst_edgetype: EdgeTypeT) -> f32 {
        let (i1, i2) = self.get_indices(src_edgetype, dst_edgetype);
        self.matrix[(i1, i2)]
    }

    /// set the transition probability from one edgtype to another
    pub fn set_probability(
        &mut self,
        src_edgetype: EdgeTypeT,
        dst_edgetype: EdgeTypeT,
        value: f32,
    ) {
        let (i1, i2) = self.get_indices(src_edgetype, dst_edgetype);
        self.matrix[(i1, i2)] = value;
    }

    // estimate this edge transition maitrx from a set of random walks
    pub fn from_walks(walks: Vec<Vec<NodeT>>, graph: &Graph) -> Self {
        // count edgetypes per walk
        let (ets, freqs) = walks_to_edgetype_frequencies(walks, graph);

        // somehow pearson_corr wants floats instead of usize
        let corr = freqs_to_correlation(freqs.mapv(|val| val as f32));
        // let edgetypenames = ets.iter().map(|x| graph.get_edge_type_name_from_edge_type_id(*x)).collect_vec();

        // values range from [-1, 1], but we want proabilities
        // thats why the authors shove the whole matrix trough a sigmoid, converting everything to [0,1]
        let transition = corr.mapv(sigmoid);
        Self::from_matrix(transition, ets)
    }
}

/// Count frequency of edgetypes in each walk
///
/// Returns a 2d matrix of #runs x edgetype
/// Todo: implement as a transform on the random walk?!
fn walks_to_edgetype_frequencies(walks: Vec<Vec<NodeT>>, graph: &Graph) -> (Vec<EdgeTypeT>, Array2<usize>) {
    let all_edge_types = graph.get_unique_edge_type_ids().unwrap();

    let mut row_list: Vec<Vec<usize>> = Vec::new();
    let mut counter: HashMap<EdgeTypeT, usize> = HashMap::new();
    if let Some(ets) = &*graph.edge_types {
        // graph.get
        for walk in walks {
            for (src, dst) in walk.iter().tuple_windows() {
                let edge_id = graph.get_edge_id_from_node_ids(*src, *dst).unwrap();
                let et = ets.ids[edge_id as usize].unwrap();
                let val = counter.entry(et).or_insert(0);
                *val += 1;
            }

            // annoying: turn the Counter into a flat array, ordered by edge type
            let x: Vec<usize> = all_edge_types
                .iter()
                .map(|et| counter.get(et).unwrap_or(&0))
                .cloned()
                .collect();
            row_list.push(x);
            counter.clear();
        }
    }
    let arr = vec_of_vec_to_ndarray(row_list);
    return (all_edge_types, arr);
}

/// turns a list of rows into a 2d array
fn vec_of_vec_to_ndarray<T>(v: Vec<Vec<T>>) -> ndarray::Array2<T> {
    let n_rows = v.len();
    let n_cols = v[0].len();

    let flat_vec = v
        .into_iter()
        .flat_map(|x| x.into_iter())
        .map(|x| x)
        .collect_vec();
    let arr = ndarray::Array::from_shape_vec((n_rows, n_cols), flat_vec).unwrap();
    return arr;
}

fn freqs_to_correlation(v: Array2<f32>) -> Array2<f32> {
    v.t().pearson_correlation().unwrap()
}

#[inline]
pub fn sigmoid<F: Float>(f: F) -> F {
    use std::f32::consts::E;
    let e = F::from(E).unwrap();
    F::one() / (F::one() + e.powf(-f))
}

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

fn load_big_graph() -> Graph {
    let nodes_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/02_intermediate/graphs/spoke_graph_genegene_augmented/nodes/part-0.parquet";
    let edges_pq = "/home/michi/postdoc_seattle/spoke-ingestion_kedro/data/02_intermediate/graphs/spoke_graph_genegene_augmented/edges/part-0.parquet";

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


#[test]
fn test_big() {

    let graph = load_big_graph();
    let walk_params = WalksParameters::new(100).unwrap();
    let res = graph
        // .par_iter_random_walks_singlenode(1000, &walk_params, 1)
        .par_iter_random_walks(1000, &walk_params)
        .unwrap();
    let walks: Vec<_> = res.enumerate().map(|(_i, walk)| walk).collect();

    // println!("{walks:?}");

    let (ets, vvv) = walks_to_edgetype_frequencies(walks, &graph);

    let edgetypenames = ets.iter().map(|x| graph.get_edge_type_name_from_edge_type_id(*x)).collect_vec();
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
    let res = graph
        // .par_iter_random_walks_singlenode(1000, &walk_params, 1)
        .par_iter_random_walks(1000, &walk_params)
        .unwrap();
    let walks: Vec<_> = res.enumerate().map(|(_i, walk)| walk).collect();

    let etm = EdgetypeTransitionMatrix::from_walks(walks, &graph);
    println!("{:?}", etm);
}


#[cfg(test)]
mod tests {
    use super::{freqs_to_correlation, get_dummy_graph, EdgetypeTransitionMatrix};
    use crate::transition_matrix::walks_to_edgetype_frequencies;
    use ndarray::prelude::*;

    #[test]
    fn test_matrix_from_walks() {
        let graph = get_dummy_graph();
        let walks = vec![
            vec![0, 1, 2], 
            vec![0, 1, 2, 0, 2]
        ];
        let (_, vvv) = walks_to_edgetype_frequencies(walks, &graph);

        assert_eq!(Array::from_vec(vec![1, 1]), vvv.slice(s![0, ..]));
        assert_eq!(Array::from_vec(vec![1, 3]), vvv.slice(s![1, ..]));

        let c = freqs_to_correlation(vvv.mapv(|x| x as f32));
        println!("{c:?}")
    }

    #[test]
    /// what happens if theres only a single edge type
    fn test_matrix_from_walks_single_edge_type() {
        let graph = get_dummy_graph();
        let walks = vec![vec![1, 2, 1], vec![2, 0, 2]];
        let (_, vvv) = walks_to_edgetype_frequencies(walks, &graph);

        assert_eq!(Array::from_vec(vec![0, 2]), vvv.slice(s![0, ..]));
        assert_eq!(Array::from_vec(vec![0, 2]), vvv.slice(s![1, ..]));
        for v in vvv {
            println!("{v:?}");
        }
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
}
