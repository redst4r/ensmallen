use super::types::Result;
use super::*;
use crate::{EdgeTypeT, Graph, NodeT, WalksParameters};
use itertools::Itertools;
use named_matrix::matrix::AnnMatrix;
use ndarray::{array, Array2};
use ndarray_stats::CorrelationExt;
use num_traits::Float;
use rayon::iter::ParallelIterator;
use std::{collections::HashMap, fs::File, io};

/// Edgetype Transition matrix
///
/// Represents the probability for a random walk to change from edgetype i to edgetype j as M_ij.
/// Automatically handles the conversion between explicit edgetype (`EdgeTypeT`) and indices in the matrix.
/// Actually just a wrapper around a NamedMatrix, enforcing matching rows/columns
#[derive(Debug, Clone, PartialEq)]
#[no_binding]
pub struct EdgetypeTransitionMatrix {
    inner: AnnMatrix<EdgeTypeT, EdgeTypeT, f32>,
}

impl EdgetypeTransitionMatrix {
    pub fn from_matrix(matrix: Array2<f32>, edgetypes: Vec<EdgeTypeT>) -> Result<Self> {
        let n = edgetypes.len();
        if matrix.shape() != [n, n] {
            return Err("wrong matrix shape; needs to match length of edgetypes".to_string());
        }

        // ensure all weights are in [0,1]
        // TODO check that its a stochastic matrix! (not really enforced in dreamwalk though)
        if matrix.iter().all(|x| *x >= 0.0 && *x <= 1.0) {
            let inner = AnnMatrix::from_matrix(matrix, edgetypes.clone(), edgetypes).unwrap();
            Ok(Self { inner })
        } else {
            Err("some values were outside [0,1]".to_string())
        }
    }

    pub(crate) fn from_walks(walks: Vec<Vec<NodeT>>, graph: &Graph) -> Self {
        // count edgetypes per walk
        let (ets, freqs) = walks_to_edgetype_frequencies(walks, graph);
        // println!("Freq matrix: {freqs:?}");
        let corr = freqs_to_correlation(freqs.mapv(|val| val as f32));
        // Nan come from columns with no variation, ndarry.corr puts Nan there instead of 0
        let corr = corr.mapv(|val| if val.is_nan() { 0.0 } else { val });
        // println!("Corr matrix: {corr:?}");
        // values range from [-1, 1], but we want proabilities
        // thats why the authors shove the whole matrix trough a sigmoid, converting everything to [0,1]
        let transition = corr.mapv(sigmoid);

        Self::from_matrix(transition, ets)
            .expect("all values need to be in [0,1], should be the case with sigmoid!")
    }

    pub fn from_file(fname: &str) -> Self {
        let file = File::open(fname).expect("file for Edgetype Matrix must exist");
        let reader = io::BufReader::new(file);

        let inner = AnnMatrix::from_reader(reader).unwrap();
        Self { inner }
    }
    pub fn to_file(&self, fname: &str) {
        let fh = File::create(fname).unwrap();
        let mut writer = io::BufWriter::new(fh);
        self.inner.to_writer(writer);
    }
    /// get the transition probability from one edgtype to another
    /// TODO: return Result, in case the edgetypes dont exist
    pub fn get_probability(&self, src_edgetype: EdgeTypeT, dst_edgetype: EdgeTypeT) -> f32 {
        *self.inner.get_value(&src_edgetype, &dst_edgetype)
    }
}

/// for a walk (sequence of nodes), retrieve the sequence of edgetypes it traverses
/// TODO: tests
/// - empty walk
/// - walk with single node
fn walk_to_edgetype_sequence(walk: &[NodeT], graph: &Graph) -> Vec<EdgeTypeT> {
    let ets = (&*graph.edge_types).as_ref().unwrap(); // to get the edge types

    let eseq: Vec<EdgeTypeT> = walk
        .iter()
        .tuple_windows()
        .map(|(src, dst)| graph.get_edge_id_from_node_ids(*src, *dst).unwrap())
        .map(|edge_id| ets.ids[edge_id as usize].unwrap())
        .collect();
    eseq
}

/// Count frequency of edgetypes in each walk
///
/// Returns a 2d matrix of #runs x edgetype
/// Todo: implement as a transform on the random walk?!
fn walks_to_edgetype_frequencies(
    walks: Vec<Vec<NodeT>>,
    graph: &Graph,
) -> (Vec<EdgeTypeT>, Array2<usize>) {
    let all_edge_types = graph.get_unique_edge_type_ids().unwrap();

    let mut row_list: Vec<Vec<usize>> = Vec::new();
    let mut counter: HashMap<EdgeTypeT, usize> = HashMap::new();
    if let Some(_ets) = &*graph.edge_types {
        for walk in walks {
            for et in walk_to_edgetype_sequence(&walk, graph) {
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

/// Pearson correlation of the columns of a 2D array, (m,n) -> (n,n)
fn freqs_to_correlation(v: Array2<f32>) -> Array2<f32> {
    // edge-case: if there's only one edgetype, pearson_correlation will be [[NaN]]
    // but should really be [[1]]
    if v.shape() == &[1, 1] {
        array![[1.0]]
    } else {
        v.t().pearson_correlation().unwrap()
    }
}

#[inline]
fn sigmoid<F: Float>(f: F) -> F {
    use std::f32::consts::E;
    let e = F::from(E).unwrap();
    F::one() / (F::one() + e.powf(-f))
}

#[test]
fn test_sigmoid() {
    assert_eq!(sigmoid(0.0_f32), 0.5);
    assert!(sigmoid(-10000.0_f32) < 0.000001 && sigmoid(-10000.0_f32) >= 0.0);
    assert!(sigmoid(10000.0_f32) <= 1.0 && sigmoid(10000.0_f32) >= 0.9999);
}

/// Estimate the edgetype transition matrix from the graph
/// iteratively as proposed by DreamWalk.
///     1. simulate random walks
///     2. count edgetype occurences
///     3. turn into a correlation/transition matrix
///     4. repeat at 1. with new walk parameters
pub(crate) fn learn_transition_matrix_from_graph(
    graph: &Graph,
    walk_params: WalksParameters,
    nwalks: usize,
    iterations: usize,
) -> EdgetypeTransitionMatrix {
    assert!(
        !walk_params.is_dreamwalk_walk(),
        "make sure paramters are not edgetype biased"
    );
    let edgetypes: Vec<EdgeTypeT> = graph.get_unique_edge_type_ids().unwrap();
    let matrix = Array2::from_elem((edgetypes.len(), edgetypes.len()), 1.0);
    let etm = EdgetypeTransitionMatrix::from_matrix(matrix, edgetypes).unwrap();
    let mut walk_params = walk_params.set_edgetype_transition_matrix(etm).unwrap();

    fn get_etm(params: &WalksParameters) -> &EdgetypeTransitionMatrix {
        params
            .single_walk_parameters
            .weights
            .edgetype_transition_matrix
            .as_ref()
            .unwrap()
    }

    let matrix_conv_rate = 0.0001; // default from the python code
    for i in 0..iterations {
        let walks: Vec<_> = graph
            .par_iter_random_walks(nwalks as NodeT, &walk_params)
            .unwrap()
            .collect();

        let new_etm = EdgetypeTransitionMatrix::from_walks(walks, graph);

        // cehck convergence
        let last_etm = &get_etm(&walk_params).inner.matrix; // wow, ugly
                                                            // relative difference
        let r = ((&new_etm.inner.matrix - (last_etm)) / last_etm)
            .abs()
            .mean()
            .unwrap();
        println!("Itertation {i}, Relative change: {r}");
        // println!("{}", &new_etm.matrix);

        if r < matrix_conv_rate {
            break;
        }

        // updating for the next iteration
        walk_params = walk_params.set_edgetype_transition_matrix(new_etm).unwrap();
    }

    let res = get_etm(&walk_params).clone(); // again, ugly to get the matrix out of the struct
    res
}

impl Graph {
    /// estimates the edgetype transition matrix (dreamwalk) via a simple random walk
    /// - walk_length:
    /// - nwalks: number of walks in each iteration of the EM estimation
    /// - iterations: repeat the EM estimation this many times (or until convergence)
    pub fn estimate_edgetype_transition_matrix(
        &self,
        walk_length: usize,
        iterations: usize,
        nwalks: usize,
    ) -> HashMap<(EdgeTypeT, EdgeTypeT), f32> {
        let params = WalksParameters::new(walk_length as u64).unwrap();
        let mat = learn_transition_matrix_from_graph(&self, params, nwalks, iterations);
        mat.inner.to_hashmap()
    }

    /// estimate the transition matrix from the graph, write to file as flat CSV
    pub fn estimate_edgetype_transition_matrix_to_file(
        &self,
        walk_length: usize,
        iterations: usize,
        nwalks: usize,
        fname: &str,
    ) {
        let params = WalksParameters::new(walk_length as u64).unwrap();
        let mat = learn_transition_matrix_from_graph(&self, params, nwalks, iterations);
        mat.to_file(fname);
    }
}
