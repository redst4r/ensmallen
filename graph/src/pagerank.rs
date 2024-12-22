use std::collections::HashMap;
use statrs::distribution::Geometric;
use rand::distributions::Distribution;

use crate::{Graph, NodeT, WalksParameters};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

/// transforming a random walk to emulate PageRank, i.e. geometrically 
/// distributed walk_length
/// 
/// 1. for each RW, sample its walklength depending on alpha
/// 2. prune the RW to that length
/// 3. return first and last node only
fn pagerank_transform_rw<T: Copy + Send + Sync>(i: usize, walk: Vec<T>, alpha: f64) -> Vec<(usize, Vec<T>)>{

    let mut rng = rand::rngs::OsRng;
    let walk_length = Geometric::new(1.0-alpha).unwrap();  // todo: move into struct.constructior
    let mut l = walk_length.sample(&mut rng).ceil() as usize;
    l -= 1; // as l==1 means we didnt move at all and should return the start node

    if l >= walk.len() {
        println!("l: {l} clipped to {}", walk.len());
        l = walk.len() -1;  // clipping to max walk length, introducds some bias
    }

    let first_node = walk[0]; // *walk.first().unwrap()
    let last_node = walk[l];

    let new_walk = vec![first_node, last_node];
    vec![(i, new_walk)]
}

// use core::fmt::Debug;

#[derive(Debug, Clone, Default)]
struct SparseVector {
    pub p: HashMap<NodeT, f64>
}
impl SparseVector {

    pub fn new() -> Self {
        Self {p: HashMap::new() }
    }

    // scalar multiplication of each component
    pub fn multiply_constant(&mut self, alpha: f64){
        for (_k, v) in self.p.iter_mut() {
            *v *= alpha; 
        }
    }
    // elementwise addition
    pub fn add(&mut self, other: Self){
        for (k, v) in other.p {
            let to_update = self.p.entry(k).or_insert(0.0);
            *to_update += v
        }
    }
}

/// convenience function, return to pyton directly
// impl pyo3::IntoPy<pyo3::PyObject> for SparseVector {
//     fn into_py(self, py: pyo3::Python<'_>) -> pyo3::PyObject {
//         self.p.to_object( py )
//     }
// }

/// The basic pagerank parameters
struct PagerankParams {
    /// exploration probability: 1-teleport
    alpha: f64,
    /// how many random walks to do to estimate the Pagerank vector
    iterations: usize,
    /// number of random walk steps to take at most
    max_walk_length: usize
}
impl PagerankParams {
    pub fn new(alpha: f64,iterations: usize, max_walk_length: usize) -> Self {
        Self {alpha, iterations, max_walk_length}
    }
}


struct PagerankResult {
    /// how often was each node visited
    pub frequencies: HashMap<NodeT, usize>
}
impl PagerankResult {
    pub fn normalize(&self) -> HashMap<NodeT, f64> {
        let total: f64 = self.frequencies.values().sum::<usize>() as f64;

        let mut normed: HashMap<NodeT, f64> = HashMap::new();
        for (k,v) in self.frequencies.iter() {
            normed.insert(*k, (*v as f64)/ total);
        }
        normed
    }
}

/// returns a progress bar instance with standard formatting
// fn get_progressbar(total: u64) -> ProgressBar {
//     let bar = ProgressBar::new(total);
//     bar.set_style(
//         ProgressStyle::default_bar()
//             .template("[{elapsed_precise} ETA {eta}] {bar:40.cyan/blue} {pos}/{len} {per_sec}")
//             // .unwrap()
//             .progress_chars("##-"),
//     );
//     bar
// }

// fn pagerank_all(g: &Graph, params: &PagerankParams) -> Vec<PagerankResult> {
//     pagerank_for_nodes(g, g.get_node_ids(), params)
// }

// fn pagerank_for_nodes(g: &Graph, nodes: Vec<NodeT>, params: &PagerankParams) -> Vec<PagerankResult> {

//     let pb = get_progressbar(nodes.len() as u64);
//     let mut freqs = Vec::new();
//     for (counter, start_node) in nodes.into_iter().enumerate() {
//         let f = pagerank_single_node(&g, start_node, params);
//         freqs.push(f);

//         if counter % 100 == 0 {
//             pb.inc(100)
//         }
//     };
//     freqs
// }


/// Estimates Pagerank starting from a single node `start_node` using random walks
fn pagerank_single_node(g: &Graph, start_node: NodeT, params: &PagerankParams) -> PagerankResult {

    assert!(start_node < g.get_number_of_nodes(), "NodeID not in graph");
    let walk_params = WalksParameters::new(params.max_walk_length as u64).unwrap();
    let res = g.par_iter_random_walks_singlenode(params.iterations as u32, &walk_params, start_node).unwrap();

    // transform the RWs (picking a gemoetrically distributed walklength)
    // let pr_transformer = PagerankTransformer::new(params.alpha);

    let pr_walk = res.enumerate().flat_map(|(i, walk)| {
        // println!("paralell walk! {i}");
        // pr_transformer.par_transform_walk(i, walk)
        pagerank_transform_rw(i, walk, params.alpha) //.into_par_iter()
    });

    // from https://stackoverflow.com/questions/70096640/how-can-i-create-a-hashmap-using-rayons-parallel-fold
    // parallel fold into a Hashmap to count
    let frequencies: HashMap<NodeT, usize> = pr_walk
        .fold(HashMap::new, |mut acc, (_i, rw)| {
            let lastnode = rw.last().unwrap();
            *acc.entry(*lastnode).or_insert(0) += 1;
            acc
        })
        .reduce_with(|mut m1, m2| {
            for (k, v) in m2 {
                *m1.entry(k).or_default() += v;
            }
            m1
        })
        .unwrap();

    PagerankResult{ frequencies }
}

/// Estimates the PSEV vector for given node weights.
/// Essentialyl a linear combination of the nodes' pagerank
/// 
/// 1. Do Pagerank for each node -> PR_i
/// 2. weighted sum across all Pagerank vectors  \sum_i w_i PR_i
/// 
fn psev_embedding(g: &Graph, node_weights:  &HashMap<NodeT, f64>, params: &PagerankParams) -> SparseVector {

    // assert!(node_weights.values().sum::<f64>() == 1.0);  # hard to float imprecsission
    let total_sum = node_weights.values().sum::<f64>();
    if !(0.9999999..=1.0000001).contains(&total_sum) {
        println!("Warning: weight sum is not 1:  {total_sum}")
    }

    // TODO assert all nodes are in the graph

    let mut accumulator = SparseVector::new();

    for (node, weight) in node_weights {
        let pr = pagerank_single_node(g, *node, params).normalize();
        let mut ve = SparseVector {p: pr};
        ve.multiply_constant(*weight);
        accumulator.add(ve);
    }
    accumulator
}

impl Graph {
    /// Estimate the Pagerank vector of the given `start_node` using random walks.
    pub fn pagerank_single_node(&self, start_node: NodeT, alpha: f64, iterations: usize, max_walk_length: usize) -> HashMap<NodeT, usize> {
        let params = PagerankParams::new(alpha, iterations, max_walk_length);
        let res = pagerank_single_node(self, start_node,  &params);
        res.frequencies
    }

    /// Estimate the PSEV vector of the given `node_weights` using random walks.
    // todo: node-weights should really be a ref, but pyo3 doesnt like refs to HashMap
    pub fn psev_estimation(&self, node_weights: HashMap<NodeT, f64>, alpha: f64, iterations: usize, max_walk_length: usize) -> HashMap<NodeT, f64> {

        // filter out nodes not in the graph
        let filter_node_weight: HashMap<_,_> = node_weights
            .into_iter().filter(|(k, _v)| self.get_node_ids().contains(k)).collect();

        // renormalize node_weights (in case they arent)
        let total: f64 = filter_node_weight.values().sum();
        let normed_weights: HashMap<NodeT, f64> = filter_node_weight.into_iter().map(|(k, v)| (k, v/total) ).collect();

        let params = PagerankParams::new(alpha, iterations, max_walk_length);
        let psev = psev_embedding(self, &normed_weights, &params);
        psev.p
    }

    /// Estimate the PSEV vector of the given `node_weights` using random walks.
    /// Here we use the node-names rather than ids.
    // todo: node-weights should really be a ref, but pyo3 doesnt like refs to HashMap
    pub fn psev_estimation_nodenames(&self, node_weights: HashMap<String, f64>, alpha: f64, iterations: usize, max_walk_length: usize) -> HashMap<String, f64> {

        // filter out nodes not in the graph
        let filter_node_weight: HashMap<_,_> = node_weights
            .into_iter().filter(|(k, _v)| self.has_node_name(k)).collect();

        // renormalize node_weights (in case they arent)
        let total: f64 = filter_node_weight.values().sum();
        let normed_weights: HashMap<String, f64> = filter_node_weight.into_iter().map(|(k, v)| (k, v/total) ).collect();

        // we need the node_ids rather than names though
        let name_to_id: HashMap<String, NodeT> = self.get_node_names().iter().enumerate().map(|(i, name)| (name.clone(), i as NodeT)).collect();
        let id_to_name: HashMap<NodeT, String> = self.get_node_names().iter().enumerate().map(|(i, name)| (i as NodeT, name.clone())).collect();

        let normed_weigths_nodeid: HashMap<NodeT, f64>  = normed_weights.iter().map(|(name, weight)| (name_to_id[name], *weight)).collect();

        let params = PagerankParams::new(alpha, iterations, max_walk_length);
        let psev = psev_embedding(self, &normed_weigths_nodeid, &params);

        //translate back into nodenames
        let psev_nodenames: HashMap<String, f64> = psev.p.iter().map(|(i, psev)| (id_to_name[i].clone(), *psev)).collect();
        psev_nodenames
    }
}
