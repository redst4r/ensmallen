use std::collections::HashMap;
use indicatif::{ProgressBar, ProgressStyle};
use statrs::distribution::Geometric;
use rand::distributions::Distribution;
// use pyo3::prelude::*;

use crate::{Graph, NodeT, WalksParameters};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
// use cpu_models::WalkTransformer;

/// Transformer that just picks the first and last node
// #[derive(Debug, Clone, Default)]
// pub struct StartEndWalkTransformer();
// unsafe impl Sync for StartEndWalkTransformer {}
// unsafe impl Send for StartEndWalkTransformer {}

// impl WalkTransformer for StartEndWalkTransformer {
//     type I<'a, T> = impl IndexedParallelIterator<Item = (usize, Vec<T>)> + 'a where
//         Self: 'a,
//         T: Copy + Send + Sync + 'a;

//     fn par_transform_walk<'a, T>(&'a self, i: usize, walk: Vec<T>) -> Self::I<'a, T>
//     where
//         T: Copy + Send + Sync + 'a,
//     {
//         let mut new_walk = Vec::new();
//         new_walk.push(*walk.first().unwrap());
//         new_walk.push(*walk.last().unwrap());
//         vec![(i, new_walk)].into_par_iter()
//     }
// }

/// dummy trait since we cant include from cpu_models
use core::fmt::Debug;
trait WalkTransformer: Send + Sync + Clone + Debug + Default {
    type I<'a, T>: IndexedParallelIterator<Item = (usize, Vec<T>)> + 'a
    where
        Self: 'a,
        T: Copy + Send + Sync + 'a;

    fn par_transform_walk<'a, T>(&'a self, i: usize, walk: Vec<T>) -> Self::I<'a, T>
    where
        T: Copy + Send + Sync + 'a;
}


/// Transformer that emulate pagerank (i.e. randomly hopping back to the starting node)
/// This is equivalent to picking a geometriclly distributed walklength.
#[derive(Debug, Clone, Default)]
pub struct PagerankTransformer {
    alpha: f64,  // prob of continuing the walk (1-teleportation), usually ~0.85
    // walklength_RV: Geometric,
}
unsafe impl Sync for PagerankTransformer {}
unsafe impl Send for PagerankTransformer {}

impl PagerankTransformer {
    pub fn new(alpha: f64) -> Self {
        Self {alpha}
    }
}
impl WalkTransformer for PagerankTransformer {
    type I<'a, T> = impl IndexedParallelIterator<Item = (usize, Vec<T>)> + 'a where
        Self: 'a,
        T: Copy + Send + Sync + 'a;

    fn par_transform_walk<'a, T>(&'a self, i: usize, walk: Vec<T>) -> Self::I<'a, T>
    where
        T: Copy + Send + Sync + 'a,
    {   
        // TODO: not sure if its a great idea to create the RNDs in here 
        // but the trait-signature doesnt allow mut self (so we could store rng in the struct)
        // maybe some workaround: pregenerate the RNDs
        let mut rng = rand::rngs::OsRng;
        // println!("paralell transform! {i}");

        let walk_length = Geometric::new(1.0-self.alpha).unwrap();  // todo: move into struct.constructior
        let mut l = walk_length.sample(&mut rng).ceil() as usize;
        l -= 1; // as l==1 means we didnt move at all and should return the start node

        if l >= walk.len() {
            println!("l: {l} clipped to {}", walk.len());
            l = walk.len() -1;  // clipping to max walk length, introducds some bias
        }   

        let first_node = walk[0]; // *walk.first().unwrap()
        let last_node = walk[l];

        let new_walk = vec![first_node, last_node];
        vec![(i, new_walk)].into_par_iter()
    }
}

// use rand::prelude::*;
pub struct SparseVector {
    pub p: HashMap<NodeT, f64>
}
impl SparseVector {

    pub fn new() -> Self {
        Self {p: HashMap::new() }
    }

    pub fn multiply_constant(&mut self, alpha: f64){
        for (_k, v) in self.p.iter_mut() {
            *v *= alpha; 
        }
    }
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

pub struct PagerankParams {
    alpha: f64,
    iterations: usize,
    max_walk_length: usize
}
impl PagerankParams {
    pub fn new(    alpha: f64,iterations: usize, max_walk_length: usize) -> Self {
        Self {alpha, iterations, max_walk_length}
    }
}

pub struct PagerankResult {
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
pub fn get_progressbar(total: u64) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise} ETA {eta}] {bar:40.cyan/blue} {pos}/{len} {per_sec}")
            // .unwrap()
            .progress_chars("##-"),
    );
    bar
}

pub fn pagerank_all(G: &Graph, params: &PagerankParams) -> Vec<PagerankResult> {
    pagerank_for_nodes(G, G.get_node_ids(), params)
}

pub fn pagerank_for_nodes(G: &Graph, nodes: Vec<NodeT>, params: &PagerankParams) -> Vec<PagerankResult> {

    let pb = get_progressbar(nodes.len() as u64);
    let mut freqs = Vec::new();
    for (counter, start_node) in nodes.into_iter().enumerate() {
        let f = pagerank_single_node(&G, start_node, params);
        freqs.push(f);

        if counter % 100 == 0 {
            pb.inc(100)
        }
    };
    freqs
}

pub fn pagerank_single_node(G: &Graph, start_node: NodeT, params: &PagerankParams) -> PagerankResult {

    let walk_params = WalksParameters::new(params.max_walk_length as u64).unwrap();

    let res = G.par_iter_random_walks_singlenode(params.iterations as u32, &walk_params, start_node).unwrap();

    let frequencies = if false {
        let q = res.map(|x| {
            let mut r = rand::rngs::OsRng;
            let walk_length = Geometric::new(1.0-params.alpha).unwrap();
            let mut l = walk_length.sample(&mut r).ceil() as usize;
            l -= 1; // as l==1 means we didnt move at all and should return the start node
            if l >= params.max_walk_length {
                println!("l: {l} clipped to {}", params.max_walk_length);
                l = params.max_walk_length -1;  // clipping to max walk length, introducds some bias
            }   
            x[l]
        });
        let mut end_nodes = Vec::new();
        q.collect_into_vec(&mut end_nodes);
        
        let mut frequencies: HashMap<NodeT, usize> = HashMap::new();
        for n in end_nodes {
            let v = frequencies.entry(n).or_insert(0);
            *v += 1;
        }
        frequencies
    } else {

        let pr_transformer = PagerankTransformer::new(params.alpha);

        let pr_walk = res.enumerate().flat_map(|(i, walk)| {
            // println!("paralell walk! {i}");
            pr_transformer.par_transform_walk(i, walk)
        });


        // from https://stackoverflow.com/questions/70096640/how-can-i-create-a-hashmap-using-rayons-parallel-fold
        // parallel fold into a Hashmap to count
        let frequencies: HashMap<NodeT, usize> = pr_walk
            .fold(HashMap::new, |mut acc, (i, rw)| {
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

        // let x: Vec<_> = pr_walk.collect();
        // let mut frequencies: HashMap<NodeT, usize> = HashMap::new();m

        // for (_i, rw) in x {
        //     let last_node = rw.last().unwrap();

        //     let v = frequencies.entry(*last_node).or_insert(0);
        //     *v += 1;

        // };

        // for n in pr_walk.collect_vec_list() {
        //     let last_node = n.last().unwrap();
        //     let v = frequencies.entry(n).or_insert(0);
        //     *v += 1;
        // }  
        frequencies
    };
    return PagerankResult{ frequencies }
}


pub fn psev_embedding(G: &Graph, node_weights:  &HashMap<NodeT, f64>, params: &PagerankParams) -> SparseVector {
    
    assert!(node_weights.values().sum::<f64>() == 1.0);

    let mut accumulator = SparseVector::new();

    for (node, weight) in node_weights {
        let pr = pagerank_single_node(G, *node, params).normalize();
        let mut ve = SparseVector {p: pr};
        ve.multiply_constant(*weight);
        accumulator.add(ve);
    }
    accumulator
}




impl Graph {
    pub fn pagerank_single_node(&self, start_node: NodeT, alpha: f64, iterations: usize, max_walk_length: usize) -> PagerankResult {
        let params = PagerankParams::new(alpha, iterations, max_walk_length);
        pagerank_single_node(self, start_node,  &params)
    }
}