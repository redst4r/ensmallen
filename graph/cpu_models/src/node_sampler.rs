use graph::{Graph, NodeT, NodeTypeT};
use vec_rand::{sample_f32, sample_from_cumsum};

use std::collections::HashMap;
use vec_rand::sample_uniform_unbiased_simple;
use vec_rand::splitmix64;

// get a mapping of nodetype -> list of node_ids
// to do the nodetype aware skipgram (negative sampling from within the nodetype) efficiently
fn nodetype_to_node_ids(graph: &Graph) -> HashMap<NodeTypeT, Vec<NodeT>> {
    let mut hmap: HashMap<NodeTypeT, Vec<NodeT>> = HashMap::new();
    for nid in graph.get_node_ids() {
        let ntypes = graph // a single node can have multiple types!
            .get_node_type_ids_from_node_id(nid)
            .expect("id must be in graph")
            .expect("node must have a type");

        assert!(ntypes.len() == 1);
        let ntype = ntypes[0];
        let nodelist = hmap.entry(ntype).or_insert(Vec::new());
        nodelist.push(nid);
    }
    hmap
}

/// Samples random nodes from the graph, but constraind to a particular node-type
pub(crate) struct NodeSamplerWithinType {
    hmap: HashMap<NodeTypeT, Vec<NodeT>>,
}

impl NodeSamplerWithinType {
    pub fn new(graph: &Graph) -> Self {
        let hmap = nodetype_to_node_ids(graph);
        NodeSamplerWithinType { hmap }
    }

    /// draw a sample of random nodes with the requested node type
    ///
    /// Will return Err if the nodetype doesnt exist in the graph
    pub fn sample(
        &self,
        nodetype: NodeTypeT,
        n_samples: usize,
        mut random_state: u64,
        // ) -> Result<Vec<NodeT>, String> {
    ) -> Result<impl Iterator<Item = NodeT> + '_, String> {
        match self.hmap.get(&nodetype) {
            Some(candidate_nodes) => {
                // sample n_samples indices in 0..candidate_nodes.len()
                // and fetch the respective node_ids
                let sampled_nodes = (0..n_samples).map(move |_| {
                    random_state = splitmix64(random_state);
                    let index =
                        sample_uniform_unbiased_simple(candidate_nodes.len() as u64, random_state);
                    candidate_nodes[index]
                });
                Ok(sampled_nodes)
            }
            None => Err("nodetype not found".to_string()),
        }
    }
}

#[cfg(test)]
mod testing {
    use super::*;
    use graph::ms_graphs::two_component_4nodes_each_graph;

    /// check basics: right number of samples, all same nodetype
    #[test]
    fn test_nodesample() {
        let graph = two_component_4nodes_each_graph();

        let sampler = NodeSamplerWithinType::new(&graph);
        let ntype = 1;
        let nsamples = 100;
        let samples: Vec<_> = sampler.sample(ntype, nsamples, 42).unwrap().collect();
        assert_eq!(samples.len(), nsamples);
        println!("{:?}", samples);

        let nodetypes: Vec<_> = samples
            .into_iter()
            .map(|n| graph.get_node_type_ids_from_node_id(n).unwrap().unwrap()[0])
            .collect();
        println!("{:?}", nodetypes);
        assert!(nodetypes.into_iter().all(|nt| nt == ntype));
    }

    #[test]
    // make sure that we evenly sampel within the nodetype
    fn test_nodesample_even_distr() {
        // TODO: put some assertions here about the freqs rather than viz inspection
        let graph = two_component_4nodes_each_graph();

        let sampler = NodeSamplerWithinType::new(&graph);
        let ntype = 0;
        let nsamples = 100_000;
        let samples = sampler.sample(ntype, nsamples, 42).unwrap();

        let mut freqs = HashMap::new();
        for s in samples {
            let v = freqs.entry(s).or_insert(0);
            *v += 1;
        }

        println!("{:?}", freqs);
    }
}
/// a group of nodes what we can sample from
pub(crate) struct NodeGroup {
    node_ids: Vec<NodeT>,
    node_degrees_cumsum: Vec<f32>, // cumulative sim of the node-degrees, used for biased sampling
}

impl NodeGroup {
    pub fn new(node_ids: Vec<NodeT>, node_degrees: Vec<usize>) -> Self {
        let node_degrees_cumsum: Vec<f32> = node_degrees
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x as f32;
                Some(*acc)
            })
            .collect();
        Self {
            node_ids,
            node_degrees_cumsum,
        }
    }
    /// A uniform samle from the group of nodes
    pub fn sample_uniform(
        &self,
        n_samples: usize,
        mut random_state: u64,
    ) -> impl Iterator<Item = NodeT> + '_ {
        let sampled_nodes = (0..n_samples).map(move |_| {
            random_state = splitmix64(random_state);
            let index = sample_uniform_unbiased_simple(self.node_ids.len() as u64, random_state);
            self.node_ids[index]
        });
        sampled_nodes
    }

    /// Sample from the group, but more likely to sample high degree nodes
    pub fn sample_degree_biased(
        &self,
        n_samples: usize,
        mut random_state: u64,
    ) -> impl Iterator<Item = NodeT> + '_ {
        let sampled_nodes = (0..n_samples).map(move |_| {
            random_state = splitmix64(random_state);
            let index = sample_from_cumsum(&self.node_degrees_cumsum, random_state);
            self.node_ids[index]
        });
        sampled_nodes
    }
}

#[cfg(test)]
mod testing_nodegroup {
    use super::*;

    /// check basics: right number of samples, all same nodetype
    #[test]
    fn test_uniform_sampler() {
        let nodes = vec![0, 1, 2, 3];
        let degrees = vec![0, 1, 1, 100];
        let ng = NodeGroup::new(nodes, degrees);
        let random_state = 42;
        let sample_unif: Vec<_> = ng.sample_uniform(1000, random_state).collect();
        assert_eq!(sample_unif.len(), 1000);
        // println!("{sample_unif:?}");

        let n_zeros = sample_unif.into_iter().filter(|x| *x == 0).count();
        // println!("n_zeros: {n_zeros}");
        assert!(n_zeros > 240 && n_zeros < 260)
    }
    #[test]
    fn test_biased_sampler() {
        let nodes = vec![0, 1, 2, 3];
        let degrees = vec![0, 1, 1, 100];
        let ng = NodeGroup::new(nodes, degrees);
        let random_state = 42;
        let sample_biased: Vec<_> = ng.sample_degree_biased(1000, random_state).collect();
        assert_eq!(sample_biased.len(), 1000);
        // println!("{sample_biased:?}");

        let n_zeros = sample_biased.iter().filter(|x| **x == 0).count();
        let n_threes = sample_biased.into_iter().filter(|x| *x == 3).count();
        // println!("n_zeros: {n_zeros}");
        // println!("n_three: {n_threes}");
        assert_eq!(n_zeros, 0, "shouldnt sample nodes with degree 0");
        assert!(
            n_threes > 950,
            "high degree node should be the most frequent"
        )
    }
}

/// Samples random nodes from the graph, but constraind to a particular node-type
pub(crate) struct NodeSamplerNew {
    // for each nodetype store the node-ids
    hmap: HashMap<NodeTypeT, NodeGroup>,
}

impl NodeSamplerNew {
    pub fn new(graph: &Graph) -> Self {
        let mut hmap_node_ids: HashMap<NodeTypeT, Vec<NodeT>> = HashMap::new();
        let mut hmap_nodedegree: HashMap<NodeTypeT, Vec<usize>> = HashMap::new();
        for nid in graph.get_node_ids() {
            let ntypes = graph // a single node can have multiple types!
                .get_node_type_ids_from_node_id(nid)
                .expect("id must be in graph")
                .expect("node must have a type");
            assert!(ntypes.len() == 1);
            let ntype = ntypes[0];
            let ndegree = graph
                .get_node_degree_from_node_id(nid)
                .expect("node must be in graph") as usize;

            let nodelist = hmap_node_ids.entry(ntype).or_insert(Vec::new());
            nodelist.push(nid);

            let degreelist = hmap_nodedegree.entry(ntype).or_insert(Vec::new());
            degreelist.push(ndegree);
        }

        let mut hmap: HashMap<NodeTypeT, NodeGroup> = HashMap::new();
        for ntype in hmap_node_ids.keys() {
            let nodeids = hmap_node_ids.get(ntype).unwrap();
            let nodedegrees = hmap_nodedegree.get(ntype).unwrap();
            let group = NodeGroup::new(nodeids.clone(), nodedegrees.clone());
            hmap.insert(*ntype, group);
        }
        Self { hmap: hmap }
    }
    /// sample nodes from the given type uniformly
    pub fn sample_uniform(
        &self,
        nodetype: NodeTypeT,
        n_samples: usize,
        random_state: u64,
    ) -> Result<impl Iterator<Item = NodeT> + '_, String> {
        if let Some(group) = self.hmap.get(&nodetype) {
            Ok(group.sample_uniform(n_samples, random_state))
        } else {
            Err("unknown nodetype requested".to_string())
        }
    }
    /// sample nodes from the given type, preferentially according to node-degree (higer degree
    /// nodes are sampled more)
    pub fn sample_degree_biased(
        &self,
        nodetype: NodeTypeT,
        n_samples: usize,
        random_state: u64,
    ) -> Result<impl Iterator<Item = NodeT> + '_, String> {
        if let Some(group) = self.hmap.get(&nodetype) {
            Ok(group.sample_degree_biased(n_samples, random_state))
        } else {
            Err("unknown nodetype requested".to_string())
        }
    }
}

#[cfg(test)]
mod testing_nodesamplernew {
    use super::*;
    use graph::ms_graphs::two_component_4nodes_each_graph;

    /// check basics: right number of samples, all same nodetype
    #[test]
    fn test_nodesampleuniform() {
        let graph = two_component_4nodes_each_graph();

        let sampler = NodeSamplerNew::new(&graph);
        let ntype = 1;
        let nsamples = 100;
        let samples: Vec<_> = sampler
            .sample_uniform(ntype, nsamples, 42)
            .unwrap()
            .collect();
        assert_eq!(samples.len(), nsamples);
        println!("{:?}", samples);

        let nodetypes: Vec<_> = samples
            .into_iter()
            .map(|n| graph.get_node_type_ids_from_node_id(n).unwrap().unwrap()[0])
            .collect();
        println!("{:?}", nodetypes);
        assert!(nodetypes.into_iter().all(|nt| nt == ntype));
    }

    #[test]
    // make sure that we evenly sampel within the nodetype
    fn test_nodesample_even_distr() {
        // TODO: put some assertions here about the freqs rather than viz inspection
        let graph = two_component_4nodes_each_graph();

        let sampler = NodeSamplerNew::new(&graph);
        let ntype = 0;
        let nsamples = 100_000;
        let samples = sampler.sample_uniform(ntype, nsamples, 42).unwrap();

        let mut freqs = HashMap::new();
        for s in samples {
            let v = freqs.entry(s).or_insert(0);
            *v += 1;
        }

        println!("{:?}", freqs);
    }
}
