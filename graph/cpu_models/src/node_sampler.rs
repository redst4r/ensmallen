use graph::{Graph, NodeT, NodeTypeT};

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
