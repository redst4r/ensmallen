use crate::*;
use express_measures::{
    dot_product_sequential_unchecked, element_wise_addition_inplace,
    element_wise_weighted_addition_inplace, ThreadFloat,
};
use graph::{Graph, NodeT, ThreadDataRaceAware};
use indicatif::ProgressIterator;
use num_traits::AsPrimitive;
use rayon::prelude::*;
use vec_rand::{sample_uniform, splitmix64};

impl<W> Node2Vec<W>
where
    W: WalkTransformer,
{
    /// Computes in the provided slice of embedding the DreamWalk node embedding.
    ///
    /// # Implementative details
    /// This implementation is NOT thread safe, that is, different threads may try
    /// to overwrite each others memory.
    ///
    /// # Arguments
    /// `graph`: &Graph - The graph to embed
    /// `embedding`: &mut [&mut [f32]] - The memory area where to write the embedding.
    pub(crate) fn fit_transform_dreamwalk<F: ThreadFloat + 'static>(
        &self,
        graph: &Graph,
        embedding: &mut [&mut [F]],
    ) -> Result<(), String>
    where
        f32: AsPrimitive<F>,
        NodeT: AsPrimitive<F>,
    {
        let scale_factor = (self.get_embedding_size() as f32).sqrt().as_();
        let mut walk_parameters = self.walk_parameters.clone();
        let mut random_state = splitmix64(self.walk_parameters.get_random_state() as u64);
        let mut learning_rate = self.learning_rate.as_();
        let cv = self.clipping_value.as_();
        let number_of_nodes = graph.get_number_of_nodes();

        let shared_embedding = ThreadDataRaceAware::new(embedding);
        // Depending whether verbosity was requested by the user
        // we create or not a visible progress bar to show the progress
        // in the training epochs.
        let pb = self.get_progress_bar();

        // this computes the update given a single central node and a single context node
        let compute_mini_batch_step = |central_node_embedding: &[F],
                                       cumulative_central_node_gradient: &mut [F],
                                       contextual_node_id: NodeT,
                                       label: F,
                                       learning_rate: F| {
            // println!("{}", central_node_embedding.len());

            // get the contexts output embedding; TODO: why is this mutatably, shouldnt change
            let node_hidden = unsafe {
                &mut (*shared_embedding.get())[1][(contextual_node_id as usize
                    * self.embedding_size)
                    ..((contextual_node_id as usize + 1) * self.embedding_size)]
            };

            let dot: F =
                unsafe { dot_product_sequential_unchecked(node_hidden, central_node_embedding) }
                    / scale_factor;

            if dot > cv || dot < -cv {
                return F::zero();
            }

            // this is the actual derivative of the loss, in case of a positive sample, simply 1-sigm(x)
            let mut variation = (label - sigmoid(dot)) * learning_rate;

            if self.normalize_learning_rate_by_degree {
                variation *= get_node_prior(graph, contextual_node_id, F::one());
            }

            unsafe {
                element_wise_weighted_addition_inplace(
                    node_hidden,
                    central_node_embedding,
                    variation,
                )
            }

            unsafe {
                element_wise_weighted_addition_inplace(
                    cumulative_central_node_gradient,
                    node_hidden,
                    variation,
                )
            };

            let loss = if label == F::one() {
                sigmoid(dot).log(F::from(10.0).unwrap())
            } else {
                sigmoid(-dot).log(F::from(10.0).unwrap())
            };
            loss
        };

        // We start to loop over the required amount of epochs.
        for _ in (0..self.epochs).progress_with(pb) {
            // We update the random state used to generate the random walks
            // and the negative samples.
            random_state = splitmix64(random_state);
            walk_parameters = walk_parameters.set_random_state(Some(random_state as usize));

            // the loss across all walks for the current epoch
            let mut loss_vector = std::iter::repeat(F::zero())
                .take(walk_parameters.get_iterations() as usize)
                .collect::<Vec<_>>();
            // We start to compute the new gradients.
            graph
                // generate random walks
                .par_iter_complete_walks(&walk_parameters)?
                .zip(&mut loss_vector)
                .enumerate()
                .for_each(|(walk_number, (random_walk, loss_field))| {
                    // the accumulated loss for this entire random walk
                    // each central node + context node pair adds to this, as well as negative
                    // samples
                    let mut pos_loss_accumulator = F::zero(); // for positive samples, i.e. center
                                                              // and true context
                    let mut neg_loss_accumulator = F::zero(); // for negative samples, i.e. center
                                                              // and random node
                    (0..random_walk.len()) // iterate over each node in the RW
                        .filter(|&central_index| {
                            // randomly skip the node based on its degree
                            if !self.stochastic_downsample_by_degree {
                                true
                            } else {
                                let degree = unsafe {
                                    graph.get_unchecked_node_degree_from_node_id(
                                        random_walk[central_index as usize],
                                    )
                                };
                                let seed = splitmix64(
                                    random_state + central_index as u64 + walk_number as u64,
                                );
                                degree < sample_uniform(number_of_nodes as _, seed) as _
                            }
                        })
                        // get the context/surrounding tokens, that is their node_ids
                        .map(|central_index| {
                            (
                                &random_walk[central_index.saturating_sub(self.window_size)
                                    ..(central_index + self.window_size).min(random_walk.len())],
                                random_walk[central_index],
                                central_index,
                            )
                        })
                        // for each node+context pair, get the gradient
                        .for_each(|(context, central_node_id, central_index)| {
                            let mut cumulative_central_node_gradient =
                                vec![F::zero(); self.get_embedding_size()];
                            let central_node_embedding = unsafe {
                                &mut (*shared_embedding.get())[0][central_node_id as usize
                                    * self.embedding_size
                                    ..(central_node_id as usize + 1) * self.embedding_size]
                            };
                            // We now compute the gradient relative to the positive
                            // `context` is a slice of node-ids
                            // i.e compute the grad, of each center-context pair
                            context
                                .iter()
                                .copied()
                                .filter(|&context_node_id| context_node_id != central_node_id)
                                .for_each(|context_node_id| {
                                    let loss = compute_mini_batch_step(
                                        &central_node_embedding,
                                        cumulative_central_node_gradient.as_mut_slice(),
                                        context_node_id,
                                        F::one(),
                                        learning_rate,
                                    );
                                    pos_loss_accumulator += loss;
                                });

                            // We compute the gradients relative to the negative classes.
                            if self.use_scale_free_distribution {
                                graph
                                    .iter_random_outbounds_scale_free_node_ids(
                                        self.number_of_negative_samples,
                                        splitmix64(
                                            random_state
                                                + central_index as u64
                                                + walk_number as u64,
                                        ),
                                    )
                                    .filter(|&non_central_node_id| {
                                        non_central_node_id != central_node_id
                                    })
                                    .for_each(|non_central_node_id| {
                                        let loss = compute_mini_batch_step(
                                            &central_node_embedding,
                                            cumulative_central_node_gradient.as_mut_slice(),
                                            non_central_node_id,
                                            F::zero(),
                                            learning_rate,
                                        );
                                        neg_loss_accumulator += loss;
                                    });
                            } else {
                                graph
                                    .iter_random_node_ids(
                                        self.number_of_negative_samples,
                                        splitmix64(
                                            random_state
                                                + central_index as u64
                                                + walk_number as u64,
                                        ),
                                    )
                                    .filter(|&non_central_node_id| {
                                        non_central_node_id != central_node_id
                                    })
                                    .for_each(|non_central_node_id| {
                                        let loss = compute_mini_batch_step(
                                            &central_node_embedding,
                                            cumulative_central_node_gradient.as_mut_slice(),
                                            non_central_node_id,
                                            F::zero(),
                                            learning_rate,
                                        );
                                        neg_loss_accumulator += loss;
                                    });
                            };
                            // apply the accumulated gradient to the central node
                            unsafe {
                                element_wise_addition_inplace(
                                    central_node_embedding,
                                    cumulative_central_node_gradient.as_slice(),
                                )
                            }
                        });

                    // the loss for this particular random walk
                    // actually looks like we dont AVERAGE across neg_samples (the gradients are
                    // not avgd
                    let rw_loss = pos_loss_accumulator + neg_loss_accumulator; //    / (self.number_of_negative_samples as f32).as_();
                    *loss_field = rw_loss;
                });
            learning_rate *= self.learning_rate_decay.as_();

            println!(
                "Loss across walks: {:?}",
                loss_vector
                    .iter()
                    .map(|x| x.to_f32().unwrap().clone())
                    .collect::<Vec<f32>>()
            );
            println!("Learning rate: {}", learning_rate.to_f32().unwrap());
        }
        Ok(())
    }
}
#[allow(dead_code)]
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

use graph::GraphBuilder;
#[allow(dead_code)]
fn two_component_4nodes_each_graph() -> Graph {
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

use graph::EdgetypeTransitionMatrix;
use graph::{EdgeTypeT, WalksParameters};

#[test]
fn test_dreamwalk() {
    let graph = load_big_graph();
    // let graph = two_component_4nodes_each_graph();

    let edgetypes: Vec<EdgeTypeT> = graph.get_unique_edge_type_ids().unwrap();
    let etm = EdgetypeTransitionMatrix::new(edgetypes);
    let parameters = WalksParameters::new(10)
        .unwrap()
        .set_edgetype_transition_matrix(etm)
        .unwrap();

    let embedding_size = 256;
    let window_size = Some(5);
    let clipping_value = None;
    let number_of_negative_samples = Some(5);
    let epochs = Some(150);
    let learning_rate = Some(0.01);
    let learning_rate_decay = Some(0.9);
    let alpha = None;
    let maximum_cooccurrence_count_threshold = None;
    let stochastic_downsample_by_degree = Some(false);
    let normalize_learning_rate_by_degree = None;
    let use_scale_free_distribution = None;
    let dtype = None;
    let verbose = Some(true);
    let transformer = IdentifyWalkTransformer {};

    let n2v = Node2Vec::new(
        Node2VecModels::DreamWalk,
        transformer,
        Some(embedding_size),
        Some(parameters),
        window_size,
        clipping_value,
        number_of_negative_samples,
        epochs,
        learning_rate,
        learning_rate_decay,
        alpha,
        maximum_cooccurrence_count_threshold,
        stochastic_downsample_by_degree,
        normalize_learning_rate_by_degree,
        use_scale_free_distribution,
        dtype,
        verbose,
    )
    .unwrap();

    // this is actually just a 2-element vector (the in/out embeddings)
    // but each vector is a flattened version of the nsample x dim
    let mut v = Vec::new();
    let mut seed = 42;
    for _i in 0..2 {
        let size = (embedding_size) * graph.get_number_of_nodes() as usize;
        let x = vec_rand::gen_random_vec_f32(size, seed);
        seed = splitmix64(seed);
        v.push(x);
    }
    let mut slice2d: Vec<_> = v.iter_mut().map(|x| x.as_mut_slice()).collect();

    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);
    let _r = n2v.fit_transform_dreamwalk(&graph, &mut slice2d).unwrap();

    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);
}
