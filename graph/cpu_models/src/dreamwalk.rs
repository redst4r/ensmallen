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
// General comment:
// Most of the DreamWalk details are actually implemented via the `WalkParameter`
// and how it is treated by the Graph.randomWalks functions, in particular:
// - the edgetype-biased Random walk
// - the teleport
// The logic is mostly in graph.get_unchecked_single_walk_from_slice()
//
// What's outside of the random walk framework is the nodetype-biased SkipGram.
// That's what we handle here!
impl<W> Node2Vec<W>
where
    W: WalkTransformer,
{
    fn sample_negative_nodes(
        &self,
        central_index: usize,
        central_node_id: NodeT,
        walk_number: usize,
        random_state: u64,
        graph: &Graph,
        // node_sampler: &NodeSamplerWithinType,
        // use_nodetype_aware_skipgram: bool,
    ) -> Vec<NodeT> {
        // previously, there were two huge if/else statements doing essentially
        // the same, except sampling the nodes differently, each arm returning
        // an iterator (then some idnetical filtering)
        // to remove the redundancy, I factored out the sampling, but you cant
        // return iterators from both arms (opaque type), so instead the
        // samples get instaniated to a Vec<NodeT>.
        // Should be fine, these vectors are at best a few hundred nodes...s
        //
        // the logic with scale free and node-aware
        // (scale=False, nodeaware=False) => simple, alrdy impl
        // (scale=True, nodeaware=False)  => simple, alrdy impl
        // (scale=False, nodeaware=True) =>  just use the nodeSampler
        // (scale=True, nodeaware=True)  => err, not suppoerted (would need to
        // smaple nodetype and degree)
        //
        //
        let use_nodetype_aware_skipgram = false;
        let sampled_nodes: Vec<NodeT> = match (
            self.use_scale_free_distribution,
            use_nodetype_aware_skipgram,
        ) {
            // scale by degree, dont do node-aware
            (true, false) => {
                // PS: this samples negatives according to their degree
                // smart: internally, just picks a random edge and return its src
                graph
                    .iter_random_outbounds_scale_free_node_ids(
                        self.number_of_negative_samples,
                        splitmix64(random_state + central_index as u64 + walk_number as u64),
                    )
                    .collect()
            }
            // dont scale, no node-aware
            (false, false) => graph
                .iter_random_node_ids(
                    self.number_of_negative_samples,
                    splitmix64(random_state + central_index as u64 + walk_number as u64),
                )
                .collect(),
            (true, true) => {
                panic!("not implemented")
            }
            // no scaling, but node-aware skipgram
            (false, true) => {
                todo!("not implemented yet")
                // // get the nodes type
                // let center_type = graph
                //     .get_node_type_ids_from_node_id(central_node_id)
                //     .unwrap()
                //     .unwrap()[0];
                //
                // node_sampler
                //     .sample(
                //         center_type,
                //         self.number_of_negative_samples,
                //         splitmix64(random_state + central_index as u64 + walk_number as u64),
                //     )
                //     .expect("nodetype must exist!")
                //     .collect()
            }
        };

        sampled_nodes
    }
    /// Computes in the provided slice of embedding the Dreamwalk node embedding.
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
        use_node_aware_skipgram: bool,
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
        println!("Hello from fit_transform_dreamwalk");
        println!("Pa: {:?}", walk_parameters);
        println!("LR: {:?}", self.learning_rate);
        println!("LR decay: {:?}", self.learning_rate_decay);
        println!("Node aware skipgram {use_node_aware_skipgram}");
        // Depending whether verbosity was requested by the user
        // we create or not a visible progress bar to show the progress
        // in the training epochs.
        let pb = self.get_progress_bar();

        let mut loss_over_epochs = Vec::new();

        let compute_mini_batch_step = |central_node_embedding: &[F],
                                       cumulative_central_node_gradient: &mut [F],
                                       contextual_node_id: NodeT,
                                       label: F,
                                       learning_rate: F| {
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

            // Sigmoid loss:  ylog(p) + (1-y)log(1-p) and p=sigmoid(dot)
            // due to symmetry: log(1-sigmoid(dot)) = log(1- [1-sigmoid(-dot)]) = log(sigmoid(-dot)
            let loss = if label == F::one() {
                sigmoid(dot).log(F::from(10.0).unwrap())
            } else {
                sigmoid(-dot).log(F::from(10.0).unwrap())
            };
            loss
        };

        // We start to loop over the required amount of epochs.
        for epoch in (0..self.epochs).progress_with(pb) {
            // We update the random state used to generate the random walks
            // and the negative samples.
            random_state = splitmix64(random_state);
            walk_parameters = walk_parameters.set_random_state(Some(random_state as usize));

            // the loss across all walks for the current epoch
            // Note: `par_iter_complete_walks()` runs a RW for each node in the graph
            // and repeats the entire thing `n_iterations` times!
            let mut loss_vector = vec![
                F::zero();
                graph.get_number_of_nodes() as usize
                    * walk_parameters.get_iterations() as usize
            ];

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
                        // get the context/surrounding tokens
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
                            let sampled_nodes = self.sample_negative_nodes(
                                central_index,
                                central_node_id,
                                walk_number,
                                random_state,
                                &graph,
                            );

                            sampled_nodes
                                .into_iter()
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
                            // apply the accumulated gradient to the central node
                            unsafe {
                                element_wise_addition_inplace(
                                    central_node_embedding,
                                    cumulative_central_node_gradient.as_slice(),
                                )
                            }
                        }); // done iterating the contexts

                    // the loss for this particular random walk
                    // actually looks like we dont AVERAGE across neg_samples (the gradients are
                    // not avgd
                    let rw_loss = pos_loss_accumulator + neg_loss_accumulator; //    / (self.number_of_negative_samples as f32).as_();
                    *loss_field = rw_loss;
                }); // done iterating over walks
            learning_rate *= self.learning_rate_decay.as_();

            // just some logging
            let total_loss: f32 = loss_vector
                .iter()
                .map(|x| x.to_f32().unwrap().clone())
                .sum();
            let avg_loss = total_loss / loss_vector.len() as f32;

            loss_over_epochs.push(avg_loss);
            if true {
                println!("Epoch {epoch}");
                println!("Loss across walks: {avg_loss:?}",);
                println!("Learning rate: {}", learning_rate.to_f32().unwrap());
            };
        } // end of epoch
        println!("Loss over epochs {loss_over_epochs:?}");
        Ok(())
    }
}

#[test]
fn test_dreamwalk() {
    use graph::ms_graphs::load_big_graph;
    // use graph::ms_graphs::load_ppi_graph;
    // use graph::EdgetypeTransitionMatrix;
    use graph::{EdgeTypeT, WalksParameters};
    let graph = load_big_graph();
    // let graph = load_ppi_graph();

    // let edgetypes: Vec<EdgeTypeT> = graph.get_unique_edge_type_ids().unwrap();
    // let etm = EdgetypeTransitionMatrix::new(edgetypes);
    let parameters = WalksParameters::new(30)
        .unwrap()
        .set_explore_weight(Some(1.0))
        .unwrap()
        .set_return_weight(Some(1.0))
        .unwrap()
        .set_iterations(Some(10_u32))
        .unwrap()
        .set_normalize_by_degree(Some(true))
        .set_change_node_type_weight(Some(1.0)) // these are multiplicative! i.e. neutral is ==1
        .unwrap()
        .set_change_edge_type_weight(Some(1.0))
        .unwrap();
    // .set_edgetype_transition_matrix(etm)
    // .unwrap();

    let embedding_size = 512;
    let window_size = Some(5);
    let clipping_value = Some(6.0);
    let number_of_negative_samples = Some(10);
    let epochs = Some(10);
    let learning_rate = Some(0.01);
    let learning_rate_decay = Some(0.9);
    let alpha = None;
    let maximum_cooccurrence_count_threshold = None;
    let stochastic_downsample_by_degree = Some(false);
    let normalize_learning_rate_by_degree = Some(false);
    let use_scale_free_distribution = Some(true);
    let dtype = None;
    let verbose = Some(true);
    let transformer = IdentifyWalkTransformer {};

    println!("instantiating N2V model");
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
    let mut embedding = init_embedding(graph.get_number_of_nodes() as usize, embedding_size);
    let mut slice2d: Vec<_> = embedding.iter_mut().map(|x| x.as_mut_slice()).collect();

    println!("instantiating embeddings randomly");
    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);
    n2v.fit_transform(&graph, &mut slice2d).unwrap();

    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);

    let shape = (graph.get_number_of_nodes() as usize, embedding_size);
    println!("writing embedding 1");
    write_embedding("/tmp/embedding.csv", &slice2d[0], shape);
    println!("writing embedding 2");
    write_embedding("/tmp/embedding2.csv", &slice2d[1], shape);
}

/// The input to the fit_transform():
/// A list of length two (context embedding and center embedding)
/// each is a flattened version of a n_nodes x n_feat matrix
fn init_embedding(n_nodes: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut v = Vec::new();
    let mut seed = 42;
    for _i in 0..2 {
        let size = (dim) * n_nodes;
        let mut x = vec_rand::gen_random_vec_f32(size, seed);
        x = x
            .into_iter()
            .map(|v| (2_f32 * v - 1_f32) * 2.45 / (dim as f32).sqrt())
            .collect();
        seed = splitmix64(seed);
        v.push(x);
    }
    v
}
use csv::WriterBuilder;
use ndarray;
use ndarray_csv;
use ndarray_csv::Array2Writer;
use std::fs::File;
fn write_embedding(fname: &str, embedding: &[f32], shape: (usize, usize)) {
    // turn flat emberdding vector into matrix
    let arr2 = ndarray::Array1::from_vec(embedding.into())
        .into_shape_with_order(shape)
        .expect("wrong size provided");

    let file = File::create(fname).unwrap();
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    writer.serialize_array2(&arr2).unwrap();
}
