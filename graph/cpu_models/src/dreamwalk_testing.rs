#[test]
/// Testing the teleport and random walk behavior
/// NOT the skipgram
fn test_dreamwalk_teleport() {
    use crate::*;
    use graph::ms_graphs::load_big_graph;
    use graph::ms_graphs::load_ppi_graph;
    use graph::{EdgeTypeT, EdgetypeTransitionMatrix, WalksParameters};
    use vec_rand::splitmix64;
    let graph = load_big_graph();
    // let graph = load_ppi_graph();

    let edgetypes: Vec<EdgeTypeT> = graph.get_unique_edge_type_ids().unwrap();
    let etm = EdgetypeTransitionMatrix::new(edgetypes);
    let parameters = WalksParameters::new(30).unwrap();
    // .set_edgetype_transition_matrix(etm)
    // .unwrap();

    let embedding_size = 256;
    let window_size = Some(5);
    let clipping_value = Some(6.0);
    let number_of_negative_samples = Some(5);
    let epochs = Some(10);
    let learning_rate = Some(0.1);
    let learning_rate_decay = Some(0.9);
    // let learning_rate_decay = None;
    let alpha = None;
    let maximum_cooccurrence_count_threshold = None;
    let stochastic_downsample_by_degree = Some(true);
    let normalize_learning_rate_by_degree = None;
    let use_scale_free_distribution = None;
    let dtype = None;
    let verbose = Some(true);
    let transformer = IdentifyWalkTransformer {};

    println!("instantiating N2V model");
    let n2v = Node2Vec::new(
        Node2VecModels::SkipGram,
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
    println!("instantiating embeddings randomly");
    let mut v = Vec::new();
    let mut seed = 42;
    for _i in 0..2 {
        let size = (embedding_size) * graph.get_number_of_nodes() as usize;
        let mut x = vec_rand::gen_random_vec_f32(size, seed);
        x = x
            .into_iter()
            .map(|v| (2_f32 * v - 1_f32) * 2.45 / (embedding_size as f32).sqrt())
            .collect();
        seed = splitmix64(seed);
        v.push(x);
    }
    let mut slice2d: Vec<_> = v.iter_mut().map(|x| x.as_mut_slice()).collect();

    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);
    n2v.fit_transform(&graph, &mut slice2d).unwrap();

    println!("{:?}", &slice2d[0][..10]);
    println!("{:?}", &slice2d[1][..10]);

    let shape = (graph.get_number_of_nodes() as usize, embedding_size);
    println!("writing embeeding 1");
    write_embedding("/tmp/embedding.csv", &slice2d[0], shape);
    println!("writing embeeding 2");
    write_embedding("/tmp/embedding2.csv", &slice2d[1], shape);
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
