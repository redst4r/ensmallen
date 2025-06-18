use std::collections::HashMap;
use super::types::Result;
use ndarray::{Array2, ArrayView1};
use crate::{NodeT, NodeTypeT};
use vec_rand::sample_f32;



#[derive(Clone, PartialEq, Debug)]
pub struct AnnMatrix {
    /// Each edgetype is a row/column in the matrix
    pub rownames: Vec<NodeT>,
    pub colnames: Vec<NodeT>,
    /// conversion between row/col index and names
    rownames_to_index: HashMap<NodeT, usize>, // which row corresponds to the rnowname
    colnames_to_index: HashMap<NodeT, usize>, // which col corresponds to the colname
    /// the actual matrix M
    matrix: Array2<f32>,
}

impl AnnMatrix {
    // pub fn new(rownames: Vec<NodeT>, colnames: Vec<NodeT>) -> Self {
    //     let n = edgetypes.len();

    //     let default_value = 0.0;
    //     let matrix = Array2::from_elem((n, n), default_value);
    //     Self::from_matrix(matrix, edgetypes).expect("cant fail as all values are ==0")
    // }

    pub fn from_matrix(matrix: Array2<f32>, rownames: Vec<NodeT>, colnames: Vec<NodeT>) -> Result<Self> {
        let n = rownames.len();
        let m = colnames.len();
        if matrix.shape() != [n, m] {
            return Err("wrong matrix shape; needs to match length of row/colnames".to_string());
        }

        let rownames_to_index: HashMap<NodeT, usize> = rownames
            .iter()
            .enumerate()
            .map(|(i, et)| (*et, i))
            .collect();

        let colnames_to_index: HashMap<NodeT, usize> = colnames
            .iter()
            .enumerate()
            .map(|(i, et)| (*et, i))
            .collect();

        // ensure all weights are in [0,1]
        if matrix.iter().all(|x| *x >= 0.0 && *x <= 1.0) {
            Ok(Self {
                rownames,
                colnames,
                rownames_to_index,
                colnames_to_index,
                matrix,
            })
        } else {
            Err("some values were outside [0,1]".to_string())
        }
    }
    fn get_row_index(&self, src_name: NodeT,) -> usize {
        self.rownames_to_index[&src_name]
    }

    fn get_col_index(&self, dst_name: NodeT,) -> usize {
        self.colnames_to_index[&dst_name]
    }

    /// return the row/column indices corresponding to the edgetypes
    /// TODO: return Result, in case the edgetypes dont exist
    fn get_indices(&self, src_name: NodeT, dst_name: NodeT) -> (usize, usize) {
        (
            self.get_row_index(src_name),
            self.get_col_index(dst_name),
        )
    }

    /// get the transition probability from one edgtype to another
    /// TODO: return Result, in case the edgetypes dont exist
    pub fn get_probability(&self, src_name: NodeT, dst_name: NodeT) -> f32 {
        let (i1, i2) = self.get_indices(src_name, dst_name);
        self.matrix[(i1, i2)]
    }

    /// set the transition probability from one edgtype to another
    pub fn set_probability(
        &mut self, src_name: NodeT, dst_name: NodeT, value: f32) {
        let (i1, i2) = self.get_indices(src_name, dst_name);
        self.matrix[(i1, i2)] = value;
    }

    pub fn get_row(&self, src_name: NodeT) -> ArrayView1<f32>{
        let row_ix = self.get_row_index(src_name);
        let r = self.matrix.row(row_ix);
        r
    }
}

#[cfg(test)]
mod test_annmatrix {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_annmatrix_basic_getters_setters() {
        let rownames = vec![10, 20, 30];
        let colnames = vec![2, 4];
        let elements = array![[0.1, 0.2], 
                              [0.3, 0.4], 
                              [0.5, 0.6]];
        let mut m = AnnMatrix::from_matrix(elements, rownames, colnames).unwrap();

        assert_eq!(m.get_probability(20, 4), 0.4);
        m.set_probability(20, 4, 1.0);
        assert_eq!(m.get_probability(20, 4), 1.0);

        // row slicing
        assert_eq!(m.get_row(10), array![0.1, 0.2]);
    }
}


/// For each node type, what are the possible teleports
// TODO grrr, dont really want to clone that x_X
#[derive(Debug, Clone, PartialEq)]
pub struct TeleportMatrix {
    teleports: HashMap<NodeTypeT, AnnMatrix>
}

impl TeleportMatrix {
    /// constructs an empty teleport matrix
    pub fn new() -> Self {
        let teleports = HashMap::new();
        Self {
            teleports
        }
    }

    pub fn add(&mut self, nodetype: NodeTypeT, matrix: AnnMatrix) {
        self.teleports.insert(nodetype, matrix);
    }

    // for the given node/type, sample a new node to teleport to
    pub fn sample_teleport(&self, nodeid: NodeT, nodetype: NodeTypeT, random_state: u64) -> Result<NodeT> {
        if let Some(matrix) = self.teleports.get(&nodetype) {

            // warning: sample_f32 mutates the row, make sure this doesnt propagate back into `matrix`!! i.e. keep the `to_vec`
            let mut row = matrix.get_row(nodeid).to_vec();
            let ix = sample_f32(&mut row, random_state);
            let sampled_nodeid = matrix.rownames[ix];
            Ok(sampled_nodeid)

        } else {
            Err("unknown nodetype".to_string())
        }
    }
}


#[cfg(test)]
mod test_teleport {
    use ndarray::array;
    use vec_rand::splitmix64;

    use super::*;
    #[test]
    // single node type, 3 nodes
    fn test_teleport_sample() {
        let rownames = vec![10, 20, 30];
        let colnames = vec![10, 20 ,30];

        // first node always teleports to sencond
        // second always teleports to 1st
        // third has 50/50
        let elements = array![[0.0, 1.0, 0.0], 
                              [1.0, 0.0, 0.0], 
                              [0.5, 0.5, 0.0]];
        let matrix = AnnMatrix::from_matrix(elements, rownames, colnames).unwrap();
        let mut tm = TeleportMatrix::new();
        let nodetype = 35; // arbitrary
        tm.add(nodetype, matrix);
        let mut random_state = 42;
        
        // first node always teleports to sencond
        assert_eq!(
            tm.sample_teleport(10, nodetype, random_state),
            Ok(20)
        );

        random_state = splitmix64(random_state);
        // second always teleports to 1st
        assert_eq!(
            tm.sample_teleport(20, nodetype, random_state),
            Ok(10)
        );

        // third has 50/50
        // so lets do a few rounds
        // PS we're not checking the exact proabilities
        for _ in 0..100 {
            random_state = splitmix64(random_state);
            let sampled = tm.sample_teleport(30, nodetype, random_state);
            assert!(vec![Ok(10), Ok(20)].contains(&sampled));
            // println!("{sampled:?}")
        }
    }
}



#[cfg(test)]
mod test_teleport_walks {
    use crate::{
        AnnMatrix, EdgetypeTransitionMatrix, NodeT, TeleportMatrix, TeleportParameters,
        WalksParameters,
    };
    use itertools::Itertools;
    use ndarray::array;
    use rayon::iter::ParallelIterator;
    use vec_rand::{sample_f32, splitmix64};
    type Walk = Vec<NodeT>; // convenience: a seuqence of nodes
    use crate::ms_graphs::{two_component_4nodes_each_graph, two_component_graph};

    #[test]
    // a simple walk with teleports between disconnected components (0,1) and (2,3)
    // 0,2 are drugs, which allow teleports
    fn test_teleport_walk() {
        let graph = two_component_graph();

        // drug teleports
        let arr = array![
            [0.0, 1.0],
            [1.0, 0.0]];
        let tm = AnnMatrix::from_matrix(
            arr,
            vec![0, 2], 
            vec![0, 2]
        ).unwrap();

        let mut teleport_matrix = TeleportMatrix::new();
        teleport_matrix.add(0, tm);

        // lets do teh RW
        let tp = TeleportParameters {
            teleport_probability: 1.0,
            teleport_matrix
        };

        let walk_params = WalksParameters::new(20)
            .unwrap()
            .set_teleport_parameters(tp)
            .unwrap();

        let walks: Vec<Walk> = graph.par_iter_random_walks(5, &walk_params).unwrap().collect();
        for w in walks {
            println!("{w:?}");
        }
    }

    #[test]
    // more complicated, allowing for edgetpye transitions
    // which are tricky after teleports (since the teleport is not an edge in the graph and has not edgetype)
    fn test_teleport_walk_with_edgetype_transition() {
        let graph = two_component_graph();

        // drug teleports
        let arr = array![
            [0.0, 1.0],
            [1.0, 0.0]];
        let tm = AnnMatrix::from_matrix(
            arr,
            vec![0, 2], 
            vec![0, 2]
        ).unwrap();

        let mut teleport_matrix = TeleportMatrix::new();
        teleport_matrix.add(0, tm);

        // lets do teh RW
        let tp = TeleportParameters {
            teleport_probability: 0.5,
            teleport_matrix
        };

        let all_edge_types = graph.get_unique_edge_type_ids().unwrap();

        let arr = array![[0.0, 1.0], [1.0, 0.0]];
            // Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let etm = EdgetypeTransitionMatrix::from_matrix(arr, all_edge_types).unwrap();
        let walk_params = WalksParameters::new(30)
            .unwrap()
            .set_teleport_parameters(tp)
            .unwrap()
            .set_edgetype_transition_matrix(etm).unwrap();

        let walks: Vec<_> = graph.par_iter_random_walks(10, &walk_params).unwrap().collect();
        for w in walks {
            println!("{w:?}");
        }
    }


    #[test]
    fn test_aaa(){
        let graph = two_component_4nodes_each_graph();
        // drug teleports
        let arr = array![
            [0.0, 1.0],
            [1.0, 0.0]];
        let tm = AnnMatrix::from_matrix(
            arr,
            vec![0, 4], 
            vec![0, 4]
        ).unwrap();

        let mut teleport_matrix = TeleportMatrix::new();
        teleport_matrix.add(0, tm);        
        // lets do teh RW
        let tp = TeleportParameters {
            teleport_probability: 0.5,
            teleport_matrix
        };

        let all_edge_types = graph.get_unique_edge_type_ids().unwrap();
        let arr = array![
            [0.0, 1.0], 
            [1.0, 0.0]];

        let etm = EdgetypeTransitionMatrix::from_matrix(arr, all_edge_types).unwrap();
        let walk_params = WalksParameters::new(50)
            .unwrap()
            .set_teleport_parameters(tp)
            .unwrap()
            .set_edgetype_transition_matrix(etm).unwrap();

        // println!("{walk_params:?}");

        let walks: Vec<_> = graph.par_iter_random_walks(10, &walk_params).unwrap().collect();
        for w in walks {
            let walk_nodenames = w.iter().map(|n| graph.get_node_name_from_node_id(*n).unwrap()).collect_vec();

            println!("{:?}", walk_nodenames.join(" "));
        }
    }

    #[test]
    fn test_sample() {
        let mut random_seed = 42;
        let mut v = Vec::new();
        for i in 0..100 {
            v.push(
                sample_f32(&mut [1.0, 1.0], random_seed)
            );
            random_seed = splitmix64(random_seed);
        }
        println!("{v:?}")
    }

    #[test]
    fn test_sample2() {
        println!("{}", sample_f32(&mut [1.0, 1.0], 15201296553083726746));
        println!("{}", sample_f32(&mut [1.0, 1.0], 543200672246369984));
        println!("{}", sample_f32(&mut [1.0, 1.0], 3636585004362955532));
        println!("{}", sample_f32(&mut [1.0, 1.0], 339885341293345912));
    }

    #[test]
    fn tttt() {
        let graph = two_component_4nodes_each_graph();

        // drug teleports
        let arr = array![
            [0.0, 1.0],
            [1.0, 0.0]];
        let tm = AnnMatrix::from_matrix(
            arr,
            vec![0, 4], 
            vec![0, 4]
        ).unwrap();

        let mut teleport_matrix = TeleportMatrix::new();
        teleport_matrix.add(0, tm);        
        // lets do teh RW
        let tp = TeleportParameters {
            teleport_probability: 0.5,
            teleport_matrix
        };

        let all_edge_types = graph.get_unique_edge_type_ids().unwrap();
        let arr = array![
            [0.0, 1.0], 
            [1.0, 0.0]];

        let etm = EdgetypeTransitionMatrix::from_matrix(arr, all_edge_types).unwrap();
        let walk_params = WalksParameters::new(1000)
            .unwrap()
            .set_teleport_parameters(tp)
            .unwrap()
            .set_edgetype_transition_matrix(etm).unwrap();


        let src = 4;
        let dst = 0;
        let edge = u64::MAX;
        let mut random_state = 42;
        let min_edge_id = 0;
        let max_edge_id = 2;
        let walks_weights = &walk_params.single_walk_parameters.weights;
        let destinations = &[1,3];
        let previous_destinations = &[5,7];
        unsafe {

            for i in 0..10 {
                random_state = splitmix64(random_state);
                let r = graph.extract_edge(
                    src, dst, edge, random_state, walks_weights, min_edge_id, max_edge_id, destinations, previous_destinations, &None, false
                );
                println!("new node: {}, edge: {}", r.0, r.1);
            }
        };
    }
}