use super::types::Result;
use crate::{NodeT, NodeTypeT};
use ndarray::{Array2, ArrayView1};
use std::collections::HashMap;
use vec_rand::sample_f32;

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct AnnMatrix {
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

    pub fn from_matrix(
        matrix: Array2<f32>,
        rownames: Vec<NodeT>,
        colnames: Vec<NodeT>,
    ) -> Result<Self> {
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
    fn get_row_index(&self, src_name: NodeT) -> usize {
        self.rownames_to_index[&src_name]
    }

    fn get_col_index(&self, dst_name: NodeT) -> usize {
        self.colnames_to_index[&dst_name]
    }

    /// return the row/column indices corresponding to the edgetypes
    /// TODO: return Result, in case the edgetypes dont exist
    fn get_indices(&self, src_name: NodeT, dst_name: NodeT) -> (usize, usize) {
        (self.get_row_index(src_name), self.get_col_index(dst_name))
    }

    /// get the transition probability from one edgtype to another
    /// TODO: return Result, in case the edgetypes dont exist
    pub fn get_probability(&self, src_name: NodeT, dst_name: NodeT) -> f32 {
        let (i1, i2) = self.get_indices(src_name, dst_name);
        self.matrix[(i1, i2)]
    }

    /// set the transition probability from one edgtype to another
    pub fn set_probability(&mut self, src_name: NodeT, dst_name: NodeT, value: f32) {
        let (i1, i2) = self.get_indices(src_name, dst_name);
        self.matrix[(i1, i2)] = value;
    }

    pub fn get_row(&self, src_name: NodeT) -> ArrayView1<f32> {
        let row_ix = self.get_row_index(src_name);
        let r = self.matrix.row(row_ix);
        r
    }
}

/// For each node type, what are the possible teleports
// TODO: grrr, dont really want to make that thing clonable, might be big
// just avoid cloning!
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TeleportMatrix {
    teleports: HashMap<NodeTypeT, AnnMatrix>,
}

impl TeleportMatrix {
    /// constructs an empty teleport matrix
    pub fn new() -> Self {
        let teleports = HashMap::new();
        Self { teleports }
    }

    pub fn add(&mut self, nodetype: NodeTypeT, matrix: AnnMatrix) {
        self.teleports.insert(nodetype, matrix);
    }

    // for the given node/type, sample a new node to teleport to
    pub fn sample_teleport(
        &self,
        nodeid: NodeT,
        nodetype: NodeTypeT,
        random_state: u64,
    ) -> Result<NodeT> {
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
