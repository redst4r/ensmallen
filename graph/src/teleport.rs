use super::types::Result;
use crate::{NodeT, NodeTypeT};
use std::collections::HashMap;
use vec_rand::sample_f32;
use named_matrix::matrix::AnnMatrix;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

/// For each node type, what are the possible teleports
// TODO: grrr, dont really want to make that thing clonable, might be big
// just avoid cloning!
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TeleportMatrix {
    teleports: HashMap<NodeTypeT, AnnMatrix<NodeT, NodeT, f32>>,
}

impl TeleportMatrix {
    /// constructs an empty teleport matrix
    pub fn new() -> Self {
        let teleports = HashMap::new();
        Self { teleports }
    }

    pub fn add(&mut self, nodetype: NodeTypeT, matrix: AnnMatrix<NodeT, NodeT, f32>) {
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
            let mut row = matrix.get_row(&nodeid).to_vec();
            let ix = sample_f32(&mut row, random_state);
            let sampled_nodeid = matrix.rownames[ix];
            Ok(sampled_nodeid)
        } else {
            Err("unknown nodetype".to_string())
        }
    }
    /// constructs the teleport matrices from a single flat dataframe
    /// with the following shape:
    /// nodeid1, nodeid2,value, nodetype
    pub fn from_file(fname: &str) -> Result<Self> {
        let mut big_hashmap: HashMap<NodeTypeT, HashMap<(NodeT, NodeT), f32>> = HashMap::new(); // from nodetype -> Matrix

        if let Ok(lines) = read_lines(fname) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(l) = line {
                    let items: Vec<_> = l.split(',').collect();

                    let n1 = items[0].parse::<NodeT>().unwrap();
                    let n2 = items[1].parse::<NodeT>().unwrap();
                    let val = items[2].parse::<f32>().unwrap();
                    let nodetype = items[3].parse::<NodeTypeT>().unwrap();
                    let hmap = big_hashmap.entry(nodetype).or_insert(HashMap::new());
                    hmap.insert((n1, n2), val);
                }
            }

            let mut teleport_matrix = Self::new();
            for (ntype, hmap) in big_hashmap {
                let adata = AnnMatrix::from_hashmap(hmap).unwrap();
                teleport_matrix.add(ntype, adata);
            }
            Ok(teleport_matrix)
        } else {
            Err("some issue".to_string())
        }
    }

    pub fn to_file(&self, fname: &str) {
        let file = File::create(fname).unwrap();
        let mut writer = io::BufWriter::new(file);

        for (&ntype, adata) in self.teleports.iter() {
            // just iterate over all row,col,val in the matrix
            adata.iter_elements().for_each(|(r, c, v)| {
                writeln!(&mut writer, "{r},{c},{v},{ntype}").unwrap();
            });
        }
    }
}

#[test]
fn test_to_file_from_file() {
    use std::iter::FromIterator;
    let hmap: HashMap<(NodeT, NodeT), f32> = HashMap::from_iter(vec![
        ((0, 0), 1.0),
        ((10, 10), 0.3),
        ((0, 10), 0.0),
        ((10, 0), 0.5),
    ]);
    let q = AnnMatrix::from_hashmap(hmap).unwrap();

    let mut teleport = TeleportMatrix::new();
    teleport.add(0, q);

    teleport.to_file("/tmp/tel.csv");

    let teleport2 = TeleportMatrix::from_file("/tmp/tel.csv").unwrap();
    println!("{teleport:?}");
    assert_eq!(teleport2.teleports[&0].get_value(&0, &0), &1.0);
    assert_eq!(teleport2.teleports[&0].get_value(&0, &10), &0.0);
    assert_eq!(teleport2.teleports[&0].get_value(&10, &0), &0.5);
}

// The output is wrapped in a Result to allow matching on errors.
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
