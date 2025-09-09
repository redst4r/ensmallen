use super::no_binding;
use super::types::Result;
// use crate::{NodeT, NodeTypeT};
// use arrow_array::ArrowNativeTypeOp;
// use named_matrix::matrix::AnnMatrix;
use named_matrix::sparse::AnnMatrixSparse;
// use ndarray::array;
use std::collections::HashMap;
// use std::fmt::{self};
use std::fs::File;
use std::io::{self, BufRead, Write};
// use std::iter;
use std::path::Path;
// use std::str::FromStr;
// use std::string::ParseError;
// use vec_rand::sample_f32;

/// For each node type, what are the possible teleports
// TODO: grrr, dont really want to make that thing clonable, might be big
// but we have to in order to use it inside WalkParams
// just avoid cloning!
#[derive(Debug, Clone, PartialEq)]
#[no_binding]
pub struct TeleportMatrix {
    teleports: HashMap<String, AnnMatrixSparse<String, String>>,
}

// #[derive(Debug, Clone, PartialEq, Hash, Ord, Eq, PartialOrd)]
// #[no_binding]
// pub(crate) struct NodeName(pub(crate) String);
// impl fmt::Display for NodeName {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         // Customize so only `x` and `y` are denoted.
//         write!(f, "{}", self.0)
//     }
// }
// impl std::str::FromStr for NodeName {
//     type Err = ParseError;
//
//     fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
//         Ok(NodeName(s.to_string()))
//     }
// }
// #[derive(Clone, Hash, Eq, PartialEq, Debug)]
// pub(crate) struct NodetypeName(pub(crate) String);
// impl fmt::Display for NodetypeName {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         // Customize so only `x` and `y` are denoted.
//         write!(f, "{}", self.0)
//     }
// }
// impl std::str::FromStr for NodetypeName {
//     type Err = ParseError;
//
//     fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
//         Ok(NodetypeName(s.to_string()))
//     }
// }
impl TeleportMatrix {
    /// constructs an empty teleport matrix
    pub fn new() -> Self {
        let teleports = HashMap::new();
        Self { teleports }
    }

    pub fn add(&mut self, nodetype: String, matrix: AnnMatrixSparse<String, String>) {
        self.teleports.insert(nodetype, matrix);
    }

    /// for the given node/type, sample a new node to teleport to
    /// returns
    /// - Ok(somenode) if successful
    /// - Err("unknown nodetype") if the nodetype doesnt have a teleprot matrix
    /// - Err("no target") if the node cant teleport anywhere
    // pub fn sample_teleport_dense(
    //     &self,
    //     nodeid: NodeT,
    //     nodetype: NodeTypeT,
    //     random_state: u64,
    // ) -> Result<NodeT> {
    //     if let Some(matrix) = self.teleports.get(&nodetype) {
    //         if let Some(row) = matrix.get_row(&nodeid) {
    //             // warning: sample_f32 mutates the row, make sure this doesnt propagate back into `matrix`!! i.e. keep the `to_vec`
    //             let mut rrr = row.to_vec();
    //
    //             if rrr.iter().all(|&x| x.is_zero()) {
    //                 return Err("all targets are zero".to_string());
    //             }
    //             let ix = sample_f32(&mut rrr, random_state);
    //             let sampled_nodeid = matrix.rownames[ix];
    //             Ok(sampled_nodeid)
    //         } else {
    //             // nodeid not in the teleport matrix
    //             Err("rowname unknown".to_string())
    //         }
    //     } else {
    //         Err("unknown nodetype".to_string())
    //     }
    // }
    pub fn sample_teleport(
        &self,
        nodeid: String,
        nodetype: String,
        random_state: u64,
    ) -> Result<String> {
        if let Some(matrix) = self.teleports.get(&nodetype) {
            // this will try to sample, but can fail if
            // - nodeid not in the rows
            // - the entire row is zeros
            matrix.sample_from_row(&nodeid, random_state)
        } else {
            Err("unknown nodetype".to_string())
        }
    }
    /// constructs the teleport matrices from a single flat dataframe
    /// with the following shape:
    /// nodeid1, nodeid2,value, nodetype
    pub fn from_file(fname: &str) -> Result<Self> {
        let mut big_hashmap: HashMap<String, HashMap<(String, String), f32>> = HashMap::new(); // from nodetype -> Matrix

        if let Ok(lines) = read_lines(fname) {
            // Consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(l) = line {
                    let items: Vec<_> = l.split(',').collect();

                    let n1 = items[0].parse::<String>().unwrap();
                    let n2 = items[1].parse::<String>().unwrap();
                    let val = items[2].parse::<f32>().unwrap();
                    let nodetype = items[3].parse::<String>().unwrap();
                    let hmap = big_hashmap.entry(nodetype).or_insert(HashMap::new());
                    hmap.insert((n1, n2), val);
                }
            }

            println!("Contrsucting teleport matrix");
            let mut teleport_matrix = Self::new();
            for (ntype, hmap) in big_hashmap {
                let adata = AnnMatrixSparse::from_hashmap(hmap);
                teleport_matrix.add(ntype, adata);
            }
            println!("Done Contrsucting teleport matrix");
            Ok(teleport_matrix)
        } else {
            Err("error reading Teleport matrix, prob the file doesnt exist".to_string())
        }
    }

    pub fn to_file(&self, fname: &str) {
        let file = File::create(fname).unwrap();
        let mut writer = io::BufWriter::new(file);

        for (ntype, adata) in &self.teleports {
            // just iterate over all row,col,val in the matrix
            adata.iter_elements().for_each(|(r, c, v)| {
                writeln!(&mut writer, "{r},{c},{v},{ntype}").unwrap();
            });
        }
    }
}

#[test]
fn test_sample_teleport() {
    use std::iter::FromIterator;
    let node0 = "0".to_string();
    let node10 = "10".to_string();
    let hmap: HashMap<(String, String), f32> = HashMap::from_iter(vec![
        ((node0.clone(), node0.clone()), 1.0),
        ((node0.clone(), node10.clone()), 0.0),
        ((node10.clone(), node0.clone()), 0.0),
        ((node10.clone(), node10.clone()), 0.0),
    ]);
    let q = AnnMatrixSparse::from_hashmap(hmap);

    let mut teleport = TeleportMatrix::new();
    let nt = "0".to_string();
    teleport.add(nt.clone(), q);
    assert_eq!(
        teleport.sample_teleport(node0.clone(), nt.clone(), 42),
        Ok(node0.clone())
    );
    assert_eq!(
        teleport.sample_teleport("0".to_string(), "1234".to_string(), 42),
        Err("unknown nodetype".to_string())
    );
    assert_eq!(
        teleport.sample_teleport("1234".to_string(), "0".to_string(), 42),
        Err("rowname unknown".to_string())
    );
    assert_eq!(
        teleport.sample_teleport("10".to_string(), "0".to_string(), 42),
        Err("all targets are zero".to_string())
    );
}

#[test]
fn test_to_file_from_file() {
    use std::iter::FromIterator;
    // let hmap: HashMap<(NodeT, NodeT), f32> = HashMap::from_iter(vec![
    //     ((0, 0), 1.0),
    //     ((10, 10), 0.3),
    //     ((0, 10), 0.0),
    //     ((10, 0), 0.5),
    // ]);
    let node0 = "0".to_string();
    let node10 = "10".to_string();
    let hmap: HashMap<(String, String), f32> = HashMap::from_iter(vec![
        ((node0.clone(), node0.clone()), 1.0),
        ((node0.clone(), node10.clone()), 0.0),
        ((node10.clone(), node0.clone()), 0.0),
        ((node10.clone(), node10.clone()), 0.0),
    ]);
    let q = AnnMatrixSparse::from_hashmap(hmap);

    let mut teleport = TeleportMatrix::new();
    let nt = "0".to_string();
    teleport.add(nt.clone(), q);
    teleport.to_file("/tmp/tel.csv");

    let teleport2 = TeleportMatrix::from_file("/tmp/tel.csv").unwrap();
    println!("{teleport:?}");
    assert_eq!(teleport2.teleports[&nt].get_value(&node0, &node0), &1.0);
    assert_eq!(teleport2.teleports[&nt].get_value(&node0, &node10), &0.0);
    assert_eq!(teleport2.teleports[&nt].get_value(&node10, &node0), &0.5);
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

#[test]
fn test_from_python_file() {
    println!("loading");
    // let teleport = TeleportMatrix::from_file("/tmp/ATC_strings.csv").unwrap();
    // let nodetypename = "biolink:Disease".to_string();
    // let file = "/tmp/Mondo_strings.csv";
    let nodetypename = "biolink:SmallMolecule".to_string();
    let file =
        "/home/michi/mounts/TB4drive/spoke-ingestion_kedro/data/02_intermediate/atc_semsim.csv";
    let teleport = TeleportMatrix::from_file(file).unwrap();

    println!("Teleport: {:?}", teleport);
    // let nodetypename = "0".to_string();
    let rows = teleport
        .teleports
        .get(&nodetypename)
        .unwrap()
        .rownames
        .clone();
    println!("loaded; now sampling");
    for i in rows.iter().take(10) {
        let r = teleport.sample_teleport(i.to_string(), nodetypename.clone(), 42);
        println!("{i:?} -> {r:?}");

        if i.contains('"') {
            panic!();
        }
    }
    println!("done sampling");

    let x = rows[0].clone();
    let y = "LALA".to_string();
    println!("{}, {}", x, y);
}
