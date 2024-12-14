use arrow_array::{ArrayRef, Float64Array, StringArray};
use arrow_array::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use parquet::schema::types::Type;
use std::convert::TryFrom;
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use parquet::basic::Compression;
use crate::build_graph_from_strings;
use crate::Graph;
use crate::ItersWrapper;
use rayon::iter::Empty as ParEmpty;
use std::iter::Empty as SeqEmpty;

impl Graph {

    /// create a graph from parquet
    /// 
    /// TODO:
    /// - make nodetype/edgetype/weight optional: Currently required
    ///   but a pain to implement (getting the types of the if clauses right reuired some dyn)
    pub fn from_parquet(
        nodes_pq: &str, 
        edges_pq: &str,
        nodename_col: &str,
        nodetype_col: Option<String>,
        edge_src_col: &str,
        edge_dst_col: &str,
        edge_type_col: Option<String>,
        edge_weight_col: Option<String>,
        directed: bool,
        name: String
    ) -> Graph {

        let has_node_types = nodetype_col.is_some();
        let has_edge_types= edge_type_col.is_some();
        let has_edge_weights= false;

        assert!(edge_weight_col.is_none(), "edge weights not supported");

        let node_paths = get_parquet_parts(nodes_pq);
        let edge_paths = get_parquet_parts(edges_pq);

        // =========================================
        // getting the nodes
        // =========================================
        // just to get the schema, peek into the first file
        let reader = SerializedFileReader::new(File::open(node_paths[0].clone()).unwrap()).unwrap();
        let parquet_metadata = reader.metadata();
        let node_schema = parquet_metadata.file_metadata().schema();
        

        let requested_fields = vec![nodename_col.to_string(), nodetype_col.expect("nodetype required").to_string()];
        let proj_schema = get_projection_schema(node_schema, requested_fields);

        // construct the node iterator: its pretty simply, but
        // the way its fed into ensamallen is convoluted:
        // each node is a Ok(counter,(nodestr, Option(vec[nodetype])))
        // let node_paths = vec![nodes_pq.to_string()];
        let it = multifile_iter(&node_paths, proj_schema);

        let q = it.map(|cols|{
            let nodename = cols[0].clone();
            let nodetype = cols[1].clone();
            (nodename, Some(vec![nodetype]))  // ensmallens weird format, nodetype is an option of multiple nodetypes
        })
            // .zip(std::iter::repeat(None)) //empty nodetype
            .enumerate()
            .map(Ok);

        let nodes_iterator = Some(
            ItersWrapper::<_, _, ParEmpty<_>>::Sequential(q)
        );


        // =========================================
        // getting the edges
        // =========================================

        let reader = SerializedFileReader::new(File::open(edge_paths[0].clone()).unwrap()).unwrap();
        let parquet_metadata = reader.metadata();
        let edge_schema = parquet_metadata.file_metadata().schema();

        let requested_fields = vec![
            edge_src_col.to_string(), 
            edge_dst_col.to_string(), 
            edge_type_col.expect("edgetyp required").to_string()];

        let proj_schema = get_projection_schema(edge_schema, requested_fields);
        // let edge_paths = vec![edges_pq.to_string()];
        let it_ed = multifile_iter(&edge_paths, proj_schema);
        let qw = it_ed
             .map(|x| {
                let src_name = x[0].clone();
                let dst_name = x[1].clone();
                let edgetype = x[2].clone();
                let the_weight = 1.0f32;
                // let the_weight = weight.parse::<WeightT>().unwrap();
                (src_name, dst_name, Some(edgetype), the_weight)
            })
             .enumerate()
             .map(Ok);
        let edges_iterator = Some(
            ItersWrapper::<_, _, ParEmpty<_>>::Sequential(qw)
        );

        build_graph_from_strings(
            None::<ItersWrapper<_, SeqEmpty<_>, ParEmpty<_>>>, // node_types_iterator
            None,                                                     // number_of_node_types
            Some(false),                                              // numeric_node_type_ids
            None,                                                     // minimum_node_type_id
            has_node_types,                                           // has_node_types
            Some(false),                                              // node_types_list_is_correct
            nodes_iterator,                                           // nodes_iterator
            None,                                                     // number_of_nodes
            false,                                                    // node_list_is_correct
            false,                                                    // numeric_node_ids
            false, // numeric_node_list_node_type_ids
            None,  // minimum_node_id
            None::<ItersWrapper<_, SeqEmpty<_>, ParEmpty<_>>>, // edge_types_iterator
            None,  // number_of_edge_types
            Some(false), // numeric_edge_type_ids
            None,  // minimum_edge_type_id
            has_edge_types, // has_edge_types
            Some(false), // edge_types_list_is_correct
            edges_iterator,
            has_edge_weights, // has_edge_weights
            directed,         // directed
            Some(false),      // correct
            Some(false),      // complete
            Some(true),       // duplicates
            Some(false),      // sorted
            None,             // number_of_edges
            Some(false),      // numeric_edge_list_node_ids
            Some(false),      // numeric_edge_list_edge_type_ids
            Some(true),       // skip_node_types_if_unavailable
            Some(true),       // skip_edge_types_if_unavailable
            true,             // may_have_singletons
            true,             // may_have_singleton_with_selfloops
            name,             // name
        ).unwrap()
    }
}

/// parquet can either be a single file or a directory with many .parquet files
/// 
/// returns: A vector of all parquet files associated with that dataset
fn get_parquet_parts(pathname: &str) -> Vec<String> {

    let p = Path::new(pathname);

    if p.is_file(){ //if we're pointing to a file, just return that single file as a vec
        vec![pathname.to_string()]
    } else {  // otherwise iter through the directory, picking up any .parquet files

        let paths = p.read_dir().expect("read_dir failed");
        let x: Vec<_> = paths.into_iter().filter_map(|path|
            { 
                let p = path.unwrap().path();
                
                if let Some(extension) = p.extension()  {  // since some dont even have an extension
                    if extension =="parquet" {
                        Some(p.display().to_string())
                    } else {
                        None   // skip via filtermap
                    }                
                } else {
                    None   // skip via filtermap
                }
            }).collect();
        x
    }
}



#[test]
fn test_pq(){

    write_nodes("/tmp/nodes.parquet");
    write_edges("/tmp/edges.parquet");

    let the_graph = Graph::from_parquet(
        "/tmp/nodes.parquet", 
        "/tmp/edges.parquet",
        "id",
        Some("category".to_string()),
        "src",
        "dst",
        Some("edgetype".to_string()),
        None, //Some("weight".to_string()),
        false,
        "somename".to_string(),
    
    );
    println!("{}, {}", the_graph.get_number_of_nodes(), the_graph.get_number_of_edges());

    for e in the_graph.get_edge_node_names(false) {
        println!("{e:?}");
    }
    println!("CCs {:?}", the_graph.get_number_of_connected_components(Some(true)));
    println!("CCs {:?}", the_graph.get_edge_type_names_counts_hashmap());
    println!("CCs {:?}", the_graph.get_node_type_names_counts_hashmap());


}

/// iterating over a parquet dataset spread over multiple files. EFor simplicity each row is represented as Vec<String> 
/// (i.e. it works only for all string schemas)
/// 
fn multifile_iter(
    paths: &[String], 
    schema:parquet::schema::types::Type
) -> impl Iterator<Item = Vec<String>> + use<'_> {
    // Return all parquet files in a directory,  and all 
    // subdirectories:

    let rows = paths.iter()
        .map(|p| 	
            SerializedFileReader::try_from(p.as_str()).unwrap())
        .flat_map(move |r| 
            RowIter::from_file_into(
                Box::new(r))
                .project(Some(schema.clone()))
            .unwrap());

    rows.into_iter()
        .flatten()
        .map(|row| row.into_columns())
        .map(|v| {
            // turn the row into a simple vec (of strings)
            let string_vec: Vec<String> = v.into_iter().map(|(_colname, val)|{
                match val {
                    parquet::record::Field::Str(value) => value,
                    _ => panic!("only works with string-type columns")
                }
            }).collect();
            string_vec
        })
}


/// schema projection onto the given columns. Useful to iterate over subsets of the pq.file
/// 
fn get_projection_schema(schema: &Type, requested_fields: Vec<String>) -> Type {
    let fields = schema.get_fields();
    let selected_fields = fields.to_vec();

    // this screws up the order of columns
    // if requested_fields.len()>0{
        // selected_fields.retain(|f|  
            // requested_fields.contains(&String::from(f.name())));
    // }

    // this on the other hand is a bit ugly, but selected the columns in the correct order
    let mut proj_fields = Vec::new();
    for field in requested_fields {
        for f in selected_fields.iter() {
            // String::from(f.name())
            if f.name() == field {
                proj_fields.push(f.clone());
            }
        }
    }

    let schema_projection = Type::group_type_builder(schema.name())
        .with_fields(proj_fields)
        .build()
        .unwrap();
    schema_projection
}



fn write_nodes(fname: &str) {
    let ids = StringArray::from(vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]);
    let nodetypes = StringArray::from(vec!["AB".to_string(), "AB".to_string(), "CD".to_string(), "CD".to_string()]);
    // let vals = Int32Array::from(vec![5, 6, 7, 8]);
    let batch = RecordBatch::try_from_iter(vec![
      ("id", Arc::new(ids) as ArrayRef),
      ("category", Arc::new(nodetypes) as ArrayRef),
    ]).unwrap();
   
    let file = File::create(fname).unwrap();
   
    // WriterProperties can be used to set Parquet file options
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
   
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
   
    writer.write(&batch).expect("Writing batch");
   
    // writer must be closed to write footer
    writer.close().unwrap();
}

fn write_edges(fname: &str) {
    let src = StringArray::from(vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()]);
    let dst = StringArray::from(vec!["b".to_string(), "a".to_string(), "d".to_string(), "c".to_string()]);
    let edgetype = StringArray::from(vec!["ET".to_string(), "ET".to_string(), "ET2".to_string(), "ET2".to_string()]);
    let weight = Float64Array::from(vec![1.0, 1.0, 0.5, 0.1]);
    // let vals = Int32Array::from(vec![5, 6, 7, 8]);
    let batch = RecordBatch::try_from_iter(vec![
      ("src", Arc::new(src) as ArrayRef),
      ("dst", Arc::new(dst.clone()) as ArrayRef),
      ("edgetype", Arc::new(edgetype) as ArrayRef),
      ("weight", Arc::new(weight) as ArrayRef),
    ]).unwrap();
   
    let file = File::create(fname).unwrap();
   
    // WriterProperties can be used to set Parquet file options
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
   
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
   
    writer.write(&batch).expect("Writing batch");
   
    // writer must be closed to write footer
    writer.close().unwrap();
}