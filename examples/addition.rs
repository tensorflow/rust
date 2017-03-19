#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate tensorflow;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process::exit;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

fn main() {
    // Putting the main code in another function serves two purposes:
    // 1. We can use the `?` operator.
    // 2. We can call exit safely, which does not run any destructors.
    exit(match run() {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

fn run() -> Result<(), Box<Error>> {
    let filename = "examples/addition-model/model.pb"; // z = x + y
    if !Path::new(filename).exists() {
        return Err(Box::new(Status::new_set(Code::NotFound,
                                            &format!("Run 'python addition.py' to generate {} \
                                                      and try again.",
                                                     filename))
            .unwrap()));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the Step
    let mut step = StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("x")?, 0, &x);
    step.add_input(&graph.operation_by_name_required("y")?, 0, &y);
    let z = step.request_output(&graph.operation_by_name_required("z")?, 0);
    session.run(&mut step)?;

    // Check our results.
    let z_res: i32 = step.take_output(z)?[0];
    println!("{:?}", z_res);

    Ok(())
}
