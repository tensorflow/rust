#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate tensorflow;

use std::error::Error;
use std::result::Result;
use std::process::exit;
use tensorflow::DataType;
use tensorflow::Graph;
use tensorflow::Output;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Shape;
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
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    let mut graph = Graph::new();

    let x_in = {
        let mut nd = graph.new_operation("Placeholder", "x").unwrap();
        nd.set_attr_type("dtype", DataType::Int32).unwrap();
        nd.set_attr_shape("shape", &Shape::from(Some(vec![]))).unwrap();
        nd.finish().unwrap()
    };

    let y_in = {
        let mut nd = graph.new_operation("Placeholder", "y").unwrap();
        nd.set_attr_type("dtype", DataType::Int32).unwrap();
        nd.set_attr_shape("shape", &Shape::from(Some(vec![]))).unwrap();
        nd.finish().unwrap()
    };

    let sum = {
        let mut nd = graph.new_operation("Add", "sum").unwrap();
        nd.add_input(Output {
            operation: x_in.clone(),
            index: 0,
        });
        nd.add_input(Output {
            operation: y_in.clone(),
            index: 0,
        });
        nd.finish().unwrap()
    };

    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&x_in, 0, &x);
    args.add_feed(&y_in, 0, &y);
    let z = args.request_fetch(&sum, 0);
    session.run(&mut args)?;

    // Check our results.
    let z_res: i32 = args.fetch(z)?[0];
    println!("{:?}", z_res);

    Ok(())
}
