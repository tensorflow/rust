use rand;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() -> Result<(), Box<dyn Error>> {
    let filename = "examples/regression/model.pb"; // y = w * x + b
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python regression.py' to generate \
                     {} and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }

    // Generate some test data.
    let w = 0.1;
    let b = 0.3;
    let num_points = 100;
    let steps = 201;
    let mut x = Tensor::new(&[num_points as u64]);
    let mut y = Tensor::new(&[num_points as u64]);
    for i in 0..num_points {
        x[i] = (2.0 * rand::random::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }

    // Load the computation graph defined by regression.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;
    let op_x = graph.operation_by_name_required("x")?;
    let op_y = graph.operation_by_name_required("y")?;
    let op_init = graph.operation_by_name_required("init")?;
    let op_train = graph.operation_by_name_required("train")?;
    let op_w = graph.operation_by_name_required("w")?;
    let op_b = graph.operation_by_name_required("b")?;

    // Load the test data into the session.
    let mut init_step = SessionRunArgs::new();
    init_step.add_target(&op_init);
    session.run(&mut init_step)?;

    // Train the model.
    let mut train_step = SessionRunArgs::new();
    train_step.add_feed(&op_x, 0, &x);
    train_step.add_feed(&op_y, 0, &y);
    train_step.add_target(&op_train);
    for _ in 0..steps {
        session.run(&mut train_step)?;
    }

    // Grab the data out of the session.
    let mut output_step = SessionRunArgs::new();
    let w_ix = output_step.request_fetch(&op_w, 0);
    let b_ix = output_step.request_fetch(&op_b, 0);
    session.run(&mut output_step)?;

    // Check our results.
    let w_hat: f32 = output_step.fetch(w_ix)?[0];
    let b_hat: f32 = output_step.fetch(b_ix)?[0];
    println!(
        "Checking w: expected {}, got {}. {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    println!(
        "Checking b: expected {}, got {}. {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    Ok(())
}
