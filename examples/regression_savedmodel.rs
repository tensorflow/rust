use random;
use random::Source;
use std::error::Error;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "examples/regression_savedmodel"; // y = w * x + b
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python regression_savedmodel.py' to generate \
                     {} and try again.",
                    export_dir
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
    let mut rand = random::default();
    let mut x = Tensor::new(&[num_points as u64]);
    let mut y = Tensor::new(&[num_points as u64]);
    for i in 0..num_points {
        x[i] = (2.0 * rand.read::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
    let session = &bundle.session;

    // train
    let train_signature = bundle.meta_graph_def().get_signature("train")?;
    let x_info = train_signature.get_input("x")?;
    let y_info = train_signature.get_input("y")?;
    let loss_info = train_signature.get_output("loss")?;
    let op_x = graph.operation_by_name_required(&x_info.name().name)?;
    let op_y = graph.operation_by_name_required(&y_info.name().name)?;
    let op_train = graph.operation_by_name_required(&loss_info.name().name)?;

    // internal parameters
    let op_b = {
        let b_signature = bundle.meta_graph_def().get_signature("b")?;
        let b_info = b_signature.get_output("output")?;
        graph.operation_by_name_required(&b_info.name().name)?
    };

    let op_w = {
        let w_signature = bundle.meta_graph_def().get_signature("w")?;
        let w_info = w_signature.get_output("output")?;
        graph.operation_by_name_required(&w_info.name().name)?
    };

    // Train the model (e.g. for fine tuning).
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
