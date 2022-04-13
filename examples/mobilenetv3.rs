use std::error::Error;
use std::path::PathBuf;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;

use tensorflow::eager::{self, raw_ops, ToTensorHandle};

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "examples/mobilenetv3";
    let model_file: PathBuf = [export_dir, "saved_model.pb"].iter().collect();
    if !model_file.exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python examples/mobilenetv3/create_model.py' to generate \
                     {} and try again.",
                    model_file.display()
                ),
            )
            .unwrap(),
        ));
    }

    // Create an eager execution context
    let opts = eager::ContextOptions::new();
    let ctx = eager::Context::new(opts)?;

    // Load an input image.
    let fname = "examples/mobilenetv3/sample.png".to_handle(&ctx)?;
    let buf = raw_ops::read_file(&ctx, &fname)?;
    let img = raw_ops::decode_image(&ctx, &buf)?;
    let cast2float = raw_ops::Cast::new().DstT(tensorflow::DataType::Float);
    let img = cast2float.call(&ctx, &img)?;
    let batch = raw_ops::expand_dims(&ctx, &img, &0)?; // add batch dim
    let readonly_x = batch.resolve()?;

    // The current eager API implementation requires unsafe block to feed the tensor into a graph.
    let x: Tensor<f32> = unsafe { readonly_x.into_tensor() };

    // Load the model.
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
    let session = &bundle.session;

    // get in/out operations
    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let x_info = signature.get_input("input_1")?;
    let op_x = &graph.operation_by_name_required(&x_info.name().name)?;
    let output_info = signature.get_output("Predictions")?;
    let op_output = &graph.operation_by_name_required(&output_info.name().name)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(op_x, 0, &x);
    let token_output = args.request_fetch(op_output, 0);
    session.run(&mut args)?;

    // Check the output.
    let output: Tensor<f32> = args.fetch(token_output)?;

    // Calculate argmax of the output
    let (max_idx, _max_val) =
        output
            .iter()
            .enumerate()
            .fold((0, output[0]), |(idx_max, val_max), (idx, val)| {
                if &val_max > val {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            });

    // This index is expected to be identical with that of the Python code,
    // but this is not guaranteed due to floating operations.
    println!("argmax={}", max_idx);

    Ok(())
}
