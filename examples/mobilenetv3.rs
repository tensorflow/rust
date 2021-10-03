use std::error::Error;
use std::path::PathBuf;
use std::result::Result;
use tensorflow::eager::raw_ops;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;

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

    // Create input variables for our addition
    // 1. load the image file
    let filename: Tensor<String> = Tensor::from(String::from("examples/mobilenetv3/sample.png"));
    let buf = raw_ops::read_file(filename).unwrap();
    let decode_png = raw_ops::DecodePng::new().channels(3);
    let img = decode_png.call(buf).unwrap();

    // 2. shrink the image with antialias, which requires ScaleAndTranslate op instead of Resize op.
    let img_height = img.dim(0).unwrap();
    let img_width = img.dim(1).unwrap();
    let size = Tensor::from(&[224, 224]); // desired size
    let scale = Tensor::from(&[
        size[0] as f32 / img_height as f32,
        size[1] as f32 / img_width as f32,
    ]);
    let translation = Tensor::from(&[0.0f32, 0.0f32]); // no translation
    let scale_and_translate = raw_ops::ScaleAndTranslate::new().kernel_type("triangle");
    let dim = Tensor::from(0); // ScaleAndTranslate requires 4D Tensor (batch, height, width, channel)
    let images = raw_ops::expand_dims(img, dim).unwrap();
    let h = scale_and_translate
        .call(images, size, scale, translation)
        .unwrap();

    // 3. get 224x224 image as usual Tensor
    let x: Tensor<f32> = h.resolve().unwrap();

    // Load the saved model exported by zenn_savedmodel.py.
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
