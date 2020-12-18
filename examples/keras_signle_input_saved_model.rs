use tensorflow::{SessionRunArgs, Graph, SessionOptions, Tensor, SavedModelBundle};

fn main() {
    // In this file test_in_input is being used while in the python script,
    // that generates the saved model from Keras model it has a name "test_in".
    // For multiple inputs _input is not being appended to the op name.
    let input_op_name = "test_in_input";
    let output_op_name = "test_out";
    let save_dir = "examples/keras_signle_input_saved_model";
    let v: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let tensor = vector_to_tensor(&v);
    let mut graph = Graph::new();

    let bundle = SavedModelBundle::load(
        &SessionOptions::new(),
        &["serve"],
        &mut graph,
        save_dir,
    ).expect("Can't load saved model");

    let session = &bundle.session;

    let signature = bundle.meta_graph_def().get_signature("serving_default").unwrap();
    let input_info = signature.get_input(input_op_name).unwrap();
    let output_info = signature.get_output(output_op_name).unwrap();
    let input_op = graph.operation_by_name_required(&input_info.name().name).unwrap();
    let output_op = graph.operation_by_name_required(&output_info.name().name).unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&input_op, 0, &tensor);
    let out = args.request_fetch(&output_op, 0);
    //
    let result = session.run(&mut args);
    if result.is_err() {
        panic!("Error occured during calculations: {:?}", result);
    }
    let out_res:f32 = args.fetch(out).unwrap()[0];
    println!("Results: {:?}", out_res);
}

pub fn vector_to_tensor(v: &Vec<f32>) -> Tensor<f32> {
    let dimension = v.len();
    let tensor = Tensor::new(&[1, dimension as u64]).with_values(&v[..]).unwrap();
    return tensor;
}
