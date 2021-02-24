use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

fn main() {
    // In this file test_in_input is being used while in the python script,
    // that generates the saved model from Keras model it has a name "test_in".
    // For multiple inputs _input is not being appended to signature input parameter name.
    let signature_input_1_parameter_name = "test_in1";
    let signature_input_2_parameter_name = "test_in2";
    let signature_output_parameter_name = "test_out";

    let save_dir = "examples/keras_multiple_inputs_saved_model";

    let tensor1: Tensor<f32> = Tensor::from(&[0.1, 0.2, 0.3, 0.4, 0.5][..]);
    let tensor2: Tensor<f32> = Tensor::from(&[0.6, 0.7, 0.8, 0.9, 0.1][..]);
    let mut graph = Graph::new();

    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
        .expect("Can't load saved model");

    let session = &bundle.session;

    let signature = bundle
        .meta_graph_def()
        .get_signature("serving_default")
        .unwrap();
    let input_info1 = signature
        .get_input(signature_input_1_parameter_name)
        .unwrap();
    let input_info2 = signature
        .get_input(signature_input_2_parameter_name)
        .unwrap();
    let output_info = signature
        .get_output(signature_output_parameter_name)
        .unwrap();

    let input_op1 = graph
        .operation_by_name_required(&input_info1.name().name)
        .unwrap();
    let input_op2 = graph
        .operation_by_name_required(&input_info2.name().name)
        .unwrap();
    let output_op = graph
        .operation_by_name_required(&output_info.name().name)
        .unwrap();
    let mut args = SessionRunArgs::new();
    args.add_feed(&input_op1, 0, &tensor1);
    args.add_feed(&input_op2, 0, &tensor2);
    let out = args.request_fetch(&output_op, 0);
    session
        .run(&mut args)
        .expect("Error occured during calculations: {:?}");
    let out_res: f32 = args.fetch(out).unwrap()[0];
    println!("Results: {:?}", out_res);
}
