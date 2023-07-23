extern crate protoc_rust;

use std::env;
use std::error::Error;
use std::path::Path;
use std::result::Result;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let tensorflow_folder = &args[1];
    let output_folder = Path::new(&args[2]);
    protoc_rust::Codegen::new()
        .out_dir(
            output_folder
                .join("src/protos")
                .to_str()
                .ok_or("Unable to format output path for main crate")?,
        )
        .inputs(
            [
                "core/framework/allocation_description.proto",
                "core/framework/attr_value.proto",
                "core/framework/cost_graph.proto",
                "core/framework/full_type.proto",
                "core/framework/function.proto",
                "core/framework/graph.proto",
                "core/framework/graph_debug_info.proto",
                "core/framework/node_def.proto",
                "core/framework/op_def.proto",
                "core/framework/resource_handle.proto",
                "core/framework/step_stats.proto",
                "core/framework/tensor.proto",
                "core/framework/tensor_description.proto",
                "core/framework/tensor_shape.proto",
                "core/framework/types.proto",
                "core/framework/variable.proto",
                "core/framework/versions.proto",
                "core/protobuf/cluster.proto",
                "core/protobuf/config.proto",
                "core/protobuf/debug.proto",
                "core/protobuf/meta_graph.proto",
                "core/protobuf/rewriter_config.proto",
                "core/protobuf/saved_model.proto",
                "core/protobuf/saved_object_graph.proto",
                "core/protobuf/saver.proto",
                "core/protobuf/struct.proto",
                "core/protobuf/trackable_object_graph.proto",
                "core/protobuf/verifier_config.proto",
                "tsl/protobuf/coordination_config.proto",
                "tsl/protobuf/rpc_options.proto",
            ]
            .iter()
            .map(|p| format!("{}/tensorflow/{}", tensorflow_folder, p))
            .collect::<Vec<_>>(),
        )
        .include(tensorflow_folder)
        .run()?;
    protoc_rust::Codegen::new()
        .out_dir(
            output_folder
                .join("tensorflow-op-codegen/src/protos")
                .to_str()
                .ok_or("Unable to format output path for ops crate")?,
        )
        .inputs([
            &format!(
                "{}/tensorflow/core/framework/attr_value.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/full_type.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/op_def.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/resource_handle.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/tensor.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/tensor_shape.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/types.proto",
                tensorflow_folder
            ),
        ])
        .include(tensorflow_folder)
        .run()?;
    Ok(())
}
