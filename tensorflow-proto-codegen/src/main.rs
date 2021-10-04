extern crate protoc_rust;

use std::env;
use std::error::Error;
use std::path::Path;
use std::result::Result;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let tensorflow_folder = &args[1];
    let output_folder = Path::new(&args[2]);
    protoc_rust::run(protoc_rust::Args {
        out_dir: output_folder
            .join("src/protos")
            .to_str()
            .ok_or("Unable to format output path for main crate")?,
        input: &[
            &format!(
                "{}/tensorflow/core/framework/attr_value.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/full_type.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/function.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/graph.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/node_def.proto",
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
                "{}/tensorflow/core/protobuf/saved_model.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/protobuf/saved_object_graph.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/protobuf/struct.proto",
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
                "{}/tensorflow/core/protobuf/trackable_object_graph.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/types.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/variable.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/framework/versions.proto",
                tensorflow_folder
            ),
            &format!(
                "{}/tensorflow/core/protobuf/meta_graph.proto",
                tensorflow_folder
            ),
            &format!("{}/tensorflow/core/protobuf/saver.proto", tensorflow_folder),
        ],
        includes: &[tensorflow_folder],
        customize: protoc_rust::Customize {
            ..Default::default()
        },
    })?;
    protoc_rust::run(protoc_rust::Args {
        out_dir: output_folder
            .join("tensorflow-op-codegen/src/protos")
            .to_str()
            .ok_or("Unable to format output path for ops crate")?,
        input: &[
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
        ],
        includes: &[tensorflow_folder],
        customize: protoc_rust::Customize {
            ..Default::default()
        },
    })?;
    Ok(())
}
