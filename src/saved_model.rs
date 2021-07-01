use super::ops;
use super::protos;
use super::Code;
use super::DataType;
use super::Graph;
use super::Operation;
use super::Output;
use super::OutputName;
use super::Result;
use super::Scope;
use super::Session;
use super::SessionRunArgs;
use super::Shape;
use super::Status;
use super::Tensor;
use super::Variable;
use protobuf::Message;
use protobuf::ProtobufError;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;

/// Key in the signature def map for `default` serving signatures. The default
/// signature is used in inference requests where a specific signature was not
/// specified.
pub const DEFAULT_SERVING_SIGNATURE_DEF_KEY: &str = "serving_default";

/// Classification inputs.
pub const CLASSIFY_INPUTS: &str = "inputs";

/// Classification method name used in a SignatureDef.
pub const CLASSIFY_METHOD_NAME: &str = "tensorflow/serving/classify";

/// Classification classes output.
pub const CLASSIFY_OUTPUT_CLASSES: &str = "classes";

/// Classification scores output.
pub const CLASSIFY_OUTPUT_SCORES: &str = "scores";

/// Predict inputs.
pub const PREDICT_INPUTS: &str = "inputs";

/// Prediction method name used in a SignatureDef.
pub const PREDICT_METHOD_NAME: &str = "tensorflow/serving/predict";

/// Predict outputs.
pub const PREDICT_OUTPUTS: &str = "outputs";

/// Regression inputs.
pub const REGRESS_INPUTS: &str = "inputs";

/// Regression method name used in a SignatureDef.
pub const REGRESS_METHOD_NAME: &str = "tensorflow/serving/regress";

///  Regression outputs.
pub const REGRESS_OUTPUTS: &str = "outputs";

/// Error generated while saving a model.
#[derive(Debug)]
pub struct SaveModelError {
    source: Box<dyn Error>,
}

impl SaveModelError {
    // We don't use From, because we don't want this to be public API.
    fn from_protobuf_error(e: ProtobufError) -> Self {
        Self {
            source: Box::new(e),
        }
    }
}

impl Display for SaveModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "SaveModelError: {}", &self.source)
    }
}

impl Error for SaveModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.borrow())
    }
}

impl From<Status> for SaveModelError {
    fn from(e: Status) -> Self {
        Self {
            source: Box::new(e),
        }
    }
}

impl From<io::Error> for SaveModelError {
    fn from(e: io::Error) -> Self {
        Self {
            source: Box::new(e),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
/// Information about a Tensor necessary for feeding or retrieval.
pub struct TensorInfo {
    dtype: DataType,
    shape: Shape,
    name: OutputName,
}

impl TensorInfo {
    /// Creates a TensorInfo.
    pub fn new(dtype: DataType, shape: Shape, name: OutputName) -> TensorInfo {
        TensorInfo { dtype, shape, name }
    }

    /// Returns the name of the tensor.
    pub fn name(&self) -> &OutputName {
        &self.name
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::meta_graph::TensorInfo {
        let mut info = protos::meta_graph::TensorInfo::new();
        info.set_dtype(self.dtype.into_proto());
        info.set_tensor_shape(self.shape.into_proto());
        info.set_name(self.name.to_string());
        info
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::meta_graph::TensorInfo) -> Result<Self> {
        Ok(Self {
            dtype: DataType::from_proto(proto.get_dtype()),
            shape: Shape::from_proto(proto.get_tensor_shape()),
            name: proto.get_name().parse()?,
        })
    }
}

#[derive(Debug, Clone)]
/// SignatureDef defines the signature of a computation supported by a
/// TensorFlow graph.
pub struct SignatureDef {
    method_name: String,
    inputs: HashMap<String, TensorInfo>,
    outputs: HashMap<String, TensorInfo>,
}

impl SignatureDef {
    /// Creates a SignatureDef with the given method name.
    pub fn new(method_name: String) -> SignatureDef {
        SignatureDef {
            method_name,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    /// Adds an input parameter.
    pub fn add_input_info(&mut self, name: String, info: TensorInfo) {
        self.inputs.insert(name, info);
    }

    /// Adds an output parameter.
    pub fn add_output_info(&mut self, name: String, info: TensorInfo) {
        self.outputs.insert(name, info);
    }

    /// Returns the method name.
    pub fn method_name(&self) -> &str {
        &self.method_name
    }

    /// Returns the input parameters.
    pub fn inputs(&self) -> &HashMap<String, TensorInfo> {
        &self.inputs
    }

    /// Returns the output parameters.
    pub fn outputs(&self) -> &HashMap<String, TensorInfo> {
        &self.outputs
    }

    /// Returns the given input parameter.
    pub fn get_input(&self, name: &str) -> Result<&TensorInfo> {
        self.inputs.get(name).ok_or_else(|| {
            Status::new_set_lossy(
                Code::InvalidArgument,
                &format!("Input '{}' not found", name),
            )
        })
    }

    /// Returns the given output parameter.
    pub fn get_output(&self, name: &str) -> Result<&TensorInfo> {
        self.outputs.get(name).ok_or_else(|| {
            Status::new_set_lossy(
                Code::InvalidArgument,
                &format!("Output '{}' not found", name),
            )
        })
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::meta_graph::SignatureDef {
        let mut signature_def = protos::meta_graph::SignatureDef::new();
        signature_def.set_method_name(self.method_name);
        for (name, info) in self.inputs {
            signature_def.mut_inputs().insert(name, info.into_proto());
        }
        for (name, info) in self.outputs {
            signature_def.mut_outputs().insert(name, info.into_proto());
        }
        signature_def
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::meta_graph::SignatureDef) -> Result<Self> {
        let mut inputs = HashMap::new();
        let mut outputs = HashMap::new();
        for (key, proto) in proto.get_inputs() {
            inputs.insert(key.clone(), TensorInfo::from_proto(proto)?);
        }
        for (key, proto) in proto.get_outputs() {
            outputs.insert(key.clone(), TensorInfo::from_proto(proto)?);
        }
        Ok(Self {
            method_name: proto.get_method_name().to_string(),
            inputs,
            outputs,
        })
    }
}

#[derive(Debug, Clone)]
/// Contains data necessary to restart training, run inference. It can be used
/// to serialize/de-serialize memory objects necessary for running computation
/// in a graph when crossing the process boundary. It can be used for long term
/// storage of graphs, cross-language execution of graphs, etc.
pub struct MetaGraphDef {
    // TODO: support all fields
    signatures: HashMap<String, SignatureDef>,
}

impl MetaGraphDef {
    // We don't use From, because we don't want this to be public API.
    pub(crate) fn from_serialized_proto(data: &[u8]) -> Result<Self> {
        let proto: protos::meta_graph::MetaGraphDef = protobuf::Message::parse_from_bytes(data)
            .map_err(|e| {
                Status::new_set_lossy(
                    Code::InvalidArgument,
                    &format!("Invalid serialized MetaGraphDef: {}", e),
                )
            })?;
        let mut signatures = HashMap::new();
        for (key, signature_proto) in proto.get_signature_def() {
            signatures.insert(key.clone(), SignatureDef::from_proto(signature_proto)?);
        }
        Ok(Self { signatures })
    }

    /// Returns the defined signatures.
    pub fn signatures(&self) -> &HashMap<String, SignatureDef> {
        &self.signatures
    }

    /// Returns the specified signature.
    pub fn get_signature(&self, name: &str) -> Result<&SignatureDef> {
        self.signatures.get(name).ok_or_else(|| {
            Status::new_set_lossy(Code::Internal, &format!("Signature '{}' not found", name))
        })
    }
}

/// Builds a SavedModelSaver, which can be used to save models.
#[derive(Debug)]
pub struct SavedModelBuilder {
    collections: HashMap<String, Vec<Variable>>,
    tags: Vec<String>,
    signatures: HashMap<String, SignatureDef>,
}

impl Default for SavedModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SavedModelBuilder {
    /// Creates a new SavedModelBuilder.
    pub fn new() -> SavedModelBuilder {
        SavedModelBuilder {
            collections: HashMap::new(),
            tags: Vec::new(),
            signatures: HashMap::new(),
        }
    }

    /// Adds a collection to be saved.
    pub fn add_collection(&mut self, key: &str, variables: &[Variable]) -> &mut Self {
        self.collections.insert(key.to_string(), variables.to_vec());
        self
    }

    /// Adds a tag.
    pub fn add_tag(&mut self, tag: &str) -> &mut Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Adds a signature.
    pub fn add_signature(&mut self, key: &str, signature_def: SignatureDef) -> &mut Self {
        self.signatures.insert(key.to_string(), signature_def);
        self
    }

    /// Adds ops to the graph necessary for saving and restoring models,
    /// returning a SavedModelSaver which handles the actual model saving.
    pub fn inject(self, scope: &mut Scope) -> Result<SavedModelSaver> {
        let all_vars = self.collections.values().flatten().collect::<Vec<_>>();
        let prefix = ops::Placeholder::new()
            .dtype(DataType::String)
            .build(scope)?;
        let save_op = {
            let tensor_names = ops::constant(
                &all_vars
                    .iter()
                    .map(|v| v.name().to_string())
                    .collect::<Vec<_>>()[..],
                scope,
            )?;
            let shape_and_slices = ops::constant(
                &all_vars.iter().map(|_| "".to_string()).collect::<Vec<_>>()[..],
                scope,
            )?;
            let tensors = all_vars
                .iter()
                .map(|v| v.output().clone())
                .collect::<Vec<_>>();
            let mut g = scope.graph_mut();
            let mut nd = g.new_operation("SaveV2", "save")?;
            nd.add_input(prefix.clone());
            nd.add_input(tensor_names);
            nd.add_input(shape_and_slices);
            nd.add_input_list(&tensors[..]);
            nd.set_attr_type_list(
                "dtypes",
                &all_vars.iter().map(|v| v.data_type()).collect::<Vec<_>>()[..],
            )?;
            nd.finish()?
        };

        let filename_tensor = ops::Placeholder::new()
            .dtype(DataType::String)
            .build(scope)?;
        let restore_op = {
            let all_var_names = all_vars
                .iter()
                .map(|v| v.name().to_string())
                .collect::<Vec<_>>();
            let tensor_names = ops::constant(&all_var_names[..], scope)?;
            let shape_and_slices = ops::constant(
                &all_vars.iter().map(|_| "".to_string()).collect::<Vec<_>>()[..],
                scope,
            )?;
            let mut g = scope.graph_mut();
            let mut nd = g.new_operation("RestoreV2", "restore")?;
            nd.add_input(filename_tensor.clone());
            nd.add_input(tensor_names);
            nd.add_input(shape_and_slices);
            nd.set_attr_type_list(
                "dtypes",
                &all_vars.iter().map(|v| v.data_type()).collect::<Vec<_>>()[..],
            )?;
            nd.finish()?
        };
        let really_restore_op = {
            let mut restore_var_ops = Vec::<Operation>::new();
            for (i, var) in all_vars.iter().enumerate() {
                restore_var_ops.push(ops::assign(
                    var.output().clone(),
                    Output {
                        operation: restore_op.clone(),
                        index: i as i32,
                    },
                    scope,
                )?);
            }
            let mut no_op = ops::NoOp::new();
            for op in restore_var_ops {
                no_op = no_op.add_control_input(op);
            }
            no_op.build(scope)?
        };

        SavedModelSaver::new(
            filename_tensor.name()?,
            prefix,
            save_op,
            really_restore_op.name()?,
            self.collections,
            self.tags,
            self.signatures,
        )
    }
}

#[derive(Debug)]
/// Creates saved models. Use a SavedModelBuilder to create a SavedModelSaver.
pub struct SavedModelSaver {
    meta_graph: protos::meta_graph::MetaGraphDef,
    prefix: Operation,
    save_op: Operation,
}

impl SavedModelSaver {
    fn new(
        filename_tensor_name: String,
        prefix: Operation,
        save_op: Operation,
        restore_op_name: String,
        collections: HashMap<String, Vec<Variable>>,
        tags: Vec<String>,
        signatures: HashMap<String, SignatureDef>,
    ) -> Result<SavedModelSaver> {
        let mut meta_graph = protos::meta_graph::MetaGraphDef::new();
        meta_graph
            .mut_saver_def()
            .set_filename_tensor_name(filename_tensor_name);
        meta_graph
            .mut_saver_def()
            .set_restore_op_name(restore_op_name);
        for (key, variables) in collections {
            let mut trainable_variables_bytes_list =
                protos::meta_graph::CollectionDef_BytesList::new();
            for variable in variables {
                let mut variable_def = protos::variable::VariableDef::new();
                variable_def.set_variable_name(variable.name().to_string());
                trainable_variables_bytes_list.mut_value().push(
                    match variable_def.write_to_bytes() {
                        Ok(x) => x,
                        Err(e) => {
                            return Err(Status::new_set_lossy(
                                Code::InvalidArgument,
                                &format!("Unable to encode variable definition: {}", e),
                            ));
                        }
                    },
                );
            }
            let mut trainable_collection_def = protos::meta_graph::CollectionDef::new();
            trainable_collection_def.set_bytes_list(trainable_variables_bytes_list);
            meta_graph
                .mut_collection_def()
                .insert(key.to_string(), trainable_collection_def);
        }
        let graph_tags = meta_graph.mut_meta_info_def().mut_tags();
        for tag in tags {
            graph_tags.push(tag);
        }
        let graph_signatures = meta_graph.mut_signature_def();
        for (key, signature) in signatures {
            graph_signatures.insert(key, signature.into_proto());
        }
        Ok(SavedModelSaver {
            meta_graph,
            prefix,
            save_op,
        })
    }

    /// Saves the graph and current variable values as a saved model.
    pub fn save<P: AsRef<Path>>(
        &self,
        session: &Session,
        graph: &Graph,
        save_dir: P,
    ) -> std::result::Result<(), SaveModelError> {
        let mut meta_graph = self.meta_graph.clone();
        let graph_bytes = graph.graph_def()?;
        let graph_def = match protobuf::Message::parse_from_bytes(&graph_bytes) {
            Ok(x) => x,
            Err(e) => {
                return Err(Status::new_set_lossy(
                    Code::InvalidArgument,
                    &format!("Unable to parse graph definition: {}", e),
                )
                .into());
            }
        };
        meta_graph.set_graph_def(graph_def);
        let mut saved_model = protos::saved_model::SavedModel::new();
        saved_model.set_saved_model_schema_version(1);
        saved_model.mut_meta_graphs().push(meta_graph);
        let saved_model_bytes = saved_model
            .write_to_bytes()
            .map_err(SaveModelError::from_protobuf_error)?;
        fs::create_dir(save_dir.as_ref())?;
        let mut file = File::create(save_dir.as_ref().join("saved_model.pb"))?;
        file.write_all(&saved_model_bytes)?;
        let prefix = Tensor::from(
            save_dir
                .as_ref()
                .join("variables/variables")
                .to_str()
                .ok_or_else(|| {
                    Status::new_set(Code::OutOfRange, "Path is not valid Unicode").unwrap()
                })?
                .to_string(),
        );

        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.prefix, 0, &prefix);
        run_args.add_target(&self.save_op);
        session.run(&mut run_args)?;
        Ok(())
    }
}
