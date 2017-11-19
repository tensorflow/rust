extern crate tensorflow_sys as tf;

use libc::c_float;
use libc::c_int;
use libc::c_uchar;
use libc::c_void;
use libc::size_t;
use std;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::os::raw::c_void as std_c_void;
use std::ptr;
use std::str::Utf8Error;
use std::sync::Arc;
use super::AnyTensor;
use super::buffer::Buffer;
use super::BufferTrait;
use super::Code;
use super::DataType;
use super::GraphTrait;
use super::OperationTrait;
use super::Shape;
use super::Status;
use super::Result;
use super::Tensor;
use super::TensorType;

#[derive(Debug)]
struct GraphLifetime;

#[derive(Debug)]
struct GraphImpl {
    inner: *mut tf::TF_Graph,
}

impl Drop for GraphImpl {
    /// Graph will be deleted once no more Sessions are referencing it.
    fn drop(&mut self) {
        unsafe {
            tf::TF_DeleteGraph(self.inner);
        }
    }
}

////////////////////////

/// `ImportGraphDefOptions` holds options that can be passed to
/// `Graph::import_graph_def`.
#[derive(Debug)]
pub struct ImportGraphDefOptions {
    inner: *mut tf::TF_ImportGraphDefOptions,
}

impl_new!(ImportGraphDefOptions,
          TF_NewImportGraphDefOptions,
          "Creates a default ImportGraphDefOptions.");
impl_drop!(ImportGraphDefOptions, TF_DeleteImportGraphDefOptions);

impl ImportGraphDefOptions {
    /// Set the prefix to be prepended to the names of nodes in `graph_def` that will
    /// be imported into `graph`.
    pub fn set_prefix(&mut self, prefix: &str) -> std::result::Result<(), NulError> {
        let s = CString::new(prefix)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsSetPrefix(self.inner, s.as_ptr());
        }
        Ok(())
    }

    /// Set any imported nodes with input `src_name:src_index` to have that input
    /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
    /// `dst` references a node already existing in the graph being imported into.
    pub fn add_input_mapping(&mut self,
                             src_name: &str,
                             src_index: usize,
                             dst: &Output)
                             -> std::result::Result<(), NulError> {
        let s = CString::new(src_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsAddInputMapping(self.inner,
                                                        s.as_ptr(),
                                                        src_index as c_int,
                                                        dst.to_c());
        }
        Ok(())
    }

    /// Set any imported nodes with control input `src_name` to have that input
    /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
    /// `dst` references an operation already existing in the graph being imported
    /// into.
    pub fn remap_control_dependency(&mut self,
                                    src_name: &str,
                                    dst: &Operation)
                                    -> std::result::Result<(), NulError> {
        let s = CString::new(src_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsRemapControlDependency(self.inner,
                                                               s.as_ptr(),
                                                               dst.inner);
        }
        Ok(())
    }

    /// Cause the imported graph to have a control dependency on `oper`. `oper`
    /// should exist in the graph being imported into.
    pub fn add_control_dependency(&mut self, oper: &Operation) {
        unsafe {
            tf::TF_ImportGraphDefOptionsAddControlDependency(self.inner, oper.inner);
        }
    }

    /// Add an output in `graph_def` to be returned via the `return_outputs` output
    /// parameter of `import_graph_def()`. If the output is remapped via an input
    /// mapping, the corresponding existing tensor in `graph` will be returned.
    pub fn add_return_output(&mut self,
                             oper_name: &str,
                             index: usize)
                             -> std::result::Result<(), NulError> {
        let s = CString::new(oper_name)?;
        unsafe {
            tf::TF_ImportGraphDefOptionsAddReturnOutput(self.inner, s.as_ptr(), index as c_int);
        }
        Ok(())
    }

    /// Returns the number of return outputs added via `add_return_output()`.
    pub fn num_return_outputs(&self) -> usize {
        unsafe { tf::TF_ImportGraphDefOptionsNumReturnOutputs(self.inner) as usize }
    }
}

////////////////////////

/// Represents a computation graph.  Graphs may be shared between sessions.
/// Graphs are thread-safe when used as directed.
#[derive(Debug)]
pub struct Graph {
    gimpl: Arc<GraphImpl>,
    lifetime: GraphLifetime,
}

impl Graph {
    /// Creates a new graph.
    pub fn new() -> Graph {
        unsafe {
            Graph {
                gimpl: Arc::new(GraphImpl { inner: tf::TF_NewGraph() }),
                lifetime: GraphLifetime,
            }
        }
    }

    /// Operation will only be added to graph when finish_operation() is called
    /// (assuming finish_operation() does not return an error).  graph must
    /// not be deleted until after finish_operation() is called.
    pub fn new_operation(&mut self,
                         op_type: &str,
                         operation_name: &str)
                         -> std::result::Result<OperationDescription, NulError> {
        let c_op_type = CString::new(op_type)?;
        let c_operation_name = CString::new(operation_name)?;
        unsafe {
            Ok(OperationDescription {
                inner: tf::TF_NewOperation(self.gimpl.inner,
                                           c_op_type.as_ptr(),
                                           c_operation_name.as_ptr()),
                graph: self,
                finished: false,
            })
        }
    }

    /// Returns the operation in the graph with the given name, if it exists.
    /// If the operation does not exist, returns `Ok(None)`.
    pub fn operation_by_name(&self,
                             operation_name: &str)
                             -> std::result::Result<Option<Operation>, NulError> {
        let c_operation_name = CString::new(operation_name)?;
        unsafe {
            let operation = tf::TF_GraphOperationByName(self.gimpl.inner,
                                                        c_operation_name.as_ptr());
            if operation.is_null() {
                Ok(None)
            } else {
                Ok(Some(Operation {
                    inner: operation,
                    gimpl: self.gimpl.clone(),
                }))
            }
        }
    }

    /// Like `operation_by_name`, except that failure to find the operation is considered an error.
    pub fn operation_by_name_required(&self,
                                      operation_name: &str)
                                      -> std::result::Result<Operation, Status> {
        match self.operation_by_name(operation_name)? {
            Some(operation) => Ok(operation),
            None => {
                Err(Status::new_set(Code::Unavailable,
                                    &format!("Operation {:?} not found", operation_name))
                    .unwrap())
            }
        }
    }

    /// Iterates over the operations in the graph.
    pub fn operation_iter(&self) -> OperationIter {
        OperationIter {
            graph: self,
            pos: 0,
        }
    }

    /// Returns the graph definition as a protobuf.
    pub fn graph_def(&self) -> Result<Vec<u8>> {
        let mut status = Status::new();
        unsafe {
            let c_buffer = tf::TF_NewBuffer();
            tf::TF_GraphToGraphDef(self.gimpl.inner, c_buffer, status.inner());
            if status.is_ok() {
                Ok(Buffer::from_c(c_buffer, true).into())
            } else {
                tf::TF_DeleteBuffer(c_buffer);
                Err(status)
            }
        }
    }

    /// Returns the number of dimensions of the Tensor referenced by `output`.
    ///
    /// If the number of dimensions in the shape is unknown, returns -1.
    ///
    /// Returns an error if:
    ///   * `output` is not in `graph`.
    pub fn num_dims(&self, output: Output) -> Result<c_int> {
        let mut status = Status::new();
        unsafe {
            let val = tf::TF_GraphGetTensorNumDims(self.gimpl.inner, output.to_c(), status.inner());
            if status.is_ok() { Ok(val) } else { Err(status) }
        }
    }

    /// Returns the shape of the Tensor referenced by `output`.
    ///
    /// Returns an error if:
    ///   * `output` is not in `graph`.
    pub fn tensor_shape(&self, output: Output) -> Result<Shape> {
        let mut status = Status::new();
        let n = self.num_dims(output.clone())?;
        if n == -1 {
            return Ok(Shape(None));
        }
        let mut dims = Vec::with_capacity(n as usize);
        unsafe {
            tf::TF_GraphGetTensorShape(self.gimpl.inner,
                                       output.to_c(),
                                       dims.as_mut_ptr(),
                                       n,
                                       status.inner());
            if status.is_ok() {
                dims.set_len(n as usize);
                Ok(Shape(Some(dims.iter().map(|x| if *x < 0 { None } else { Some(*x) }).collect())))
            } else {
                Err(status)
            }
        }
    }

    /// Import the graph serialized in `graph_def`.
    pub fn import_graph_def(&mut self,
                            graph_def: &[u8],
                            options: &ImportGraphDefOptions)
                            -> Result<()> {
        let buf = Buffer::from(graph_def);
        let mut status = Status::new();
        unsafe {
            tf::TF_GraphImportGraphDef(self.gimpl.inner,
                                       buf.inner(),
                                       options.inner,
                                       status.inner());
            status.into_result()
        }
    }

    /// Import the graph serialized in `graph_def`.
    pub fn import_graph_def_with_return_outputs(&mut self,
                                                graph_def: &[u8],
                                                options: &ImportGraphDefOptions)
                                                -> Result<Vec<Output>> {
        let buf = Buffer::from(graph_def);
        let mut status = Status::new();
        let mut c_return_outputs = Vec::new();
        let n = options.num_return_outputs();
        unsafe {
            c_return_outputs.set_len(n);
            tf::TF_GraphImportGraphDefWithReturnOutputs(self.gimpl.inner,
                                                        buf.inner(),
                                                        options.inner,
                                                        c_return_outputs.as_mut_ptr(),
                                                        n as c_int,
                                                        status.inner());
        }
        status.into_result()?;
        Ok(c_return_outputs
               .iter()
               .map(|x| Output::from_c(self, x))
               .collect())
    }
}

impl GraphTrait for Graph {
    fn inner(&self) -> *mut tf::TF_Graph {
        self.gimpl.inner
    }
}

////////////////////////

/// Iterator over the operations in a `Graph`.
#[derive(Debug)]
pub struct OperationIter<'a> {
    // We could just have a gimpl field, but keeping a reference to the Graph
    // means that the graph can't be modified while iterating through it.
    graph: &'a Graph,
    pos: size_t,
}

impl<'a> Iterator for OperationIter<'a> {
    type Item = Operation;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let operation = tf::TF_GraphNextOperation(self.graph.gimpl.inner, &mut self.pos);
            if operation.is_null() {
                None
            } else {
                Some(Operation {
                    inner: operation,
                    gimpl: self.graph.gimpl.clone(),
                })
            }
        }
    }
}

////////////////////////

/// An `Operation` is a node in a `Graph`.
/// It is a computation which accepts inputs and produces outputs.
#[derive(Debug,Clone)]
pub struct Operation {
    inner: *mut tf::TF_Operation,
    gimpl: Arc<GraphImpl>,
}

impl Operation {
    /// Returns the name of the operation.
    ///
    /// This is the name of the specific computational step,
    /// not an operation type, so it may look like `'add_x_and_y'` instead of `'Add'`,
    /// although it may be a generated ID like `'Add_123'`.
    pub fn name(&self) -> std::result::Result<String, Utf8Error> {
        unsafe { CStr::from_ptr(tf::TF_OperationName(self.inner)).to_str().map(|x| x.to_string()) }
    }

    /// Returns the type of operation.
    /// This will be something like `'Add'`, `'Mul'`, etc.
    pub fn op_type(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_OperationOpType(self.inner)).to_str().map(|x| x.to_string())
        }
    }

    /// Returns the device for this operation.
    /// The empty string means unconstrained.
    pub fn device(&self) -> std::result::Result<String, Utf8Error> {
        unsafe {
            CStr::from_ptr(tf::TF_OperationOpType(self.inner)).to_str().map(|x| x.to_string())
        }
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        unsafe { tf::TF_OperationNumOutputs(self.inner) as usize }
    }

    /// Returns the type of a specific output.
    pub fn output_type(&self, index: usize) -> DataType {
        unsafe {
            DataType::from_c(tf::TF_OperationOutputType(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            }))
        }
    }

    // TODO: Figure out what this does and document it.
    #[allow(missing_docs)]
    pub fn output_list_length(&self, arg_name: &str) -> Result<usize> {
        let c_arg_name = CString::new(arg_name)?;
        let mut status = Status::new();
        let length = unsafe {
            tf::TF_OperationOutputListLength(self.inner, c_arg_name.as_ptr(), status.inner())
        };
        if status.is_ok() {
            Ok(length as usize)
        } else {
            Err(status)
        }
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        unsafe { tf::TF_OperationNumInputs(self.inner) as usize }
    }

    /// Returns the type of a specific input.
    pub fn input_type(&self, index: usize) -> DataType {
        unsafe {
            DataType::from_c(tf::TF_OperationInputType(tf::TF_Input {
                oper: self.inner,
                index: index as c_int,
            }))
        }
    }

    // TODO: Figure out what this does and document it.
    #[allow(missing_docs)]
    pub fn input_list_length(&self, arg_name: &str) -> Result<usize> {
        let c_arg_name = CString::new(arg_name)?;
        let mut status = Status::new();
        let length = unsafe {
            tf::TF_OperationInputListLength(self.inner, c_arg_name.as_ptr(), status.inner())
        };
        if status.is_ok() {
            Ok(length as usize)
        } else {
            Err(status)
        }
    }

    /// Returns the given input edge.
    /// The index argument is the index into the current operation's input array,
    /// and the return value is the source operation and the index into its output array.
    pub fn input(&self, index: usize) -> (Operation, usize) {
        unsafe {
            let port = tf::TF_OperationInput(tf::TF_Input {
                oper: self.inner,
                index: index as c_int,
            });
            (Operation {
                 inner: port.oper,
                 gimpl: self.gimpl.clone(),
             },
             port.index as usize)
        }
    }

    /// Returns the number of consumers of a specific output.
    pub fn output_num_consumers(&self, index: usize) -> usize {
        unsafe {
            tf::TF_OperationOutputNumConsumers(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            }) as usize
        }
    }

    /// Returns the consumers of a specific output.
    /// The index argument is the index into the current operation's output array,
    /// and the return value is a vector of the destination operation and the index
    /// into its input array.
    pub fn output_consumers(&self, index: usize) -> Vec<(Operation, usize)> {
        unsafe {
            let num_consumers = tf::TF_OperationOutputNumConsumers(tf::TF_Output {
                oper: self.inner,
                index: index as c_int,
            });
            let mut vec = <Vec<tf::TF_Input>>::with_capacity(num_consumers as usize);
            let len = tf::TF_OperationOutputConsumers(tf::TF_Output {
                                                          oper: self.inner,
                                                          index: index as c_int,
                                                      },
                                                      vec.as_mut_ptr(),
                                                      vec.len() as c_int);
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|port| {
                    (Operation {
                         inner: port.oper,
                         gimpl: self.gimpl.clone(),
                     },
                     port.index as usize)
                })
                .collect()
        }
    }

    /// Returns the number of control inputs.
    pub fn num_control_inputs(&self) -> usize {
        unsafe { tf::TF_OperationNumControlInputs(self.inner) as usize }
    }

    /// Returns the control inputs.
    pub fn control_inputs(&self) -> Vec<Operation> {
        unsafe {
            let num_consumers = tf::TF_OperationNumControlInputs(self.inner);
            let mut vec =
                <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
            let len =
                tf::TF_OperationGetControlInputs(self.inner, vec.as_mut_ptr(), vec.len() as c_int);
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|operation| {
                    Operation {
                        inner: operation,
                        gimpl: self.gimpl.clone(),
                    }
                })
                .collect()
        }
    }

    /// Returns the number of control outputs.
    pub fn num_control_outputs(&self) -> usize {
        unsafe { tf::TF_OperationNumControlOutputs(self.inner) as usize }
    }

    /// Returns the control outputs.
    pub fn control_outputs(&self) -> Vec<Operation> {
        unsafe {
            let num_consumers = tf::TF_OperationNumControlOutputs(self.inner);
            let mut vec =
                <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
            let len =
                tf::TF_OperationGetControlOutputs(self.inner, vec.as_mut_ptr(), vec.len() as c_int);
            vec.set_len(len as usize);
            vec.into_iter()
                .map(|operation| {
                    Operation {
                        inner: operation,
                        gimpl: self.gimpl.clone(),
                    }
                })
                .collect()
        }
    }
}

impl OperationTrait for Operation {
    fn inner(&self) -> *mut tf::TF_Operation {
        self.inner
    }
}

////////////////////////

/// A `Input` is one end of a graph edge.
/// It holds an operation and an index into the inputs of that operation.
#[derive(Debug,Copy,Clone)]
pub struct Input<'a> {
    /// Operation the edge connects to.
    pub operation: &'a Operation,

    /// Index into either the inputs of the operation.
    pub index: c_int,
}

impl<'a> Input<'a> {
    fn to_c(&self) -> tf::TF_Input {
        tf::TF_Input {
            oper: self.operation.inner,
            index: self.index,
        }
    }
}

////////////////////////

/// A `Output` is one end of a graph edge.
/// It holds an operation and an index into the outputs of that operation.
#[derive(Debug,Clone)]
pub struct Output {
    /// Operation the edge connects to.
    pub operation: Operation,

    /// Index into either the outputs of the operation.
    pub index: c_int,
}

impl Output {
    fn to_c(&self) -> tf::TF_Output {
        tf::TF_Output {
            oper: self.operation.inner,
            index: self.index,
        }
    }

    fn from_c(graph: &Graph, output: &tf::TF_Output) -> Self {
        Output {
            operation: Operation {
                inner: output.oper,
                gimpl: graph.gimpl.clone(),
            },
            index: output.index,
        }
    }
}

////////////////////////

/// An `OperationDescription` is an `Operation` in the process of being built
/// (i.e. the builder pattern).
///
/// An `OperationDescription` is required to be finished before the graph
/// goes out of scope,
/// so `finish()` will be called on drop if it was not already called.
#[derive(Debug)]
pub struct OperationDescription<'a> {
    inner: *mut tf::TF_OperationDescription,
    // This keeps self from outliving the Graph, which is required by
    // the docs on TF_NewOperation.
    graph: &'a Graph,
    finished: bool,
}

impl<'a> Drop for OperationDescription<'a> {
    fn drop(&mut self) {
        if !self.finished {
            unsafe {
                // TF_NewOperation requires us to make sure TF_FinishOperation is called before the
                // graph is deleted.  Combined with guaranteeing that OperationDescription does
                // not outlive Graph, this ensures that the contract is held.
                let status = tf::TF_NewStatus();
                tf::TF_FinishOperation(self.inner, status);
                tf::TF_DeleteStatus(status);
            }
        }
    }
}

impl<'a> OperationDescription<'a> {
    /// Builds the operation and adds it to the graph.
    pub fn finish(mut self) -> Result<Operation> {
        self.finished = true; // used by the drop code
        let mut status = Status::new();
        let operation = unsafe { tf::TF_FinishOperation(self.inner, status.inner()) };
        if status.is_ok() {
            Ok(Operation {
                inner: operation,
                gimpl: self.graph.gimpl.clone(),
            })
        } else {
            Err(status)
        }
    }

    /// Sets the preferred device.
    /// The empty string means unconstrained.
    pub fn set_device(&mut self, device: &str) -> std::result::Result<(), NulError> {
        let c_device = CString::new(device)?;
        unsafe {
            tf::TF_SetDevice(self.inner, c_device.as_ptr());
        }
        Ok(())
    }

    /// Adds an input to this operation.
    ///
    /// The index in the port is an index into the source operation's output array.
    pub fn add_input(&mut self, input: Output) {
        unsafe {
            tf::TF_AddInput(self.inner, input.to_c());
        }
    }

    /// Adds multiple inputs to this operation.
    ///
    /// The index in the ports is an index into the source operation's output array.
    pub fn add_input_list(&mut self, inputs: &[Output]) {
        let c_inputs: Vec<tf::TF_Output> = inputs.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TF_AddInputList(self.inner, c_inputs.as_ptr(), c_inputs.len() as c_int);
        }
    }

    /// Adds a control input.
    pub fn add_control_input(&mut self, input: &Operation) {
        unsafe {
            tf::TF_AddControlInput(self.inner, input.inner);
        }
    }

    /// Sets the value of a string attribute.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_string(&mut self,
                           attr_name: &str,
                           value: &str)
                           -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value = value.as_bytes();
        unsafe {
            tf::TF_SetAttrString(self.inner,
                                 c_attr_name.as_ptr(),
                                 c_value.as_ptr() as *const std_c_void,
                                 c_value.len() as size_t);
        }
        Ok(())
    }

    /// Sets the value of an attribute which holds a list of strings.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_string_list<S: AsRef<str>>(&mut self,
                                               attr_name: &str,
                                               value: &[S])
                                               -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let bytes: Vec<&[u8]> = value.iter().map(|x| x.as_ref().as_bytes()).collect();
        let ptrs: Vec<*const c_void> = bytes.iter().map(|x| x.as_ptr() as *const c_void).collect();
        let lens: Vec<size_t> = bytes.iter().map(|x| x.len() as size_t).collect();
        unsafe {
            tf::TF_SetAttrStringList(self.inner,
                                     c_attr_name.as_ptr(),
                                     ptrs.as_ptr() as *const *const std_c_void,
                                     lens.as_ptr(),
                                     ptrs.len() as c_int);
        }
        Ok(())
    }

    /// Sets an int-valued attribute.
    pub fn set_attr_int(&mut self,
                        attr_name: &str,
                        value: i64)
                        -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of ints.
    pub fn set_attr_int_list(&mut self,
                             attr_name: &str,
                             value: &[i64])
                             -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrIntList(self.inner,
                                  c_attr_name.as_ptr(),
                                  value.as_ptr(),
                                  value.len() as i32);
        }
        Ok(())
    }

    /// Sets a float-valued attribute.
    pub fn set_attr_float(&mut self,
                          attr_name: &str,
                          value: f32)
                          -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of floats.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_float_list(&mut self,
                               attr_name: &str,
                               value: &[f32])
                               -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        // Allow trivial_numeric_casts here because f32 is not necessarily equal to c_float.
        let c_value: Vec<c_float> = value.iter().map(|x| *x as c_float).collect();
        unsafe {
            tf::TF_SetAttrFloatList(self.inner,
                                    c_attr_name.as_ptr(),
                                    c_value.as_ptr(),
                                    c_value.len() as i32);
        }
        Ok(())
    }

    /// Sets a boolean-valued attribute.
    pub fn set_attr_bool(&mut self,
                         attr_name: &str,
                         value: bool)
                         -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of booleans.
    pub fn set_attr_bool_list(&mut self,
                              attr_name: &str,
                              value: &[bool])
                              -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<c_uchar> = value.iter().map(|x| if *x { 1 } else { 0 }).collect();
        unsafe {
            tf::TF_SetAttrBoolList(self.inner,
                                   c_attr_name.as_ptr(),
                                   c_value.as_ptr(),
                                   c_value.len() as c_int);
        }
        Ok(())
    }

    /// Sets a type-valued attribute.
    pub fn set_attr_type(&mut self,
                         attr_name: &str,
                         value: DataType)
                         -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TF_SetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of types.
    pub fn set_attr_type_list(&mut self,
                              attr_name: &str,
                              value: &[DataType])
                              -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TF_SetAttrTypeList(self.inner,
                                   c_attr_name.as_ptr(),
                                   c_value.as_ptr(),
                                   c_value.len() as i32);
        }
        Ok(())
    }

    /// Sets a shape-valued attribute.
    pub fn set_attr_shape(&mut self,
                          attr_name: &str,
                          value: &Shape)
                          -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            match value.0 {
                None => tf::TF_SetAttrShape(self.inner, c_attr_name.as_ptr(), ptr::null(), -1),
                Some(ref dims) => {
                    let c_dims: Vec<i64> = dims.iter()
                        .map(|x| match *x {
                            Some(d) => d,
                            None => -1,
                        })
                        .collect();
                    tf::TF_SetAttrShape(self.inner,
                                        c_attr_name.as_ptr(),
                                        c_dims.as_ptr(),
                                        c_dims.len() as i32);
                }
            }
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of shapes.
    pub fn set_attr_shape_list(&mut self,
                               attr_name: &str,
                               value: &[Shape])
                               -> std::result::Result<(), NulError> {
        let c_attr_name = CString::new(attr_name)?;
        // Convert Option<i64> in each shape to i64 with None becoming -1.
        let c_dims: Vec<Option<Vec<i64>>> = value.iter()
            .map(|x| match x.0 {
                None => None,
                Some(ref dims) => {
                    Some(dims.iter()
                        .map(|x| match *x {
                            None => -1,
                            Some(d) => d,
                        })
                        .collect())
                }
            })
            .collect();
        let ptrs: Vec<*const i64> = c_dims.iter()
            .map(|x| match *x {
                None => ptr::null(),
                Some(ref dims) => dims.as_ptr(),
            })
            .collect();
        let lens: Vec<c_int> = value.iter()
            .map(|x| match x.0 {
                None => -1,
                Some(ref dims) => dims.len() as c_int,
            })
            .collect();
        unsafe {
            tf::TF_SetAttrShapeList(self.inner,
                                    c_attr_name.as_ptr(),
                                    ptrs.as_ptr(),
                                    lens.as_ptr(),
                                    ptrs.len() as c_int);
        }
        Ok(())
    }

    /// Sets an attribute with a `TensorShapeProto` protobuf.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_tensor_shape_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensorShapeProto(self.inner,
                                           c_attr_name.as_ptr(),
                                           value.as_ptr() as *const std_c_void,
                                           value.len() as size_t,
                                           status.inner());
        }
        status.into_result()
    }

    /// Sets an attribute with an array of `TensorShapeProto` protobufs.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_tensor_shape_proto_list<T: AsRef<[u8]>>(&mut self,
                                                            attr_name: &str,
                                                            value: &[T])
                                                            -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let ptrs: Vec<*const c_void> = value.iter()
            .map(|x| x.as_ref().as_ptr() as *const c_void)
            .collect();
        let lens: Vec<size_t> = value.iter().map(|x| x.as_ref().len() as size_t).collect();
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensorShapeProtoList(self.inner,
                                               c_attr_name.as_ptr(),
                                               ptrs.as_ptr() as *const *const std_c_void,
                                               lens.as_ptr(),
                                               ptrs.len() as c_int,
                                               status.inner());
        }
        status.into_result()
    }

    /// Sets a tensor-valued attribute.
    pub fn set_attr_tensor<T: TensorType>(&mut self,
                                          attr_name: &str,
                                          value: Tensor<T>)
                                          -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrTensor(self.inner,
                                 c_attr_name.as_ptr(),
                                 value.inner()?,
                                 status.inner());
        }
        status.into_result()
    }

    /// Sets an attribute which holds an array of tensors.
    pub fn set_attr_tensor_list<I, T>(
        &mut self,
        attr_name: &str,
        value: I
        ) -> Result<()> 
        where I: IntoIterator<Item = Tensor<T>>, 
            T: TensorType 
    {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            let maybe_ptrs: Result<_> = value.into_iter().map(|x| x.inner()).collect();
            let ptrs: Vec<*mut tf::TF_Tensor> = maybe_ptrs?;
            tf::TF_SetAttrTensorList(self.inner,
                                     c_attr_name.as_ptr(),
                                     ptrs.as_ptr() as *const *const tf::TF_Tensor,
                                     ptrs.len() as c_int,
                                     status.inner());
        }
        status.into_result()
    }

    /// Sets an attribute with an `AttrValue` proto.
    #[allow(trivial_numeric_casts)]
    pub fn set_attr_to_attr_value_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TF_SetAttrValueProto(self.inner,
                                     c_attr_name.as_ptr(),
                                     value.as_ptr() as *const std_c_void,
                                     // Allow trivial_numeric_casts because usize is not
                                     // necessarily size_t.
                                     value.len() as size_t,
                                     status.inner());
        }
        status.into_result()
    }
}

////////////////////////

/// Options that can be passed during function creation.
#[derive(Debug)]
#[allow(missing_copy_implementations)]
pub struct FunctionOptions {
    inner: *mut tf::TF_FunctionOptions,
}

impl FunctionOptions {
    /// Creates a blank set of options.
    fn new() -> Self {
        FunctionOptions {
            inner: ptr::null_mut(), // TODO: Use real options when they become available
        }
    }
}

////////////////////////

/// Function is a grouping of operations with defined inputs and outputs.
/// Once created and added to graphs, functions can be invoked by creating an
/// operation whose operation type matches the function name.
#[derive(Debug)]
pub struct Function {
    inner: *mut tf::TF_Function,
}

impl_drop!(Function, TF_DeleteFunction);

impl Function {
    /// Returns a serialized representation of the function (as a FunctionDef
    /// protocol message).
    ///
    /// May fail on very large graphs in the future.
    pub fn to_function_def(&self) -> Result<Vec<u8>> {
        let status = Status::new();
        unsafe {
            let mut buf = Buffer::from_ptr(ptr::null_mut(), 0);
            tf::TF_FunctionToFunctionDef(self.inner, buf.inner_mut(), status.inner);
            status.into_result()?;
            Ok(buf.into())
        }
    }

    /// Construct and return the function whose FunctionDef representation is
    /// serialized in `proto`. Returns a newly created `Function` instance.
    pub fn import_function_def(proto: &[u8]) -> Result<Function> {
        let status = Status::new();
        unsafe {
            let inner = tf::TF_FunctionImportFunctionDef(
                proto.as_ptr() as *const std_c_void,
                proto.len(),
                status.inner,
            );
            status.into_result()?;
            Ok(Function { inner })
        }
    }

    /// Sets function attribute named `attr_name` to value stored in `proto`. If
    /// this attribute is already set to another value, it is overriden. `proto`
    /// should be a sequence of bytes representing a binary serialization of an
    /// AttrValue protocol buffer.
    pub fn set_attr_value_proto(&mut self, attr_name: &str, proto: &[u8]) -> Result<()> {
        let status = Status::new();
        let attr_name_cstr = CString::new(attr_name)?;
        unsafe {
            tf::TF_FunctionSetAttrValueProto(
                self.inner,
                attr_name_cstr.as_ptr(),
                proto.as_ptr() as *const std_c_void,
                proto.len(),
                status.inner,
            );
        }
        status.into_result()
    }

    /// Returns the binary-serialized AttrValue proto representation of the
    /// value of the `attr_name` attr of the function. If `attr_name` attribute
    /// is not present, returns an error.
    pub fn get_attr_value_proto(&self, attr_name: &str) -> Result<Vec<u8>> {
        let status = Status::new();
        let attr_name_cstr = CString::new(attr_name)?;
        unsafe {
            let mut buf = Buffer::from_ptr(ptr::null_mut(), 0);
            tf::TF_FunctionGetAttrValueProto(
                self.inner,
                attr_name_cstr.as_ptr(),
                buf.inner_mut(),
                status.inner,
            );
            status.into_result()?;
            Ok(buf.into())
        }
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::DataType;
    use super::super::Shape;

    fn add_operation(g: &mut Graph) {
        g.new_operation("Variable", "foo").unwrap();
    }

    #[test]
    fn smoke() {
        let mut g = Graph::new();
        add_operation(&mut g);
        let operation = {
            let mut nd = g.new_operation("Variable", "foo").unwrap();
            nd.set_attr_type("dtype", DataType::Float).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap()
        };
        let mut nd2 = g.new_operation("Variable", "foo2").unwrap();
        nd2.set_attr_type("dtype", DataType::Float).unwrap();
        nd2.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
        let operation2 = nd2.finish().unwrap();
        assert_eq!("foo", operation.name().unwrap());
        assert_eq!("foo2", operation2.name().unwrap());
    }

    #[test]
    fn test_import_graph_def() {
        let mut g = Graph::new();
        let opts = ImportGraphDefOptions::new();
        // An empty array is a valid proto, since all fields are optional.
        let status = g.import_graph_def(&[], &opts);
        assert!(status.is_ok());
    }

    #[test]
    fn test_get_tensor_shape() {
        fn constant<T: TensorType>(graph: &mut Graph, name: &str, value: Tensor<T>) -> Operation {
            let mut c = graph.new_operation("Const", name).unwrap();
            c.set_attr_tensor("value", value).unwrap();
            c.set_attr_type("dtype", T::data_type()).unwrap();
            c.finish().unwrap()
        }

        let mut graph = Graph::new();
        let x_init = Tensor::<i32>::new(&[3, 3]);
        let x = constant(&mut graph, "x/assign_0", x_init);
        assert_eq!(1, x.num_outputs());
        assert_eq!(x.output_type(0), DataType::Int32);
        let dims = graph
            .num_dims(Output {
                          operation: x.clone(),
                          index: 0,
                      })
            .unwrap();
        assert_eq!(dims, 2);
        let shape = graph
            .tensor_shape(Output {
                              operation: x.clone(),
                              index: 0,
                          })
            .unwrap();
        assert_eq!(shape, Shape(Some(vec![Some(3_i64), Some(3_i64)])));
    }
}
