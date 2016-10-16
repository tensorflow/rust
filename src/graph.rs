// -*- tab-width: 2 -*-

extern crate libc;
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
use std::str::Utf8Error;
use std::sync::Arc;
use super::Buffer;
use super::Code;
use super::DataType;
use super::GraphTrait;
use super::OperationTrait;
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
        gimpl: Arc::new(GraphImpl{
          inner: tf::TF_NewGraph(),
        }),
        lifetime: GraphLifetime,
      }
    }
  }

  /// Operation will only be added to graph when finish_operation() is called
  /// (assuming finish_operation() does not return an error).  graph must
  /// not be deleted until after finish_operation() is called.
  pub fn new_operation(&mut self, op_type: &str, operation_name: &str) -> std::result::Result<OperationDescription, NulError> {
    let c_op_type = try!(CString::new(op_type));
    let c_operation_name = try!(CString::new(operation_name));
    unsafe {
      Ok(OperationDescription{
        inner: tf::TF_NewOperation(self.gimpl.inner, c_op_type.as_ptr(), c_operation_name.as_ptr()),
        graph: self,
        finished: false,
      })
    }
  }

  /// Returns the operation in the graph with the given name, if it exists.
  /// If the operation does not exist, returns `Ok(None)`.
  pub fn operation_by_name(&self, operation_name: &str) -> std::result::Result<Option<Operation>, NulError> {
    let c_operation_name = try!(CString::new(operation_name));
    unsafe {
      let operation = tf::TF_GraphOperationByName(self.gimpl.inner, c_operation_name.as_ptr());
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
  pub fn operation_by_name_required(&self, operation_name: &str) -> std::result::Result<Operation, Status> {
    match try!(self.operation_by_name(operation_name)) {
      Some(operation) => Ok(operation),
      None => Err(Status::new_set(Code::Unavailable, &format!("Operation {:?} not found", operation_name)).unwrap()),
    }
  }

  /// Iterates over the operations in the graph.
  pub fn operation_iter<'a>(&'a self) -> OperationIter<'a> {
    OperationIter {
      graph: &self,
      pos: 0,
    }
  }

  /// Returns the graph definition as a protobuf.
  pub fn graph_def(&self) -> Result<Buffer<u8>> {
    let status = Status::new();
    unsafe {
      let c_buffer = tf::TF_NewBuffer();
      tf::TF_GraphToGraphDef(self.gimpl.inner, c_buffer, status.inner);
      if status.is_ok() {
        Ok(Buffer::from_c(c_buffer, true))
      } else {
        tf::TF_DeleteBuffer(c_buffer);
        Err(status)
      }
    }
  }
}

impl GraphTrait for Graph {
  fn inner(&self) -> *mut tf::TF_Graph {
    self.gimpl.inner
  }
}

////////////////////////

/// Iterator over the operations in a `Graph`.
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
      let operation = tf::TF_GraphNextOperation(self.graph.gimpl.inner, &mut self.pos as *mut size_t);
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
#[derive(Debug)]
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
    unsafe {
      CStr::from_ptr(tf::TF_OperationName(self.inner)).to_str().map(|x| x.to_string())
    }
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
    unsafe {
      tf::TF_OperationNumOutputs(self.inner) as usize
    }
  }

  /// Returns the type of a specific output.
  pub fn output_type(&self, index: usize) -> DataType {
    unsafe {
      DataType::from_c(tf::TF_OperationOutputType(tf::TF_Port{operation: self.inner, index: index as c_int}))
    }
  }

  // TODO: Figure out what this does and document it.
  #[allow(missing_docs)]
  pub fn output_list_length(&self, arg_name: &str) -> Result<usize> {
    let c_arg_name = try!(CString::new(arg_name));
    let status = Status::new();
    let length = unsafe {
      tf::TF_OperationOutputListLength(self.inner, c_arg_name.as_ptr(), status.inner)
    };
    if status.is_ok() {
      Ok(length as usize)
    } else {
      Err(status)
    }
  }

  /// Returns the number of inputs.
  pub fn num_inputs(&self) -> usize {
    unsafe {
      tf::TF_OperationNumInputs(self.inner) as usize
    }
  }

  /// Returns the type of a specific input.
  pub fn input_type(&self, index: usize) -> DataType {
    unsafe {
      DataType::from_c(tf::TF_OperationInputType(tf::TF_Port{operation: self.inner, index: index as c_int}))
    }
  }

  // TODO: Figure out what this does and document it.
  #[allow(missing_docs)]
  pub fn input_list_length(&self, arg_name: &str) -> Result<usize> {
    let c_arg_name = try!(CString::new(arg_name));
    let status = Status::new();
    let length = unsafe {
      tf::TF_OperationInputListLength(self.inner, c_arg_name.as_ptr(), status.inner)
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
      let port = tf::TF_OperationInput(tf::TF_Port{operation: self.inner, index: index as c_int});
      (Operation {
        inner: port.operation,
        gimpl: self.gimpl.clone(),
      }, port.index as usize)
    }
  }

  /// Returns the number of consumers of a specific output.
  pub fn output_num_consumers(&self, index: usize) -> usize {
    unsafe {
      tf::TF_OperationOutputNumConsumers(tf::TF_Port{operation: self.inner, index: index as c_int}) as usize
    }
  }

  /// Returns the consumers of a specific output.
  /// The index argument is the index into the current operation's output array,
  /// and the return value is a vector of the destination operation and the index into its input array.
  pub fn output_consumers(&self, index: usize) -> Vec<(Operation, usize)> {
    unsafe {
      let num_consumers = tf::TF_OperationOutputNumConsumers(tf::TF_Port{operation: self.inner, index: index as c_int});
      let mut vec = <Vec<tf::TF_Port>>::with_capacity(num_consumers as usize);
      let len = tf::TF_OperationOutputConsumers(
        tf::TF_Port{operation: self.inner, index: index as c_int},
        vec.as_mut_ptr(),
        vec.len() as c_int);
      vec.set_len(len as usize);
      vec.into_iter().map(
        |port| (Operation {inner: port.operation, gimpl: self.gimpl.clone()}, port.index as usize)
          ).collect()
    }
  }

  /// Returns the number of control inputs.
  pub fn num_control_inputs(&self) -> usize {
    unsafe {
      tf::TF_OperationNumControlInputs(self.inner) as usize
    }
  }

  /// Returns the control inputs.
  pub fn control_inputs(&self) -> Vec<Operation> {
    unsafe {
      let num_consumers = tf::TF_OperationNumControlInputs(self.inner);
      let mut vec = <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
      let len = tf::TF_OperationGetControlInputs(
        self.inner,
        vec.as_mut_ptr(),
        vec.len() as c_int);
      vec.set_len(len as usize);
      vec.into_iter().map(
        |operation| Operation {inner: operation, gimpl: self.gimpl.clone()}
          ).collect()
    }
  }

  /// Returns the number of control outputs.
  pub fn num_control_outputs(&self) -> usize {
    unsafe {
      tf::TF_OperationNumControlOutputs(self.inner) as usize
    }
  }

  /// Returns the control outputs.
  pub fn control_outputs(&self) -> Vec<Operation> {
    unsafe {
      let num_consumers = tf::TF_OperationNumControlOutputs(self.inner);
      let mut vec = <Vec<*mut tf::TF_Operation>>::with_capacity(num_consumers as usize);
      let len = tf::TF_OperationGetControlOutputs(
        self.inner,
        vec.as_mut_ptr(),
        vec.len() as c_int);
      vec.set_len(len as usize);
      vec.into_iter().map(
        |operation| Operation {inner: operation, gimpl: self.gimpl.clone()}
          ).collect()
    }
  }

  /// Returns the value of an attribute as an `AttrValue` protobuf.
  pub fn attr_value_proto(&self, attr_name: &str) -> Result<Buffer<u8>> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      let status = Status::new();
      let buffer = tf::TF_NewBuffer();
      tf::TF_OperationGetAttrValueProto(
        self.inner,
        c_attr_name.as_ptr(),
        buffer,
        status.inner);
      if !status.is_ok() {
        tf::TF_DeleteBuffer(buffer);
        Err(status)
      } else {
        Ok(Buffer::from_c(buffer, true))
      }
    }
  }

  /// Returns the operation definition as a protobuf.
  pub fn operation_def(&self) -> Result<Buffer<u8>> {
    let status = Status::new();
    unsafe {
      let c_buffer = tf::TF_NewBuffer();
      tf::TF_OperationToNodeDef(self.inner, c_buffer, status.inner);
      if status.is_ok() {
        Ok(Buffer::from_c(c_buffer, true))
      } else {
        tf::TF_DeleteBuffer(c_buffer);
        Err(status)
      }
    }
  }
}

impl OperationTrait for Operation {
  fn inner(&self) -> *mut tf::TF_Operation {
    self.inner
  }
}

////////////////////////

/// A `Port` is one end of a graph edge.
/// It holds an operation and an index into the inputs or outputs of that operation.
#[derive(Debug,Copy,Clone)]
pub struct Port<'a> {
  /// Operation the edge connects to.
  pub operation: &'a Operation,

  /// Index into either the inputs or outputs of the operation.
  pub index: c_int,
}

impl<'a> Port<'a> {
  fn to_c(&self) -> tf::TF_Port {
    tf::TF_Port {
      operation: self.operation.inner,
      index: self.index,
    }
  }
}

////////////////////////

/// An `OperationDescription` is an `Operation` in the process of being built (i.e. the builder pattern).
///
/// An `OperationDescription` is required to be finished before the graph goes out of scope,
/// so `finish()` will be called on drop if it was not already called.
#[derive(Debug)]
pub struct OperationDescription<'a> {
  inner: *mut tf::TF_OperationDescription,
  // This keeps self from outliving the Graph, which is required by the docs on TF_NewOperation.
  graph: &'a Graph,
  finished: bool,
}

impl<'a> OperationDescription<'a> {
  /// Builds the operation and adds it to the graph.
  pub fn finish(mut self) -> Result<Operation> {
    self.finished = true; // used by the drop code
    let status = Status::new();
    let operation = unsafe {
      tf::TF_FinishOperation(self.inner, status.inner)
    };
    if status.is_ok() {
      Ok(Operation{
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
    let c_device = try!(CString::new(device));
    unsafe {
      tf::TF_SetDevice(self.inner, c_device.as_ptr());
    }
    Ok(())
  }

  /// Adds an input to this operation.
  ///
  /// The index in the port is an index into the source operation's output array.
  pub fn add_input(&mut self, input: Port) {
    unsafe {
      tf::TF_AddInput(self.inner, input.to_c());
    }
  }

  /// Adds multiple inputs to this operation.
  ///
  /// The index in the ports is an index into the source operation's output array.
  pub fn add_input_list(&mut self, inputs: &[Port]) {
    let c_inputs: Vec<tf::TF_Port> = inputs.iter().map(|x| x.to_c()).collect();
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
  pub fn set_attr_string(&mut self, attr_name: &str, value: &str) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let c_value = value.as_bytes();
    unsafe {
      tf::TF_SetAttrString(
        self.inner,
        c_attr_name.as_ptr(),
        c_value.as_ptr() as *const c_void,
        c_value.len() as c_int);
    }
    Ok(())
  }

  /// Sets the value of an attribute which holds a list of strings.
  pub fn set_attr_string_list<S: AsRef<str>>(&mut self, attr_name: &str, value: &[S]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let bytes: Vec<&[u8]> = value.iter().map(|x| x.as_ref().as_bytes()).collect();
    let ptrs: Vec<*const c_void> = bytes.iter().map(|x| x.as_ptr() as *const c_void).collect();
    let lens: Vec<c_int> = bytes.iter().map(|x| x.len() as c_int).collect();
    unsafe {
      tf::TF_SetAttrStringList(
        self.inner,
        c_attr_name.as_ptr(),
        ptrs.as_ptr() as *const *const c_void,
        lens.as_ptr(),
        ptrs.len() as c_int);
    }
    Ok(())
  }

  /// Sets an int-valued attribute.
  pub fn set_attr_int(&mut self, attr_name: &str, value: i64) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrInt(self.inner, c_attr_name.as_ptr(), value);
    }
    Ok(())
  }

  /// Sets an attribute which holds an array of ints.
  pub fn set_attr_int_list(&mut self, attr_name: &str, value: &[i64]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrIntList(self.inner, c_attr_name.as_ptr(), value.as_ptr(), value.len() as i32);
    }
    Ok(())
  }

  /// Sets a float-valued attribute.
  pub fn set_attr_float(&mut self, attr_name: &str, value: f32) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
    }
    Ok(())
  }

  /// Sets an attribute which holds an array of floats.
  pub fn set_attr_float_list(&mut self, attr_name: &str, value: &[f32]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let c_value: Vec<c_float> = value.iter().map(|x| *x as c_float).collect();
    unsafe {
      tf::TF_SetAttrFloatList(self.inner, c_attr_name.as_ptr(), c_value.as_ptr(), c_value.len() as i32);
    }
    Ok(())
  }

  /// Sets a boolean-valued attribute.
  pub fn set_attr_bool(&mut self, attr_name: &str, value: bool) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrBool(self.inner, c_attr_name.as_ptr(), if value {1} else {0});
    }
    Ok(())
  }

  /// Sets an attribute which holds an array of booleans.
  pub fn set_attr_bool_list(&mut self, attr_name: &str, value: &[bool]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let c_value: Vec<c_uchar> = value.iter().map(|x| if *x {1} else {0}).collect();
    unsafe {
      tf::TF_SetAttrBoolList(self.inner, c_attr_name.as_ptr(), c_value.as_ptr(), c_value.len() as c_int);
    }
    Ok(())
  }

  /// Sets a type-valued attribute.
  pub fn set_attr_type(&mut self, attr_name: &str, value: DataType) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
    }
    Ok(())
  }

  /// Sets an attribute which holds an array of types.
  pub fn set_attr_type_list(&mut self, attr_name: &str, value: &[DataType]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
    unsafe {
      tf::TF_SetAttrTypeList(self.inner, c_attr_name.as_ptr(), c_value.as_ptr(), c_value.len() as i32);
    }
    Ok(())
  }

  /// Sets a shape-valued attribute.
  ///
  /// Entries must be at least -1, where -1 means "unknown dimension".
  pub fn set_attr_shape(&mut self, attr_name: &str, value: &[i64]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    unsafe {
      tf::TF_SetAttrShape(self.inner, c_attr_name.as_ptr(), value.as_ptr(), value.len() as i32);
    }
    Ok(())
  }

  /// Sets an attribute which holds an array of shapes.
  ///
  /// Entries must be at least -1, where -1 means "unknown dimension".
  pub fn set_attr_shape_list<T: AsRef<[i64]>>(&mut self, attr_name: &str, value: &[T]) -> std::result::Result<(), NulError> {
    let c_attr_name = try!(CString::new(attr_name));
    let ptrs: Vec<*const i64> = value.iter().map(|x| x.as_ref().as_ptr()).collect();
    let lens: Vec<c_int> = value.iter().map(|x| x.as_ref().len() as c_int).collect();
    unsafe {
      tf::TF_SetAttrShapeList(
        self.inner,
        c_attr_name.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        ptrs.len() as c_int);
    }
    Ok(())
  }

  /// Sets an attribute with a `TensorShapeProto` protobuf.
  pub fn set_attr_tensor_shape_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
    let c_attr_name = try!(CString::new(attr_name));
    let status = Status::new();
    unsafe {
      tf::TF_SetAttrTensorShapeProto(
        self.inner,
        c_attr_name.as_ptr(),
        value.as_ptr() as *const c_void,
        value.len() as c_int,
        status.inner);
    }
    status.as_result()
  }

  /// Sets an attribute with an array of `TensorShapeProto` protobufs.
  pub fn set_attr_tensor_shape_proto_list<T: AsRef<[u8]>>(&mut self, attr_name: &str, value: &[T]) -> Result<()> {
    let c_attr_name = try!(CString::new(attr_name));
    let ptrs: Vec<*const c_void> = value.iter().map(|x| x.as_ref().as_ptr() as *const c_void).collect();
    let lens: Vec<c_int> = value.iter().map(|x| x.as_ref().len() as c_int).collect();
    let status = Status::new();
    unsafe {
      tf::TF_SetAttrTensorShapeProtoList(
        self.inner,
        c_attr_name.as_ptr(),
        ptrs.as_ptr(),
        lens.as_ptr(),
        ptrs.len() as c_int,
        status.inner);
    }
    status.as_result()
  }

  /// Sets a tensor-valued attribute.
  pub fn set_attr_tensor<T: TensorType>(&mut self, attr_name: &str, value: Tensor<T>) -> Result<()> {
    let c_attr_name = try!(CString::new(attr_name));
    let status = Status::new();
    unsafe {
      tf::TF_SetAttrTensor(
        self.inner,
        c_attr_name.as_ptr(),
        value.into_ptr(),
        status.inner);
    }
    status.as_result()
  }

  /// Sets an attribute which holds an array of tensors.
  pub fn set_attr_tensor_list<T: IntoIterator<Item=Tensor<u8>>>(&mut self, attr_name: &str, value: T) -> Result<()> {
    let c_attr_name = try!(CString::new(attr_name));
    let status = Status::new();
    unsafe {
      let ptrs: Vec<*mut tf::TF_Tensor> = value.into_iter().map(|x| x.into_ptr()).collect();
      tf::TF_SetAttrTensorList(
        self.inner,
        c_attr_name.as_ptr(),
        ptrs.as_ptr(),
        ptrs.len() as c_int,
        status.inner);
    }
    status.as_result()
  }

  /// Sets an attribute with an `AttrValue` proto.
  pub fn set_attr_to_attr_value_proto(&mut self, attr_name: &str, value: &[u8]) -> Result<()> {
    let c_attr_name = try!(CString::new(attr_name));
    let status = Status::new();
    unsafe {
      tf::TF_SetAttrToAttrValueProto(
        self.inner,
        c_attr_name.as_ptr(),
        value.as_ptr() as *const c_void,
        value.len() as size_t,
        status.inner);
    }
    status.as_result()
  }
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

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;
  use super::super::DataType;

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
      nd.set_attr_shape("shape", &vec![]).unwrap();
      nd.finish().unwrap()
    };
    let mut nd2 = g.new_operation("Variable", "foo2").unwrap();
    nd2.set_attr_type("dtype", DataType::Float).unwrap();
    nd2.set_attr_shape("shape", &vec![]).unwrap();
    let operation2 = nd2.finish().unwrap();
    assert_eq!("foo", operation.name().unwrap());
    assert_eq!("foo2", operation2.name().unwrap());
  }
}
