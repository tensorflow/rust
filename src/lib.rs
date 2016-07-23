// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
//! This crate provides Rust bindings for the [TensorFlow](https://www.tensorflow.org) machine learning library.

extern crate libc;
extern crate tensorflow_sys as tf;

use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::error::Error;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt;
use std::marker;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Drop;

mod buffer;
pub use buffer::Buffer;

pub mod expr;

////////////////////////

fn check_not_null<T>(p: *mut T) -> *mut T {
  assert!(!p.is_null());
  p
}

////////////////////////

macro_rules! impl_new {
  ($name: ident, $call:ident, $doc:expr) => {
    impl $name {
      #[doc = $doc]
      pub fn new() -> Self {
        unsafe {
          $name {
            inner: check_not_null(tf::$call()),
          }
        }
      }
    }
  }
}

////////////////////////

macro_rules! impl_drop {
  ($name: ident, $call:ident) => {
    impl Drop for $name {
      fn drop(&mut self) {
        unsafe {
          tf::$call(self.inner);
        }
      }
    }
  }
}

////////////////////////

/// Will panic if `msg` contains an embedded 0 byte.
macro_rules! invalid_arg {
  ($fmt:expr) => {
    Status::new_set(Code::InvalidArgument, $fmt).unwrap()
  };
  ($fmt:expr, $($arg:tt)*) => ({
    let msg = format!($fmt, $($arg)*);
    Status::new_set(Code::InvalidArgument, &msg).unwrap()
  });
}

////////////////////////

macro_rules! c_enum {
  ($doc:expr, $enum_name:ident { $($name:ident = $num:expr),* }) => {
    #[doc = $doc]
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug)]
    pub enum $enum_name {
      UnrecognizedEnumValue(c_uint),
      $($name),*
    }

    impl $enum_name {
      #[allow(dead_code)]
      fn from_int(value: c_uint) -> $enum_name {
        match value {
          $($num => $enum_name::$name,)*
          c => $enum_name::UnrecognizedEnumValue(c),
        }
      }

      #[allow(dead_code)]
      fn to_int(&self) -> c_uint {
        match self {
          &$enum_name::UnrecognizedEnumValue(c) => c,
          $(&$enum_name::$name => $num),*
        }
      }
    }

    impl ::std::fmt::Display for $enum_name {
      fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match self {
          $(&$enum_name::$name => f.write_str(stringify!($name)),)*
          &$enum_name::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
      }
    }
  };
  ($doc:expr, $enum_name:ident { $($name:ident = $num:expr,)* }) => {
    c_enum!($doc, $enum_name { $($name = $num),* });
  }
}

////////////////////////

c_enum!("Error values that can be returned.", Code {
  Ok = 0,
  Cancelled = 1,
  Unknown = 2,
  InvalidArgument = 3,
  DeadlineExceeded = 4,
  NotFound = 5,
  AlreadyExists = 6,
  PermissionDenied = 7,
  ResourceExhausted = 8,
  FailedPrecondition = 9,
  Aborted = 10,
  OutOfRange = 11,
  Unimplemented = 12,
  Internal = 13,
  Unavailable = 14,
  DataLoss = 15,
  Unauthenticated = 16,
});

////////////////////////

c_enum!("Type of a single tensor element.", DataType {
  Float = 1,
  Double = 2,
  Int32 = 3,
  UInt8 = 4,
  Int16 = 5,
  Int8 = 6,
  String = 7,
  Complex = 8,
  Int64 = 9,
  Bool = 10,
  QInt8 = 11,
  QUInt8 = 12,
  QInt32 = 13,
  BFloat16 = 14,
  QInt16 = 15,
  QUInt16 = 16,
});

////////////////////////

/// Holds error information.  It either has an OK code, or else an error code with an associated error message.
pub struct Status {
  inner: *mut tf::TF_Status,
}

impl_new!(Status, TF_NewStatus, "Creates a status with `Code::Ok` and no message.");
impl_drop!(Status, TF_DeleteStatus);

impl Status {
  /// Creates a status and sets its code and message.
  pub fn new_set(code: Code, msg: &str) -> std::result::Result<Status, NulError> {
    let mut status = Status::new();
    try!(status.set(code, msg));
    Ok(status)
  }

  /// Returns the status's code.
  pub fn code(&self) -> Code {
    unsafe {
      Code::from_int(tf::TF_GetCode(self.inner) as u32)
    }
  }

  /// Returns true if the status's code is `Code::Ok`.
  pub fn is_ok(&self) -> bool {
    self.code() == Code::Ok
  }

  fn as_result(self) -> Result<()> {
    if self.is_ok() {
      Ok(())
    } else {
      Err(self)
    }
  }

  /// Sets the code and message.
  pub fn set(&mut self, code: Code, msg: &str) -> std::result::Result<(), NulError> {
    let message = try!(CString::new(msg));
    unsafe {
      tf::TF_SetStatus(self.inner, mem::transmute(code.to_int()), message.as_ptr());
    }
    Ok(())
  }
}

impl Display for Status {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    unsafe {
      try!(write!(f, "{}: ", self.code()));
      let msg = match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      };
      f.write_str(msg)
    }
  }
}

impl Debug for Status {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    unsafe {
      try!(write!(f, "{{inner:{:?}, ", self.inner));
      try!(write!(f, "{}: ", self.code()));
      let msg = match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      };
      try!(f.write_str(msg));
      try!(write!(f, "}}"));
      Ok(())
    }
  }
}

impl From<NulError> for Status {
  fn from(_e: NulError) -> Self {
    invalid_arg!("String contained NUL byte")
  }
}

impl Error for Status {
  fn description(&self) -> &str {
    unsafe {
      match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      }
    }
  }

  fn cause(&self) -> Option<&Error> {
    None
  }
}

////////////////////////

/// Options that can be passed during session creation.
pub struct SessionOptions {
  inner: *mut tf::TF_SessionOptions,
}

impl SessionOptions {
  /// Set the target.
  ///
  /// `target` can be empty, a single entry, or a comma separated list of entries.
  /// Each entry is in one of the following formats :
  ///
  /// - "local"
  /// - ip:port
  /// - host:port
  pub fn set_target(&mut self, target: &str) -> std::result::Result<(), NulError> {
    let cstr = try!(CString::new(target));
    unsafe {
      tf::TF_SetTarget(self.inner, cstr.as_ptr());
    }
    Ok(())
  }

  /// Set the config.
  ///
  /// `config` should be a serialized brain.ConfigProto proto.
  /// Returns an error if config was not parsed successfully as a ConfigProto.
  pub fn set_config(&mut self, config: &[u8]) -> Result<()> {
    let status = Status::new();
    unsafe {
      tf::TF_SetConfig(self.inner, config.as_ptr() as *const _, config.len(), status.inner);
    }
    if status.is_ok() {
      Ok(())
    } else {
      Err(status)
    }
  }
}

impl_new!(SessionOptions, TF_NewSessionOptions, "Creates a blank set of options.");
impl_drop!(SessionOptions, TF_DeleteSessionOptions);

////////////////////////

/// Manages a single graph and execution.
pub struct Session {
  inner: *mut tf::TF_Session,
}

impl Session {
  /// Creates a session.
  pub fn new(options: &SessionOptions) -> Result<Self> {
    let status = Status::new();
    let inner = unsafe { tf::TF_NewSession(options.inner, status.inner) };
    if inner.is_null() {
      Err(status)
    } else {
      Ok(Session {
        inner: inner,
      })
    }
  }

  /// Closes the session.
  pub fn close(&mut self) -> Result<()> {
    let status = Status::new();
    unsafe {
      tf::TF_CloseSession(self.inner, status.inner);
    }
    status.as_result()
  }

  /// Treat `proto` as a serialized `GraphDef` and add the nodes in that `GraphDef` to the graph for the session.
  pub fn extend_graph(&mut self, proto: &[u8]) -> Result<()> {
    let status = Status::new();
    unsafe {
      tf::TF_ExtendGraph(self.inner, proto.as_ptr() as *const _, proto.len(), status.inner);
    }
    status.as_result()
  }

  /// Runs the graph, feeding the inputs and then fetching the outputs requested in the step.
  pub fn run(&mut self, step: &mut Step) -> Result<()> {
    // Copy the input tensors because TF_Run consumes them.
    let mut input_tensors = Vec::with_capacity(step.input_tensors.len());
    for &input_tensor in &step.input_tensors {
      let input_tensor = input_tensor as *const tf::TF_Tensor;
      unsafe {
        let mut dims = Vec::with_capacity(tf::TF_NumDims(input_tensor) as usize);
        for i in 0..dims.capacity() {
          dims.push(tf::TF_Dim(input_tensor, i as c_int));
        }
        input_tensors.push(tf::TF_NewTensor(tf::TF_TensorType(input_tensor),
                                            dims.as_ptr(),
                                            dims.len() as c_int,
                                            tf::TF_TensorData(input_tensor),
                                            tf::TF_TensorByteSize(input_tensor),
                                            Some(noop_deallocator),
                                            std::ptr::null_mut()));
      }
    }

    // In case we're running it a second time and not all outputs were taken out.
    step.drop_output_tensors();

    let status = Status::new();
    unsafe {
      tf::TF_Run(
        self.inner,
        std::ptr::null(),
        step.input_name_ptrs.as_mut_ptr(),
        input_tensors.as_mut_ptr(),
        input_tensors.len() as c_int,
        step.output_name_ptrs.as_mut_ptr(),
        step.output_tensors.as_mut_ptr(),
        step.output_tensors.len() as c_int,
        step.target_name_ptrs.as_mut_ptr(),
        step.target_name_ptrs.len() as c_int,
        std::ptr::null_mut(),
        status.inner);
    };
    status.as_result()
  }
}

impl Drop for Session {
  fn drop(&mut self) {
    let status = Status::new();
    unsafe {
      tf::TF_DeleteSession(self.inner, status.inner);
    }
    // TODO: What do we do with the status?
  }
}

/// Manages the inputs and outputs for a single execution of a graph.
///
/// Typical usage involves creating an instance of this struct,
/// adding some inputs to it, requesting some outputs, passing it to `Session::run`
/// and then taking the outputs out of it.
pub struct Step<'l> {
  input_name_ptrs: Vec<*const c_char>,
  input_name_c_strings: Vec<CString>,
  input_tensors: Vec<*mut tf::TF_Tensor>,

  output_name_ptrs: Vec<*const c_char>,
  output_name_c_strings: Vec<CString>,
  output_tensors: Vec<*mut tf::TF_Tensor>,

  target_name_ptrs: Vec<*const c_char>,
  target_name_c_strings: Vec<CString>,

  phantom: marker::PhantomData<&'l ()>,
}

impl<'l> Step<'l> {
  /// Creates a Step.
  pub fn new() -> Self {
    Step {
      input_name_ptrs: vec![],
      input_name_c_strings: vec![],
      input_tensors: vec![],

      output_name_ptrs: vec![],
      output_name_c_strings: vec![],
      output_tensors: vec![],

      target_name_ptrs: vec![],
      target_name_c_strings: vec![],

      phantom: marker::PhantomData,
    }
  }

  /// Adds an input to be fed to the graph.
  pub fn add_input<T>(&mut self, name: &str, tensor: &'l Tensor<T>) -> std::result::Result<(), NulError> {
    let c_string = try!(CString::new(name));
    self.input_name_ptrs.push(c_string.as_ptr());
    self.input_name_c_strings.push(c_string);
    self.input_tensors.push(tensor.inner);
    Ok(())
  }

  /// Requests that an output is fetched from the graph after running this step.
  /// Returns an index that you can then use to fetch this output from the step after running it.
  pub fn request_output(&mut self, name: &str) -> std::result::Result<usize, NulError> {
    let c_string = try!(CString::new(name));
    self.output_name_ptrs.push(c_string.as_ptr());
    self.output_name_c_strings.push(c_string);
    self.output_tensors.push(std::ptr::null_mut());
    Ok(self.output_tensors.len() - 1)
  }

  /// Extracts a tensor output given an index. A given index can only be extracted once per `Session::run`.
  /// Returns an error if output_idx is out of range, output is unavailable or the
  /// requested type does not match the type of the actual tensor.
  pub fn take_output<T: TensorType>(&mut self, output_idx: usize) -> Result<Tensor<T>> {
    if output_idx >= self.output_tensors.len() {
      return Err(Status::new_set(Code::OutOfRange,
        &format!("Requested output index is out of range: {} vs {}",
          output_idx,
          self.output_tensors.len())).unwrap());
    }
    if self.output_tensors[output_idx].is_null() {
      return Err(Status::new_set(Code::Unavailable,
        "Output not available. Either it was already taken, or this step \
        has not been sucessfully run yet.").unwrap());
    }
    let actual_data_type = self.get_output_data_type(output_idx).unwrap();
    if actual_data_type != T::data_type() {
      return Err(invalid_arg!(
        "Requested tensor type does not match actual tensor type: {} vs {}",
        actual_data_type,
        T::data_type()));
    }
    let tensor = unsafe {
      Tensor::from_tf_tensor(self.output_tensors[output_idx]).unwrap()
    };
    self.output_tensors[output_idx] = std::ptr::null_mut();
    Ok(tensor)
  }

  /// Adds a target node to be executed when running the graph.
  pub fn add_target(&mut self, name: &str) -> std::result::Result<(), NulError> {
    let c_string = try!(CString::new(name));
    self.target_name_ptrs.push(c_string.as_ptr());
    self.target_name_c_strings.push(c_string);
    Ok(())
  }

  /// Retuns the type of the tensor given an index.
  /// Returns `None` if the index is out of range or the output is not yet available.
  pub fn get_output_data_type(&self, output_idx: usize) -> Option<DataType> {
    if output_idx >= self.output_tensors.len() {
      return None;
    }
    if self.output_tensors[output_idx].is_null() {
      return None;
    }
    unsafe {
      Some(DataType::from_int(mem::transmute(tf::TF_TensorType(self.output_tensors[output_idx]))))
    }
  }

  fn drop_output_tensors(&mut self) {
    for &tensor in &self.output_tensors {
      // TODO: Is TF_DeleteTensor NULL safe?
      if !tensor.is_null() {
        unsafe {
          tf::TF_DeleteTensor(tensor);
        }
      }
    }
  }
}

impl<'l> Drop for Step<'l> {
  fn drop(&mut self) {
    self.drop_output_tensors();
  }
}

////////////////////////

/// Convenience type for `Result` with `Status` as the error type.
pub type Result<T> = std::result::Result<T, Status>;

////////////////////////

/// A Rust type that maps to a `DataType`.
pub trait TensorType: Default + Clone + Display + Debug + 'static {
  // TODO: Use associated constants when/if available
  /// Returns the DataType that corresponds to this type.
  fn data_type() -> DataType;
}

macro_rules! tensor_type {
  ($rust_type:ident, $tensor_type:ident) => {
    impl TensorType for $rust_type {
      fn data_type() -> DataType {
        DataType::$tensor_type
      }
    }
  }
}

tensor_type!(f32, Float);
tensor_type!(f64, Double);
tensor_type!(i32, Int32);
tensor_type!(u8, UInt8);
tensor_type!(i16, Int16);
tensor_type!(i8, Int8);
// TODO: provide type for String
// TODO: provide type for Complex. Pending impl of Default: https://github.com/rust-num/num/issues/198
tensor_type!(i64, Int64);
tensor_type!(bool, Bool);
// TODO: provide type for BFloat16

macro_rules! q_type {
  ($rust_type:ident, $q_type:ident) => {
    #[derive(Clone,Default,Debug,Eq,PartialEq,Ord,PartialOrd)]
    pub struct $q_type($rust_type);

    impl Display for $q_type {
      fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        <$rust_type as Display>::fmt(&self.0, f)
      }
    }

    impl From<$rust_type> for $q_type {
      fn from(x: $rust_type) -> Self {
        $q_type(x)
      }
    }

    tensor_type!($q_type, $q_type);
  }
}

q_type!(i8, QInt8);
q_type!(u8, QUInt8);
q_type!(i16, QInt16);
q_type!(u16, QUInt16);
q_type!(i32, QInt32);

////////////////////////

/// Holds a multi-dimensional array of elements of a single data type.
///
/// For all types other than strings, the data buffer stores elements
/// in row major order.  E.g. if data is treated as a vector of `T`:
///
/// ```text
///   element 0:   index (0, ..., 0)
///   element 1:   index (0, ..., 1)
///   ...
/// ```
///
/// The layout for strings is currently undefined.
pub struct Tensor<T> {
  inner: *mut tf::TF_Tensor,
  data: Buffer<T>,
  dims: Vec<u64>,
}

unsafe extern "C" fn noop_deallocator(_: *mut c_void, _: size_t, _: *mut c_void) -> () {}

// TODO: Replace with Iterator::product once that's stable
fn product(values: &[u64]) -> u64 {
  let mut product = 1;
  for v in values.iter() {
    product *= *v;
  }
  product
}

impl<T: TensorType> Tensor<T> {
  /// Creates a new tensor.
  ///
  /// The data is initialized to zeros.
  pub fn new(dims: &[u64]) -> Self {
    let total = product(dims);
    let data = <Buffer<T>>::new(total as usize);
    // Guaranteed safe to unwrap, because the only way for it to fail is for the
    // length of the buffer not to match the dimensions, and we created it with
    // exactly the right size.
    Self::new_with_buffer(dims, data).unwrap()
  }

  /// Creates a new tensor from existing data.
  pub fn new_with_buffer(dims: &[u64], data: Buffer<T>) -> Result<Self> {
    let total = product(dims);
    if total != data.len() as u64 {
      return Err(invalid_arg!("Dimensions {:?} do not match buffer length {}", dims, data.len()));
    }
    let inner = unsafe {
      tf::TF_NewTensor(mem::transmute(T::data_type().to_int()),
                       dims.as_ptr() as *const _,
                       dims.len() as c_int,
                       data.as_ptr() as *mut _,
                       data.len(),
                       Some(noop_deallocator),
                       std::ptr::null_mut())
    };
    Ok(Tensor {
      inner: inner,
      data: data,
      dims: Vec::from(dims),
    })
  }

  /// Returns the tensor's data.
  pub fn data(&self) -> &Buffer<T> {
    &self.data
  }

  /// Returns the tensor's data.
  pub fn data_mut(&mut self) -> &mut Buffer<T> {
    &mut self.data
  }

  /// Returns the tensor's dimensions.
  pub fn dims(&self) -> &[u64] {
    &self.dims
  }

  // Wraps a TF_Tensor. Returns None if types don't match.
  unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self> {
    if DataType::from_int(mem::transmute(tf::TF_TensorType(tensor))) != T::data_type() {
      return None;
    }
    let mut dims = Vec::with_capacity(tf::TF_NumDims(tensor) as usize);
    for i in 0..dims.capacity() {
      dims.push(tf::TF_Dim(tensor, i as c_int) as u64);
    }
    let data = Buffer::from_ptr(tf::TF_TensorData(tensor) as *mut _, product(&dims) as usize);
    Some(Tensor {
      inner: tensor,
      data: data,
      dims: dims
    })
  }
}

impl<T> Drop for Tensor<T> {
  fn drop(&mut self) {
    unsafe {
      tf::TF_DeleteTensor(self.inner);
    }
  }
}

impl<T> Deref for Tensor<T> {
  type Target = Buffer<T>;

  #[inline]
  fn deref(&self) -> &Buffer<T> {
    &self.data
  }
}

impl<T> DerefMut for Tensor<T> {
  #[inline]
  fn deref_mut<'a>(&'a mut self) -> &'a mut Buffer<T> {
    &mut self.data
  }
}

////////////////////////

/// Dynamically loaded plugins.
/// The C API doesn't provide a way to unload libraries, so nothing happens when this goes out of scope.
pub struct Library {
  inner: *mut tf::TF_Library,
}

impl Library {
  /// Loads a library.
  pub fn load(library_filename: &str) -> Result<Self> {
    let c_filename = try!(CString::new(library_filename));
    let status = Status::new();
    let inner = unsafe { tf::TF_LoadLibrary(c_filename.as_ptr(), status.inner) };
    if inner.is_null() {
      Err(status)
    } else {
      Ok(Library {
        inner: inner,
      })
    }
  }

  // TODO: Implement TF_GetOpList once we can deserialize protos.
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;

  fn create_session() -> Session {
    let options = SessionOptions::new();
    match Session::new(&options) {
      Ok(session) => session,
      Err(status) => panic!("Creating session failed with status: {}", status),
    }
  }

  #[test]
  fn smoke() {
    create_session();
  }

  #[test]
  fn test_close() {
    let status = create_session().close();
    assert!(status.is_ok());
  }

  #[test]
  fn test_tensor() {
    let mut tensor = <Tensor<f32>>::new(&[2, 3]);
    assert_eq!(tensor.data().len(), 6);
    tensor.data_mut()[0] = 1.0;
  }

  #[test]
  fn test_set_target() {
    let mut options = SessionOptions::new();
    options.set_target("local").unwrap();
  }

  #[test]
  fn test_set_config() {
    let mut options = SessionOptions::new();
    // An empty array is a valid proto, since all fields are optional.
    options.set_config(&vec![]).unwrap();
  }

  #[test]
  fn test_extend_graph() {
    let mut session = create_session();
    // An empty array is a valid proto, since all fields are optional.
    let status = session.extend_graph(&vec![]);
    assert!(status.is_ok());
  }

  #[test]
  fn test_run() {
    // Graph is just y = 2 * x
    let graph_proto = vec![
      0x0a, 0x2a, 0x0a, 0x01, 0x78, 0x12, 0x0b, 0x50, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f, 0x6c, 0x64,
      0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12, 0x02, 0x30, 0x01, 0x2a,
      0x0b, 0x0a, 0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x02, 0x3a, 0x00, 0x0a, 0x30, 0x0a, 0x03,
      0x79, 0x2f, 0x79, 0x12, 0x05, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74,
      0x79, 0x70, 0x65, 0x12, 0x02, 0x30, 0x01, 0x2a, 0x15, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65,
      0x12, 0x0c, 0x42, 0x0a, 0x08, 0x01, 0x12, 0x00, 0x2a, 0x04, 0x00, 0x00, 0x00, 0x40, 0x0a, 0x19,
      0x0a, 0x01, 0x79, 0x12, 0x03, 0x4d, 0x75, 0x6c, 0x1a, 0x01, 0x78, 0x1a, 0x03, 0x79, 0x2f, 0x79,
      0x2a, 0x07, 0x0a, 0x01, 0x54, 0x12, 0x02, 0x30, 0x01
    ];
    let mut session = create_session();
    let status = session.extend_graph(&graph_proto);
    assert!(status.is_ok());
    let mut x = <Tensor<f32>>::new(&[2]);
    x.data_mut()[0] = 2.0;
    x.data_mut()[1] = 3.0;
    let mut step = Step::new();
    step.add_input("x:0", &x).unwrap();
    let output_ix = step.request_output("y:0").unwrap();
    session.run(&mut step).unwrap();
    let output_tensor = step.take_output::<f32>(output_ix).unwrap();
    let data = output_tensor.data();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0], 4.0);
    assert_eq!(data[1], 6.0);
  }
}
