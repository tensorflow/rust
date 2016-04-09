// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
//! This crate provides Rust bindings for the [TensorFlow](https://www.tensorflow.org) machine learning library.
#![cfg(feature = "tensorflow_unstable")]

extern crate libc;
extern crate libtensorflow_sys;

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem;
use std::ops::Drop;
use std::os::raw;

use libtensorflow_sys as tf;

mod buffer;
pub use buffer::Buffer;

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

macro_rules! c_enum {
  ($doc:expr, $enum_name:ident { $($name:ident = $num:expr),* }) => {
    #[doc = $doc]
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug)]
    pub enum $enum_name {
      UnrecognizedEnumValue(raw::c_uint),
      $($name),*
    }

    impl $enum_name {
      #[allow(dead_code)]
      fn from_int(value: raw::c_uint) -> $enum_name {
        match value {
          $($num => $enum_name::$name,)*
          c => $enum_name::UnrecognizedEnumValue(c),
        }
      }

      #[allow(dead_code)]
      fn to_int(&self) -> raw::c_uint {
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

  /// Sets the code and message.
  pub fn set(&mut self, code: Code, msg: &str) -> std::result::Result<(), NulError> {
    let message = try!(CString::new(msg)).as_ptr();
    unsafe {
      tf::TF_SetStatus(self.inner, mem::transmute(code.to_int()), message);
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
      tf::TF_SetConfig(self.inner, config.as_ptr() as *const raw::c_void, config.len(), status.inner);
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
  pub fn close(&mut self) -> Status {
    let status = Status::new();
    unsafe {
      tf::TF_CloseSession(self.inner, status.inner);
    }
    status
  }

  /// Treat `proto` as a serialized `GraphDef` and add the nodes in that `GraphDef` to the graph for the session.
  pub fn extend_graph(&mut self, proto: &[u8]) -> Status {
    let status = Status::new();
    unsafe {
      tf::TF_ExtendGraph(self.inner, proto.as_ptr() as *const raw::c_void, proto.len(), status.inner);
    }
    status
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

////////////////////////

/// Convenience type for `Result` with `Status` as the error type.
pub type Result<T> = std::result::Result<T, Status>;

////////////////////////

/// A Rust type that maps to a `DataType`.
pub trait TensorType: Default + Clone {
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
// TODO: provide type for Complex
tensor_type!(i64, Int64);
tensor_type!(bool, Bool);
// TODO: provide type for QInt8
// TODO: provide type for QUInt8
// TODO: provide type for QInt32
// TODO: provide type for BFloat16
// TODO: provide type for QInt16
// TODO: provide type for QUInt16

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

unsafe extern "C" fn noop_deallocator(_data: *mut raw::c_void,
                               _len: ::libc::size_t,
                               _arg: *mut raw::c_void)-> () {
}

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
  pub fn new_with_buffer(dims: &[u64], data: Buffer<T>) -> Option<Self> {
    let total = product(dims);
    if total != data.len() as u64 {
      return None
    }
    let inner = unsafe {
      tf::TF_NewTensor(mem::transmute(T::data_type().to_int()),
                       dims.as_ptr() as *mut i64,
                       dims.len() as i32,
                       data.as_ptr() as *mut raw::c_void,
                       data.len(),
                       Some(noop_deallocator),
                       std::ptr::null_mut())
    };
    let mut dims_vec = Vec::new();
    // TODO: Use extend_from_slice once we're on Rust 1.6
    dims_vec.extend(dims.iter());
    Some(Tensor {
      inner: inner,
      data: data,
      dims: dims_vec,
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
}

impl<T> Drop for Tensor<T> {
  fn drop(&mut self) {
    unsafe {
      tf::TF_DeleteTensor(self.inner);
    }
  }
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
}
