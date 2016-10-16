// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
//! This crate provides Rust bindings for the [TensorFlow](https://www.tensorflow.org) machine learning library.

#![warn(missing_copy_implementations,
        missing_debug_implementations,
        missing_docs,
        trivial_casts,
        trivial_numeric_casts,
        unused_extern_crates,
        unused_import_braces,
        unused_qualifications)]

extern crate libc;
extern crate num_complex;
extern crate tensorflow_sys as tf;

use libc::{c_char, c_int, c_uint, c_void, size_t};
use num_complex::Complex;
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

mod buffer;
pub use buffer::Buffer;

mod graph;
pub use graph::*;

mod session;
pub use session::*;

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

// We would like to use the pattern:
//   ($doc:expr, $c_name:ident, $enum_name:ident { $( $(#[$attr:meta])* $name:ident = $num:expr),* })
// so the enum variants would look like:
//   /// Denotes a foo.
//   Foo = 1,
// but the compiler complains:
//   error: local ambiguity: multiple parsing options: built-in NTs ident ('name') or 1 other option.
// This is https://github.com/rust-lang/rust/issues/24189. Rather than make our
// macro rules inscrutably convoluted, we'll just make our grammar slightly
// noisier and insert a 'value' token before the variant name.
macro_rules! c_enum {
  ($doc:expr, $c_name:ident, $enum_name:ident { $( $(#[$attr:meta])* value $name:ident = $num:expr),* }) => {
    #[doc = $doc]
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug,Copy,Clone)]
    pub enum $enum_name {
      /// Represents an unrecognized value.
      ///
      /// This allows such values to come from the C API and be sent back to the
      /// C API without loss in case new values are added in the future.
      UnrecognizedEnumValue(c_uint),
      $($(#[$attr])* $name),*
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

      #[allow(dead_code)]
      fn to_c(&self) -> tf::$c_name {
        unsafe {
          ::std::mem::transmute(self.to_int())
        }
      }

      #[allow(dead_code)]
      fn from_c(value: tf::$c_name) -> $enum_name {
        $enum_name::from_int(value as c_uint)
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
  ($doc:expr, $c_name:ident, $enum_name:ident { $( $(#[$attr:meta])* value $name:ident = $num:expr,)* }) => {
    c_enum!($doc, $c_name, $enum_name { $( $(#[$attr])* value $name = $num),* });
  }
}

////////////////////////

c_enum!("Error values that can be returned.", TF_Code, Code {
  /// Not an error; returned on success.
  value Ok = 0,

  /// The operation was cancelled (typically by the caller).
  value Cancelled = 1,

  /// Unknown error.  An example of where this error may be returned is
  /// if a Status value received from another address space belongs to
  /// an error-space that is not known in this address space.  Also
  /// errors raised by APIs that do not return enough error information
  /// may be converted to this error.
  value Unknown = 2,

  /// Client specified an invalid argument.  Note that this differs
  /// from FAILED_PRECONDITION.  INVALID_ARGUMENT indicates arguments
  /// that are problematic regardless of the state of the system
  /// (e.g., a malformed file name).
  value InvalidArgument = 3,

  /// Deadline expired before operation could complete.  For operations
  /// that change the state of the system, this error may be returned
  /// even if the operation has completed successfully.  For example, a
  /// successful response from a server could have been delayed long
  /// enough for the deadline to expire.
  value DeadlineExceeded = 4,

  /// Some requested entity (e.g., file or directory) was not found.
  /// For privacy reasons, this code *may* be returned when the client
  /// does not have the access right to the entity.
  value NotFound = 5,

  /// Some entity that we attempted to create (e.g., file or directory)
  /// already exists.
  value AlreadyExists = 6,

  /// The caller does not have permission to execute the specified
  /// operation.  PERMISSION_DENIED must not be used for rejections
  /// caused by exhausting some resource (use RESOURCE_EXHAUSTED
  /// instead for those errors).  PERMISSION_DENIED must not be
  /// used if the caller can not be identified (use UNAUTHENTICATED
  /// instead for those errors).
  value PermissionDenied = 7,

  /// Some resource has been exhausted, perhaps a per-user quota, or
  /// perhaps the entire file system is out of space.
  value ResourceExhausted = 8,

  /// Operation was rejected because the system is not in a state
  /// required for the operation's execution.  For example, directory
  /// to be deleted may be non-empty, an rmdir operation is applied to
  /// a non-directory, etc.
  ///
  /// A litmus test that may help a service implementor in deciding
  /// between FAILED_PRECONDITION, ABORTED, and UNAVAILABLE:
  ///  (a) Use UNAVAILABLE if the client can retry just the failing call.
  ///  (b) Use ABORTED if the client should retry at a higher-level
  ///      (e.g., restarting a read-modify-write sequence).
  ///  (c) Use FAILED_PRECONDITION if the client should not retry until
  ///      the system state has been explicitly fixed.  E.g., if an "rmdir"
  ///      fails because the directory is non-empty, FAILED_PRECONDITION
  ///      should be returned since the client should not retry unless
  ///      they have first fixed up the directory by deleting files from it.
  ///  (d) Use FAILED_PRECONDITION if the client performs conditional
  ///      REST Get/Update/Delete on a resource and the resource on the
  ///      server does not match the condition. E.g., conflicting
  ///      read-modify-write on the same resource.
  value FailedPrecondition = 9,

  /// The operation was aborted, typically due to a concurrency issue
  /// like sequencer check failures, transaction aborts, etc.
  ///
  /// See litmus test above for deciding between FAILED_PRECONDITION,
  /// ABORTED, and UNAVAILABLE.
  value Aborted = 10,

  /// Operation tried to iterate past the valid input range.  E.g., seeking or
  /// reading past end of file.
  ///
  /// Unlike INVALID_ARGUMENT, this error indicates a problem that may
  /// be fixed if the system state changes. For example, a 32-bit file
  /// system will generate INVALID_ARGUMENT if asked to read at an
  /// offset that is not in the range [0,2^32-1], but it will generate
  /// OUT_OF_RANGE if asked to read from an offset past the current
  /// file size.
  ///
  /// There is a fair bit of overlap between FAILED_PRECONDITION and
  /// OUT_OF_RANGE.  We recommend using OUT_OF_RANGE (the more specific
  /// error) when it applies so that callers who are iterating through
  /// a space can easily look for an OUT_OF_RANGE error to detect when
  /// they are done.
  value OutOfRange = 11,

  /// Operation is not implemented or not supported/enabled in this service.
  value Unimplemented = 12,

  /// Internal errors.  Means some invariants expected by underlying
  /// system has been broken.  If you see one of these errors,
  /// something is very broken.
  value Internal = 13,

  /// The service is currently unavailable.  This is a most likely a
  /// transient condition and may be corrected by retrying with
  /// a backoff.
  ///
  /// See litmus test above for deciding between FAILED_PRECONDITION,
  /// ABORTED, and UNAVAILABLE.
  value Unavailable = 14,

  /// Unrecoverable data loss or corruption.
  value DataLoss = 15,

  /// The request does not have valid authentication credentials for the
  /// operation.
  value Unauthenticated = 16,
});

////////////////////////

c_enum!("Type of a single tensor element.", TF_DataType, DataType {
  /// 32-bit floating point.
  value Float = 1,

  /// 64-bit floating point.
  value Double = 2,

  /// 32-bit signed integer.
  value Int32 = 3,

  /// 8-bit unsigned integer.
  value UInt8 = 4,

  /// 16-bit signed integer.
  value Int16 = 5,

  /// 8-bit signed integer.
  value Int8 = 6,

  /// String.
  value String = 7,

  /// Complex number composed of two 32-bit floats.
  value Complex64 = 8,

  /// 64-bit signed integer.
  value Int64 = 9,

  /// Boolean.
  value Bool = 10,

  /// Quantized 8-bit signed integer.
  value QInt8 = 11,

  /// Quantized 8-bit unsigned integer.
  value QUInt8 = 12,

  /// Quantized 32-bit signed integer.
  value QInt32 = 13,

  /// Float32 truncated to 16 bits.  Only for cast ops.
  value BFloat16 = 14,

  /// Quantized 16-bit signed integer.
  value QInt16 = 15,

  /// Quantized 16-bit unsigned integer.
  value QUInt16 = 16,

  /// 16-bit unsigned integer.
  value UInt16 = 17,

  /// Complex number composed of two 64-bit floats.
  value Complex128 = 18,

  /// 16-bit floating point.
  value Half = 19,
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
      tf::TF_SetStatus(self.inner, code.to_c(), message.as_ptr());
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
#[derive(Debug)]
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
#[deprecated(note="Use SessionWithGraph instead.")]
#[allow(deprecated)]
#[derive(Debug)]
pub struct Session {
  inner: *mut tf::TF_Session,
}

#[allow(deprecated)]
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

  /// Treat `proto` as a serialized `GraphDef` and add the operations in that `GraphDef` to the graph for the session.
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

#[allow(deprecated)]
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
#[deprecated(note="Use StepWithGraph instead.")]
#[allow(deprecated)]
#[derive(Debug)]
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

#[allow(deprecated)]
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
  pub fn add_input<T: TensorType>(&mut self, name: &str, tensor: &'l Tensor<T>) -> std::result::Result<(), NulError> {
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
        has not been successfully run yet.").unwrap());
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

  /// Adds a target operation to be executed when running the graph.
  pub fn add_target(&mut self, name: &str) -> std::result::Result<(), NulError> {
    let c_string = try!(CString::new(name));
    self.target_name_ptrs.push(c_string.as_ptr());
    self.target_name_c_strings.push(c_string);
    Ok(())
  }

  /// Retuns the type of the tensor given an index.
  /// Returns `None` if the index is out of range or the output is not yet available.
  pub fn get_output_data_type(&self, output_idx: usize) -> Option<DataType> {
    // TODO: rename to output_data_type()
    if output_idx >= self.output_tensors.len() {
      return None;
    }
    if self.output_tensors[output_idx].is_null() {
      return None;
    }
    unsafe {
      Some(DataType::from_c(tf::TF_TensorType(self.output_tensors[output_idx])))
    }
  }

  fn drop_output_tensors(&mut self) {
    for mut tensor in &mut self.output_tensors {
      // TODO: Is TF_DeleteTensor NULL safe?
      if !tensor.is_null() {
        unsafe {
          tf::TF_DeleteTensor(*tensor);
        }
      }
      *tensor = std::ptr::null_mut();
    }
  }
}

#[allow(deprecated)]
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
///
/// Currently, all implementors must *not* implement Drop (or transitively contain
/// anything that does) and must be bit-for-bit compatible with the corresponding C
/// type. Clients must not implement this trait.
pub trait TensorType: Default + Clone + Copy + Display + Debug + 'static {
  // TODO: Use associated constants when/if available
  /// Returns the DataType that corresponds to this type.
  fn data_type() -> DataType;
}

macro_rules! tensor_type {
  ($rust_type:ty, $tensor_type:ident) => {
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
tensor_type!(Complex<f32>, Complex64);
tensor_type!(Complex<f64>, Complex128);
tensor_type!(i64, Int64);
tensor_type!(bool, Bool);
// TODO: provide type for BFloat16

macro_rules! q_type {
  ($rust_type:ident, $(#[$attr:meta])* type $q_type:ident) => {
    $(#[$attr])*
    #[derive(Clone,Copy,Default,Debug,Eq,PartialEq,Ord,PartialOrd)]
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

q_type!(i8,
        /// Quantized type for i8.
        type QInt8);
q_type!(u8,
        /// Quantized type for u8.
        type QUInt8);
q_type!(i16,
        /// Quantized type for i16.
        type QInt16);
q_type!(u16,
        /// Quantized type for u16.
        type QUInt16);
q_type!(i32,
        /// Quantized type for i32.
        type QInt32);

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
#[derive(Debug)]
pub struct Tensor<T: TensorType> {
  inner: *mut tf::TF_Tensor,
  data: Buffer<T>,
  dims: Vec<u64>,
  owned: bool,
}

unsafe extern "C" fn noop_deallocator(_: *mut c_void, _: size_t, _: *mut c_void) -> () {}

unsafe extern "C" fn deallocator(_: *mut c_void, _: size_t, buffer: *mut c_void) -> () {
  tf::TF_DeleteBuffer(buffer as *mut tf::TF_Buffer);
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
  pub fn new_with_buffer(dims: &[u64], mut data: Buffer<T>) -> Result<Self> {
    let total = product(dims);
    if total != data.len() as u64 {
      return Err(invalid_arg!("Dimensions {:?} do not match buffer length {}", dims, data.len()));
    }
    unsafe {
      // Be careful.  TF_NewTensor may copy the data and deallocate the original buffer.
      let inner = tf::TF_NewTensor(
        T::data_type().to_c(),
        dims.as_ptr() as *const _,
        dims.len() as c_int,
        data.as_ptr() as *mut _,
        data.len() * mem::size_of::<T>(),
        Some(if data.is_owned() {deallocator} else {noop_deallocator}),
        if data.is_owned() {data.inner_mut() as *mut _} else {std::ptr::null_mut()});
      data.set_owned(false);
      Ok(Tensor {
        inner: inner,
        data: Buffer::from_ptr(tf::TF_TensorData(inner) as *mut T, total as usize),
        dims: Vec::from(dims),
        owned: true,
      })
    }
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
    if DataType::from_c(tf::TF_TensorType(tensor)) != T::data_type() {
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
      dims: dims,
      owned: true,
    })
  }

  /// The caller is responsible for deleting the tensor.
  unsafe fn into_ptr(mut self) -> *mut tf::TF_Tensor {
    // This flag is used by drop.
    self.owned = false;
    self.inner
  }
}

impl<T: TensorType> Drop for Tensor<T> {
  fn drop(&mut self) {
    if self.owned {
      unsafe {
        tf::TF_DeleteTensor(self.inner);
      }
    }
  }
}

impl<T: TensorType> Deref for Tensor<T> {
  type Target = Buffer<T>;

  #[inline]
  fn deref(&self) -> &Buffer<T> {
    &self.data
  }
}

impl<T: TensorType> DerefMut for Tensor<T> {
  #[inline]
  fn deref_mut<'a>(&'a mut self) -> &'a mut Buffer<T> {
    &mut self.data
  }
}

////////////////////////

/// Dynamically loaded plugins.
/// The C API doesn't provide a way to unload libraries, so nothing happens when this goes out of scope.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
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

/// This exposes Buffer behavior without making it public.
trait BufferTrait {
  fn is_owned(&self) -> bool;
  fn set_owned(&mut self, owned: bool);
  fn inner(&self) -> *const tf::TF_Buffer;
  fn inner_mut(&mut self) -> *mut tf::TF_Buffer;
}

/// This exposes Graph behavior without making it public.
trait GraphTrait {
  fn inner(&self) -> *mut tf::TF_Graph;
}


/// This exposes Operation behavior without making it public.
trait OperationTrait {
  fn inner(&self) -> *mut tf::TF_Operation;
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;

  #[allow(deprecated)]
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
