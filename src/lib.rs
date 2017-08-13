//! This crate provides Rust bindings for the
//! [`TensorFlow`](https://www.tensorflow.org) machine learning library.

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

use libc::{c_int, c_uint};
use num_complex::Complex;
use std::cmp::Ordering;
use std::error::Error;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Drop;
use std::ops::Index;
use std::str::Utf8Error;

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

macro_rules! impl_new {
  ($name: ident, $call:ident, $doc:expr) => {
    impl $name {
      #[doc = $doc]
      pub fn new() -> Self {
        unsafe {
          let inner = tf::$call();
          assert!(!inner.is_null());
          $name {
            inner: inner,
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
//   ($doc:expr, $c_name:ident, $enum_name:ident
//      { $( $(#[$attr:meta])* $name:ident = $num:expr),* })
// so the enum variants would look like:
//   /// Denotes a foo.
//   Foo = 1,
// but the compiler complains:
//   error: local ambiguity: multiple parsing options: built-in NTs ident ('name')
//   or 1 other option.
// This is https://github.com/rust-lang/rust/issues/24189. Rather than make our
// macro rules inscrutably convoluted, we'll just make our grammar slightly
// noisier and insert a 'value' token before the variant name.
macro_rules! c_enum {
  ($doc:expr, $c_name:ident, $enum_name:ident { $( $(#[$attr:meta])* value
      $name:ident = $num:expr),* }) => {
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
  ($doc:expr, $c_name:ident, $enum_name:ident { $( $(#[$attr:meta])* value
      $name:ident = $num:expr,)* }) => {
    c_enum!($doc, $c_name, $enum_name { $( $(#[$attr])* value $name = $num),* });
  }
}

////////////////////////

mod buffer;
use buffer::Buffer;

mod graph;
pub use graph::*;

mod session;
pub use session::*;

pub mod expr;

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

  /// TensorFlow Resource (name, container, device,...)
  value Resource = 20,
});

////////////////////////

/// Holds error information when communicating with back and forth with `tensorflow`.
///
/// It either has an `Code::Ok` code, or otherwise an error code with an associated message.
pub struct Status {
    inner: *mut tf::TF_Status,
}

impl_new!(Status, TF_NewStatus, "Creates a status with `Code::Ok` and no message.");
impl_drop!(Status, TF_DeleteStatus);

impl Status {
    /// Creates a status and sets its code and message.
    pub fn new_set(code: Code, msg: &str) -> std::result::Result<Status, NulError> {
        let mut status = Status::new();
        status.set(code, msg)?;
        Ok(status)
    }

    /// Returns the status's code.
    pub fn code(&self) -> Code {
        unsafe { Code::from_int(tf::TF_GetCode(self.inner) as u32) }
    }

    /// Returns true if the status's code is `Code::Ok`.
    pub fn is_ok(&self) -> bool {
        self.code() == Code::Ok
    }

    /// Turns the current `Status` into a `Result`.
    fn into_result(self) -> Result<()> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }

    /// Sets the code and message.
    pub fn set(&mut self, code: Code, msg: &str) -> std::result::Result<(), NulError> {
        let message = CString::new(msg)?;
        unsafe {
            tf::TF_SetStatus(self.inner, code.to_c(), message.as_ptr());
        }
        Ok(())
    }

    /// Returns a mutable pointer to the inner tensorflow Status `TF_Status`.
    fn inner(&mut self) -> *mut tf::TF_Status {
        self.inner
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}: ", self.code())?;
        let msg = unsafe {
            match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
                Ok(s) => s,
                Err(_) => "<invalid UTF-8 in message>",
            }
        };
        f.write_str(msg)
    }
}

impl Debug for Status {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{{inner:{:?}, ", self.inner)?;
        write!(f, "{}: ", self.code())?;
        let msg = unsafe {
            match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
                Ok(s) => s,
                Err(_) => "<invalid UTF-8 in message>",
            }
        };
        f.write_str(msg)?;
        write!(f, "}}")?;
        Ok(())
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
        let cstr = CString::new(target)?;
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
        let mut status = Status::new();
        unsafe {
            tf::TF_SetConfig(self.inner,
                             config.as_ptr() as *const _,
                             config.len(),
                             status.inner());
        }
        if status.is_ok() { Ok(()) } else { Err(status) }
    }
}

impl_new!(SessionOptions, TF_NewSessionOptions, "Creates a blank set of options.");
impl_drop!(SessionOptions, TF_DeleteSessionOptions);

////////////////////////

/// Convenience type for `Result` with `Status` as the error type.
pub type Result<T> = std::result::Result<T, Status>;

////////////////////////

/// A Rust type that maps to a `DataType`.
///
/// Currently, all implementors must *not* implement Drop (or transitively contain
/// anything that does) and must be bit-for-bit compatible with the corresponding C
/// type. Clients must not implement this trait.
///
/// This trait doesn't require `num::Zero` or `num::One` because some tensor
/// types (such as `bool` and `String`) don't implement them and we need to
/// supply custom implementations.
pub trait TensorType: Default + Clone + Display + Debug + 'static {
    // TODO: Use associated constants when/if available
    /// Returns the DataType that corresponds to this type.
    fn data_type() -> DataType;

    /// Returns the zero value.
    fn zero() -> Self;

    /// Returns the one value.
    fn one() -> Self;
}

macro_rules! tensor_type {
  ($rust_type:ty, $tensor_type:ident, $zero:expr, $one:expr) => {
    impl TensorType for $rust_type {
      fn data_type() -> DataType {
        DataType::$tensor_type
      }

      fn zero() -> Self {
        $zero
      }

      fn one() -> Self {
        $one
      }
    }
  }
}

tensor_type!(f32, Float, 0.0, 1.0);
tensor_type!(f64, Double, 0.0, 1.0);
tensor_type!(i32, Int32, 0, 1);
tensor_type!(u8, UInt8, 0, 1);
tensor_type!(i16, Int16, 0, 1);
tensor_type!(i8, Int8, 0, 1);
// TODO: provide type for String
tensor_type!(Complex<f32>, Complex64, Complex::new(0.0, 0.0), Complex::new(1.0, 0.0));
tensor_type!(Complex<f64>, Complex128, Complex::new(0.0, 0.0), Complex::new(1.0, 0.0));
tensor_type!(i64, Int64, 0, 1);
tensor_type!(bool, Bool, false, true);

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

    tensor_type!($q_type, $q_type, $q_type(0), $q_type(1));
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

/// BFloat16 provides a Rust type for BFloat16.
#[derive(Debug,Clone,Copy,Default)]
pub struct BFloat16(u16);

impl Display for BFloat16 {
    fn fmt(&self, f: &mut Formatter) -> ::std::fmt::Result {
        let val: f32 = (*self).into();
        Display::fmt(&val, f)
    }
}

impl Into<f32> for BFloat16 {
    fn into(self) -> f32 {
        unsafe {
            // Assumes that the architecture uses IEEE-754 natively for floats
            // and twos-complement for integers.
            mem::transmute::<u32, f32>((self.0 as u32) << 16)
        }
    }
}

impl From<f32> for BFloat16 {
    fn from(value: f32) -> Self {
        unsafe {
            // Assumes that the architecture uses IEEE-754 natively for floats
            // and twos-complement for integers.
            BFloat16((mem::transmute::<f32, u32>(value) >> 16) as u16)
        }
    }
}

impl PartialEq for BFloat16 {
    fn eq(&self, other: &BFloat16) -> bool {
        let x: f32 = (*self).into();
        let y: f32 = (*other).into();
        x.eq(&y)
    }
}

impl PartialOrd for BFloat16 {
    fn partial_cmp(&self, other: &BFloat16) -> Option<Ordering> {
        let x: f32 = (*self).into();
        let y: f32 = (*other).into();
        x.partial_cmp(&y)
    }
}

tensor_type!(BFloat16, BFloat16, BFloat16::from(0.0f32), BFloat16::from(1.0f32));

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

#[inline]
fn product(values: &[u64]) -> u64 {
    values.iter().product()
}

impl<T: TensorType> Tensor<T> {
    /// Creates a new tensor.
    ///
    /// The data is initialized to zeros.
    pub fn new(dims: &[u64]) -> Self {
        let total = product(dims);
        unsafe {
            let inner = tf::TF_AllocateTensor(T::data_type().to_c(),
                                              dims.as_ptr() as *const _,
                                              dims.len() as c_int,
                                              total as usize * mem::size_of::<T>());
            Tensor {
                inner: inner,
                data: Buffer::from_ptr(tf::TF_TensorData(inner) as *mut T, total as usize),
                dims: Vec::from(dims),
                owned: true,
            }
        }
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
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T: TensorType> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: TensorType> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        let mut tensor = Tensor::new(&[1]);
        tensor[0] = value;
        tensor
    }
}

impl<'a, T: TensorType> From<&'a [T]> for Tensor<T> {
    fn from(value: &'a [T]) -> Self {
        let mut tensor = Tensor::new(&[value.len() as u64]);
        for i in 0..value.len() {
            tensor[i] = value[i].clone();
        }
        tensor
    }
}

////////////////////////

/// Dynamically loaded plugins.
/// The C API doesn't provide a way to unload libraries, so nothing happens when this
/// goes out of scope.
#[allow(missing_copy_implementations)]
#[derive(Debug)]
pub struct Library {
    inner: *mut tf::TF_Library,
}

impl Library {
    /// Loads a library.
    pub fn load(library_filename: &str) -> Result<Self> {
        let c_filename = CString::new(library_filename)?;
        let mut status = Status::new();
        let inner = unsafe { tf::TF_LoadLibrary(c_filename.as_ptr(), status.inner()) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Library { inner: inner })
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

/// Returns a string describing version information of the
/// `TensorFlow` library. `TensorFlow` is using semantic versioning.
pub fn version() -> std::result::Result<String, Utf8Error> {
    unsafe { CStr::from_ptr(tf::TF_Version()).to_str().map(|s| s.to_string()) }
}

////////////////////////

/// A Shape is the shape of a tensor.  A Shape may be an unknown rank, or it may
/// have a known rank with each dimension being known or unknown.
#[derive(Debug,Eq,Ord,PartialEq,PartialOrd,Hash,Clone)]
pub struct Shape(Option<Vec<Option<i64>>>);

impl Shape {
    /// Returns the number of dimensions if known, or None if unknown.
    pub fn dims(&self) -> Option<usize> {
        match *self {
            Shape(None) => None,
            Shape(Some(ref v)) => Some(v.len()),
        }
    }
}

impl From<Option<Vec<Option<i64>>>> for Shape {
    fn from(data: Option<Vec<Option<i64>>>) -> Shape {
        Shape(data)
    }
}

impl Into<Option<Vec<Option<i64>>>> for Shape {
    fn into(self) -> Option<Vec<Option<i64>>> {
        self.0
    }
}

static UNKNOWN_DIMENSION: Option<i64> = None;

impl Index<usize> for Shape {
    type Output = Option<i64>;

    fn index(&self, index: usize) -> &Option<i64> {
        match self.0 {
            None => &UNKNOWN_DIMENSION,
            Some(ref v) => {
                if index < v.len() {
                    &v[index]
                } else {
                    &UNKNOWN_DIMENSION
                }
            }
        }
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    fn create_session() -> (Session, Graph) {
        let graph = Graph::new();
        let options = SessionOptions::new();
        match Session::new(&options, &graph) {
            Ok(session) => (session, graph),
            Err(status) => panic!("Creating session failed with status: {}", status),
        }
    }

    #[test]
    fn smoke() {
        create_session();
    }

    #[test]
    fn test_close() {
        let (mut session, _) = create_session();
        let status = session.close();
        assert!(status.is_ok());
    }

    #[test]
    fn test_tensor() {
        let mut tensor = <Tensor<f32>>::new(&[2, 3]);
        assert_eq!(tensor.len(), 6);
        tensor[0] = 1.0;
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
    fn test_run() {
        // Graph is just y = 2 * x
        let graph_proto = vec![
      0x0a, 0x2a, 0x0a, 0x01, 0x78, 0x12, 0x0b, 0x50, 0x6c, 0x61, 0x63, 0x65, 0x68,
      0x6f, 0x6c, 0x64, 0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70,
      0x65, 0x12, 0x02, 0x30, 0x01, 0x2a, 0x0b, 0x0a, 0x05, 0x73, 0x68, 0x61, 0x70,
      0x65, 0x12, 0x02, 0x3a, 0x00, 0x0a, 0x30, 0x0a, 0x03, 0x79, 0x2f, 0x79, 0x12,
      0x05, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79,
      0x70, 0x65, 0x12, 0x02, 0x30, 0x01, 0x2a, 0x15, 0x0a, 0x05, 0x76, 0x61, 0x6c,
      0x75, 0x65, 0x12, 0x0c, 0x42, 0x0a, 0x08, 0x01, 0x12, 0x00, 0x2a, 0x04, 0x00,
      0x00, 0x00, 0x40, 0x0a, 0x19, 0x0a, 0x01, 0x79, 0x12, 0x03, 0x4d, 0x75, 0x6c,
      0x1a, 0x01, 0x78, 0x1a, 0x03, 0x79, 0x2f, 0x79, 0x2a, 0x07, 0x0a, 0x01, 0x54,
      0x12, 0x02, 0x30, 0x01
    ];
        let (mut session, mut graph) = create_session();
        let opts = ImportGraphDefOptions::new();
        let status = graph.import_graph_def(&graph_proto, &opts);
        assert!(status.is_ok());
        let mut x = <Tensor<f32>>::new(&[2]);
        x[0] = 2.0;
        x[1] = 3.0;
        let mut step = StepWithGraph::new();
        let x_op = graph.operation_by_name_required("x").unwrap();
        step.add_input(&x_op, 0, &x);
        let y_op = graph.operation_by_name_required("y").unwrap();
        let output_ix = step.request_output(&y_op, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.take_output::<f32>(output_ix).unwrap();
        assert_eq!(output_tensor.len(), 2);
        assert_eq!(output_tensor[0], 4.0);
        assert_eq!(output_tensor[1], 6.0);
    }

    #[test]
    fn test_bfloat16() {
        let data = [-1.0f32, 0.0, 1.0, 2.5];
        for i in 0..data.len() {
            let x = data[i];
            let bfx = BFloat16::from(x);
            assert_eq!(<BFloat16 as Into<f32>>::into(bfx), x);
            assert_eq!(bfx.partial_cmp(&bfx), Some(Ordering::Equal));
            assert!(bfx.eq(&bfx));
            for j in 0..i {
                let y = data[j];
                let bfy = BFloat16::from(y);
                assert_eq!(bfx.partial_cmp(&bfy), Some(Ordering::Greater));
                assert_eq!(bfy.partial_cmp(&bfx), Some(Ordering::Less));
                assert!(!bfx.eq(&bfy));
            }
        }
        assert_eq!(<BFloat16 as Into<f32>>::into(BFloat16::default()), 0.0f32);
        assert_eq!(BFloat16::from(1.5f32).to_string(), "1.5");
    }
}
