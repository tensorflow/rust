//! This crate provides Rust bindings for the
//! [`TensorFlow`](https://www.tensorflow.org) machine learning library.
//!
//! If you aren't sure how to use something, please see the
//! [examples](https://github.com/tensorflow/rust/tree/master/examples) folder.

// Note that we allow trivial_casts, trivial_numeric_casts, and
// unused_qualifications, because they can show up when casting to a C type that
// may differ between platforms.
#![warn(
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    unused_extern_crates,
    unused_import_braces
)]

use half::f16;
use libc::size_t;
use libc::{c_int, c_uint};
#[cfg(feature = "ndarray")]
use ndarray::{Array, ArrayBase, Data, Dim, Dimension, IxDynImpl};
use num_complex::Complex;
use protobuf::ProtobufEnum;
use std::alloc;
use std::borrow::Borrow;
use std::cell::Cell;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::error::Error;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::IntoStringError;
use std::ffi::NulError;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::mem;
use std::num::ParseIntError;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Drop;
use std::ops::Index;
use std::os::raw::c_char;
use std::os::raw::c_void as std_c_void;
use std::process;
use std::ptr;
use std::slice;
use std::str::Utf8Error;
use tensorflow_sys as tf;

////////////////////////

/// Will panic if `msg` contains an embedded 0 byte.
macro_rules! invalid_arg {
  ($fmt:expr) => {
    crate::Status::new_set(crate::Code::InvalidArgument, $fmt).unwrap()
  };
  ($fmt:expr, $($arg:tt)*) => ({
    let msg = format!($fmt, $($arg)*);
    crate::Status::new_set(crate::Code::InvalidArgument, &msg).unwrap()
  });
}

////////////////////////

macro_rules! impl_new {
    ($name: ident, $call:ident, $doc:expr) => {
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            #[doc = $doc]
            pub fn new() -> Self {
                unsafe {
                    let inner = tf::$call();
                    assert!(!inner.is_null());
                    $name { inner: inner }
                }
            }
        }
    };
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
    };
}

////////////////////////

macro_rules! c_enum {
  ($c_name:ident, $(#[$enum_attr:meta])* $enum_name:ident { $( $(#[$attr:meta])*
      $name:ident = $num:expr),* }) => {
    $(#[$enum_attr])*
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
          $enum_name::UnrecognizedEnumValue(c) => *c,
          $($enum_name::$name => $num),*
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
      fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        match self {
          $($enum_name::$name => f.write_str(stringify!($name)),)*
          $enum_name::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
      }
    }
  };
  ($c_name:ident, $(#[$enum_attr:meta])* $enum_name:ident { $( $(#[$attr:meta])*
      $name:ident = $num:expr,)* }) => {
    c_enum!($c_name, $(#[$enum_attr])* $enum_name { $( $(#[$attr])* $name = $num),* });
  };
  // Deprecated pattern.
  ($doc:expr, $c_name:ident, $(#[$enum_attr:meta])* $enum_name:ident { $( $(#[$attr:meta])* value
      $name:ident = $num:expr),* }) => {
    c_enum!($c_name, #[doc = $doc] $(#[$enum_attr])*
            $enum_name { $( $(#[$attr])* $name = $num),* });
  };
  // Deprecated pattern.
  ($doc:expr, $c_name:ident, $(#[$enum_attr:meta])* $enum_name:ident { $( $(#[$attr:meta])* value
      $name:ident = $num:expr,)* }) => {
    c_enum!($c_name, #[doc = $doc] $(#[$enum_attr])*
            $enum_name { $( $(#[$attr])* $name = $num),* });
  }
}

////////////////////////

mod protos;

mod buffer;
use crate::buffer::Buffer;

mod graph;
pub use crate::graph::*;

mod scope;
pub use crate::scope::*;

mod session;
pub use crate::session::*;

pub mod expr;

pub mod io;

pub mod ops;

mod variable;
pub use crate::variable::*;

pub mod train;

mod saved_model;
pub use saved_model::*;

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
  /// offset that is not in the range [0,2<sup>32</sup>-1], but it will generate
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
  /// Note that this is not the same as Half.  BFloat16 is not an IEEE-754
  /// 16-bit float.  See
  /// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.h
  /// for details.
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

  /// A dynamic type similar to std::any::Any.
  value Variant = 21,

  /// 32-bit unsigned integer.
  value UInt32 = 22,

  /// 64-bit unsigned integer.
  value UInt64 = 23,
});

impl Default for DataType {
    fn default() -> DataType {
        DataType::Float
    }
}

impl DataType {
    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::types::DataType {
        if let Some(d) = protos::types::DataType::from_i32(self.to_int() as i32) {
            d
        } else {
            // This is unfortunate, but the protobuf crate doesn't support unrecognized enum values.
            panic!("Unable to convert {} to a protobuf DataType", self);
        }
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: protos::types::DataType) -> Self {
        Self::from_int(proto.value() as c_uint)
    }
}

////////////////////////

/// Holds error information when communicating with back and forth with `tensorflow`.
///
/// It either has an `Code::Ok` code, or otherwise an error code with an associated message.
pub struct Status {
    inner: *mut tf::TF_Status,
}

impl_new!(
    Status,
    TF_NewStatus,
    "Creates a status with `Code::Ok` and no message."
);
impl_drop!(Status, TF_DeleteStatus);

unsafe impl Send for Status {}
unsafe impl Sync for Status {}

impl Status {
    /// Creates a status and sets its code and message.
    pub fn new_set(code: Code, msg: &str) -> std::result::Result<Status, NulError> {
        let mut status = Status::new();
        status.set(code, msg)?;
        Ok(status)
    }

    /// Creates a status and sets its code and message.
    pub fn new_set_lossy(code: Code, msg: &str) -> Status {
        let mut status = Status::new();
        status.set_lossy(code, msg);
        status
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
        if self.is_ok() {
            Ok(())
        } else {
            Err(self)
        }
    }

    /// Sets the code and message.
    pub fn set(&mut self, code: Code, msg: &str) -> std::result::Result<(), NulError> {
        let message = CString::new(msg)?;
        unsafe {
            tf::TF_SetStatus(self.inner, code.to_c(), message.as_ptr());
        }
        Ok(())
    }

    /// Sets the code and message.
    pub fn set_lossy(&mut self, code: Code, msg: &str) {
        let message = match CString::new(msg) {
            Ok(x) => x,
            Err(e) => {
                let pos = e.nul_position();
                let mut truncated_bytes = e.into_vec();
                truncated_bytes.truncate(pos);
                let mut new_msg: Vec<u8> = "(original error truncated due to internal nul byte) "
                    .as_bytes()
                    .into();
                new_msg.extend(&truncated_bytes);
                unsafe { CString::from_vec_unchecked(new_msg) }
            }
        };
        unsafe {
            tf::TF_SetStatus(self.inner, code.to_c(), message.as_ptr());
        }
    }

    /// Returns a mutable pointer to the inner tensorflow Status `TF_Status`.
    fn inner(&mut self) -> *mut tf::TF_Status {
        self.inner
    }
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

impl From<Utf8Error> for Status {
    fn from(_e: Utf8Error) -> Self {
        invalid_arg!("String contained invalid UTF-8")
    }
}

impl From<IntoStringError> for Status {
    fn from(e: IntoStringError) -> Self {
        invalid_arg!("Error converting C string to Rust string: {}", e)
    }
}

impl From<ParseIntError> for Status {
    fn from(e: ParseIntError) -> Self {
        invalid_arg!("Error parsing an integer: {}", e)
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

    fn cause(&self) -> Option<&dyn Error> {
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
    /// `config` should be a serialized [`ConfigProto` proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto).
    /// Returns an error if config was not parsed successfully as a `ConfigProto`.
    pub fn set_config(&mut self, config: &[u8]) -> Result<()> {
        let mut status = Status::new();
        unsafe {
            tf::TF_SetConfig(
                self.inner,
                config.as_ptr() as *const _,
                config.len(),
                status.inner(),
            );
        }
        if status.is_ok() {
            Ok(())
        } else {
            Err(status)
        }
    }
}

impl_new!(
    SessionOptions,
    TF_NewSessionOptions,
    "Creates a blank set of options."
);
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
    /// Internal only; do not use outside of the tensorflow crate.
    ///
    /// Tensor representation for this type. Normally `TensorDataCRepr` for types
    /// that have the same representation in Rust; or `TensorDataNoCRepr` for
    /// types where the Rust and C representations differ.
    #[doc(hidden)]
    type InnerType: TensorInner<Self>;

    /// Returns the DataType that corresponds to this type.
    fn data_type() -> DataType;

    /// Returns the zero value.
    fn zero() -> Self;

    /// Returns the one value.
    fn one() -> Self;

    /// Return true if the data has the same representation in C and Rust and
    /// can be written/read directly.
    fn is_repr_c() -> bool;

    /// Unpacks data from C. Returns an error if `is_repr_c()` is true for this
    /// type or some other error occurred.
    fn unpack(data: &[u8], count: usize) -> Result<Vec<Self>>;

    /// Packs data for sending to C.  Returns an error if `is_repr_c()` returns
    /// true for this type or some other error occurred.
    fn pack(data: &[Self], dims: &[u64]) -> Result<*mut tf::TF_Tensor>;
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr, $one:expr) => {
        impl TensorType for $rust_type {
            type InnerType = TensorDataCRepr<$rust_type>;

            fn data_type() -> DataType {
                DataType::$tensor_type
            }

            fn zero() -> Self {
                $zero
            }

            fn one() -> Self {
                $one
            }

            fn is_repr_c() -> bool {
                true
            }

            fn unpack(_data: &[u8], _count: usize) -> Result<Vec<Self>> {
                Err(Status::new_set(
                    Code::Unimplemented,
                    concat!("Unpacking is not necessary for ", stringify!($rust_type)),
                )
                .unwrap())
            }

            fn pack(_data: &[Self], _dims: &[u64]) -> Result<*mut tf::TF_Tensor> {
                Err(Status::new_set(
                    Code::Unimplemented,
                    concat!("Packing is not necessary for ", stringify!($rust_type)),
                )
                .unwrap())
            }
        }
    };
}

tensor_type!(f16, Half, f16::ZERO, f16::ONE);
tensor_type!(f32, Float, 0.0, 1.0);
tensor_type!(f64, Double, 0.0, 1.0);
tensor_type!(i32, Int32, 0, 1);
tensor_type!(u8, UInt8, 0, 1);
tensor_type!(u16, UInt16, 0, 1);
tensor_type!(u32, UInt32, 0, 1);
tensor_type!(u64, UInt64, 0, 1);
tensor_type!(i16, Int16, 0, 1);
tensor_type!(i8, Int8, 0, 1);
tensor_type!(
    Complex<f32>,
    Complex64,
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0)
);
tensor_type!(
    Complex<f64>,
    Complex128,
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0)
);
tensor_type!(i64, Int64, 0, 1);
tensor_type!(bool, Bool, false, true);

macro_rules! q_type {
  ($rust_type:ident, $(#[$attr:meta])* type $q_type:ident) => {
    $(#[$attr])*
    #[derive(Clone,Copy,Default,Debug,Eq,PartialEq,Ord,PartialOrd)]
    pub struct $q_type($rust_type);

    impl Display for $q_type {
      fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
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
/// Note that this is not the same as half::f16.  BFloat16 is not an IEEE-754
/// 16-bit float.  See
/// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/bfloat16.h
/// for details.
#[derive(Debug, Clone, Copy, Default)]
pub struct BFloat16(u16);

impl Display for BFloat16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> ::std::fmt::Result {
        let val: f32 = (*self).into();
        Display::fmt(&val, f)
    }
}

impl Into<f32> for BFloat16 {
    fn into(self) -> f32 {
        // Assumes that the architecture uses IEEE-754 natively for floats
        // and twos-complement for integers.
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl From<f32> for BFloat16 {
    fn from(value: f32) -> Self {
        // Assumes that the architecture uses IEEE-754 natively for floats
        // and twos-complement for integers.
        BFloat16((value.to_bits() >> 16) as u16)
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

tensor_type!(
    BFloat16,
    BFloat16,
    BFloat16::from(0.0f32),
    BFloat16::from(1.0f32)
);

////////////////////////

unsafe extern "C" fn string_deallocator(
    data: *mut std_c_void,
    length: size_t,
    _deallocator_arg: *mut std_c_void,
) {
    let align = mem::align_of::<tf::TF_TString>();
    let size = mem::size_of::<tf::TF_TString>();
    let layout = alloc::Layout::from_size_align(length, align).unwrap_or_else(|_| {
        eprintln!("internal error: failed to construct layout");
        // make sure not to unwind
        process::abort();
    });
    let count = length / size;
    let tstrings = slice::from_raw_parts_mut(data as *mut tf::TF_TString, count);
    for i in 0..count {
        tf::TF_StringDealloc(&mut tstrings[i]);
    }
    alloc::dealloc(data as *mut _, layout);
}

impl TensorType for String {
    type InnerType = TensorDataNoCRepr<String>;

    fn data_type() -> DataType {
        DataType::String
    }

    fn zero() -> Self {
        "".to_string()
    }

    fn one() -> Self {
        "\u{0001}".to_string()
    }

    fn is_repr_c() -> bool {
        false
    }

    fn unpack(data: &[u8], count: usize) -> Result<Vec<Self>> {
        let tstrings =
            unsafe { slice::from_raw_parts(data.as_ptr() as *const tf::TF_TString, count) };
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let byte_slice = unsafe {
                slice::from_raw_parts(
                    tf::TF_StringGetDataPointer(&tstrings[i]) as *const u8,
                    tf::TF_StringGetSize(&tstrings[i]),
                )
            };
            out.push(std::str::from_utf8(byte_slice)?.to_string());
        }
        Ok(out)
    }

    fn pack(data: &[Self], dims: &[u64]) -> Result<*mut tf::TF_Tensor> {
        let align = mem::align_of::<tf::TF_TString>();
        let size = mem::size_of::<tf::TF_TString>();
        let packed_size = data.len() * size;
        let inner = unsafe {
            let ptr =
                alloc::alloc(alloc::Layout::from_size_align(size * data.len(), align).unwrap());
            assert!(!ptr.is_null(), "allocation failure");
            let inner = tf::TF_NewTensor(
                DataType::String.to_c(),
                dims.as_ptr() as *const _,
                dims.len() as c_int,
                ptr as *mut std_c_void,
                size * data.len(),
                Some(string_deallocator),
                ptr::null_mut(),
            );
            if inner.is_null() {
                return Err(Status::new_set_lossy(
                    crate::Code::Internal,
                    "TF_NewTensor returned null",
                ));
            }
            let buf = slice::from_raw_parts_mut(ptr, packed_size);
            let tstrings =
                slice::from_raw_parts_mut(buf.as_ptr() as *mut tf::TF_TString, data.len());
            for i in 0..data.len() {
                tf::TF_StringInit(&mut tstrings[i]);
                tf::TF_StringCopy(
                    &mut tstrings[i],
                    data[i].as_bytes().as_ptr() as *const c_char,
                    data[i].len(),
                );
            }
            inner
        };
        Ok(inner)
    }
}

////////////////////////

pub(crate) trait AnyTensor: Debug {
    fn inner(&self) -> Result<*mut tf::TF_Tensor>;

    fn data_type(&self) -> DataType;
}

impl AnyTensor for Box<dyn AnyTensor> {
    fn inner(&self) -> Result<*mut tf::TF_Tensor> {
        let borrowed: &dyn AnyTensor = self.borrow();
        borrowed.inner()
    }

    fn data_type(&self) -> DataType {
        let borrowed: &dyn AnyTensor = self.borrow();
        borrowed.data_type()
    }
}

////////////////////////

unsafe fn tensor_dims(tensor: *mut tf::TF_Tensor) -> Vec<u64> {
    let mut dims = Vec::with_capacity(tf::TF_NumDims(tensor) as usize);
    for i in 0..dims.capacity() {
        dims.push(tf::TF_Dim(tensor, i as c_int) as u64);
    }
    dims
}

/// Internal only; do not use outside of the tensorflow crate.
///
/// Inner representation of `Tensor`s.
#[doc(hidden)]
pub trait TensorInner<T>: Debug + Clone
where
    Self: Sized + Deref<Target = [T]> + DerefMut<Target = [T]>,
{
    /// Return the inner representation of a tensor with the given
    /// dimensions.
    fn new_inner(dims: &[u64]) -> Self;

    /// Wraps a TF_Tensor. Returns None if types don't match.
    ///
    /// # Safety
    ///
    /// Must be a valid, non-null pointer. Takes ownership of the TF_Tensor.
    unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self>;

    /// Return a mutable pointer to the C tensor.
    fn as_mut_ptr(&self, dims: &[u64]) -> Result<*mut tf::TF_Tensor>;
}

////////////////////////

/// Inner representation for `Tensor`s of types where C and Rust have the
/// same representation.
#[derive(Debug)]
#[doc(hidden)]
pub struct TensorDataCRepr<T>
where
    T: TensorType,
{
    inner: *mut tf::TF_Tensor,
    /// Equal to the product of the tensor's dimensions.
    data_count: usize,
    phantom: PhantomData<T>,
}

unsafe impl<T> Send for TensorDataCRepr<T> where T: TensorType {}
unsafe impl<T> Sync for TensorDataCRepr<T> where T: TensorType {}

impl<T: TensorType> Drop for TensorDataCRepr<T> {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                tf::TF_DeleteTensor(self.inner);
            }
        }
        self.inner = ptr::null_mut();
    }
}

impl<T> TensorInner<T> for TensorDataCRepr<T>
where
    T: Debug + TensorType + Copy,
{
    fn new_inner(dims: &[u64]) -> Self {
        let total = product(dims) as usize;
        unsafe {
            let inner = tf::TF_AllocateTensor(
                T::data_type().to_c(),
                dims.as_ptr() as *const _,
                dims.len() as c_int,
                total * mem::size_of::<T>(),
            );

            // Zero-initialize allocated memory.
            let data = tf::TF_TensorData(inner);
            let byte_size = tf::TF_TensorByteSize(inner);
            libc::memset(data as *mut libc::c_void, 0, byte_size);

            TensorDataCRepr {
                inner,
                data_count: total,
                phantom: PhantomData,
            }
        }
    }

    // Wraps a TF_Tensor. Returns None if types don't match.
    unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self> {
        if DataType::from_c(tf::TF_TensorType(tensor)) != T::data_type() {
            return None;
        }
        Some(TensorDataCRepr {
            inner: tensor,
            data_count: product(&tensor_dims(tensor)) as usize,
            phantom: PhantomData,
        })
    }

    fn as_mut_ptr(&self, _dims: &[u64]) -> Result<*mut tf::TF_Tensor> {
        assert!(!self.inner.is_null());
        Ok(self.inner)
    }
}

impl<T: TensorType> Deref for TensorDataCRepr<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        let data = unsafe { tf::TF_TensorData(self.inner) } as *mut T;
        unsafe { slice::from_raw_parts(data, self.data_count) }
    }
}

impl<T: TensorType> DerefMut for TensorDataCRepr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let data = unsafe { tf::TF_TensorData(self.inner) } as *mut T;
        unsafe { slice::from_raw_parts_mut(data, self.data_count) }
    }
}

impl<T: TensorType + Copy> Clone for TensorDataCRepr<T> {
    fn clone(&self) -> Self {
        let (inner, total) = unsafe {
            let dims = tensor_dims(self.inner);
            let total = product(&dims) as usize;
            let inner = tf::TF_AllocateTensor(
                T::data_type().to_c(),
                dims.as_ptr() as *const _,
                dims.len() as c_int,
                total * mem::size_of::<T>(),
            );
            (inner, total)
        };
        let mut clone = TensorDataCRepr {
            inner,
            data_count: total,
            phantom: PhantomData,
        };
        clone.deref_mut().copy_from_slice(self.deref());
        clone
    }
}

////////////////////////

/// Inner representation for `Tensor`s of types where C and Rust have
/// different representations.
#[derive(Debug)]
#[doc(hidden)]
pub struct TensorDataNoCRepr<T>
where
    T: TensorType,
{
    inner: Cell<*mut tf::TF_Tensor>,
    /// Points to either the TF_Tensor data or the contents of `unpacked_data`.
    data: Cell<*mut T>,
    /// Equal to the product of the tensor's dimensions.
    data_count: usize,
    unpacked: Cell<bool>,
    /// This is just an easy way to handle deallocation correctly.  According to
    /// the aliasing rules, we shouldn't touch this data because it can be
    /// modified through `data`.
    unpacked_data: RefCell<Option<Vec<T>>>,
}

impl<T> TensorInner<T> for TensorDataNoCRepr<T>
where
    T: Debug + TensorType,
{
    /// Creates a new tensor.
    ///
    /// The data is initialized to zeros.
    fn new_inner(dims: &[u64]) -> Self {
        let total = product(dims) as usize;
        let mut data = Vec::with_capacity(total);
        data.resize(total, T::zero());
        TensorDataNoCRepr {
            inner: Cell::new(ptr::null_mut()),

            data: Cell::new(data.as_mut_ptr()),
            data_count: total,
            unpacked: Cell::new(true),
            unpacked_data: RefCell::new(Some(data)),
        }
    }

    unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self> {
        if DataType::from_c(tf::TF_TensorType(tensor)) != T::data_type() {
            return None;
        }
        Some(TensorDataNoCRepr {
            inner: Cell::new(tensor),
            data: Cell::new(tf::TF_TensorData(tensor) as *mut _),
            data_count: product(&tensor_dims(tensor)) as usize,
            unpacked: Cell::new(false),
            unpacked_data: RefCell::new(None),
        })
    }

    fn as_mut_ptr(&self, dims: &[u64]) -> Result<*mut tf::TF_Tensor> {
        let mut inner = self.inner.get();
        if inner.is_null() {
            let data: &[T] = self;
            inner = T::pack(data, dims)?;
            self.inner.set(inner);
        }

        Ok(inner)
    }
}

impl<T: TensorType> Drop for TensorDataNoCRepr<T> {
    fn drop(&mut self) {
        self.drop_tensor();
    }
}

impl<T> TensorDataNoCRepr<T>
where
    T: TensorType,
{
    // This will panic if `unpacked` is false and `unpacked_data` is already borrowed.
    #[allow(trivial_numeric_casts)]
    fn unpack(&self) {
        if !self.unpacked.get() {
            let mut data = self.unpacked_data.borrow_mut();
            let tensor = self.inner.get();
            let bytes = unsafe {
                slice::from_raw_parts(
                    tf::TF_TensorData(tensor) as *const u8,
                    tf::TF_TensorByteSize(tensor) as usize,
                )
            };
            // The unwrap() may panic (e.g. if a string contains a 0 byte),
            // but there's nothing we can do.  This function is always
            // called from contexts that don't allow us to return an error.
            let mut unpacked = T::unpack(bytes, self.data_count).unwrap();
            assert_eq!(unpacked.len(), self.data_count);
            self.data.set(unpacked.as_mut_ptr());
            *data = Some(unpacked);
            self.unpacked.set(true);
        }
    }

    fn drop_tensor(&self) {
        let inner = self.inner.get();
        if !inner.is_null() {
            unsafe {
                tf::TF_DeleteTensor(inner);
            }
        }
        self.inner.set(ptr::null_mut());
    }
}

impl<T: TensorType> Deref for TensorDataNoCRepr<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.unpack();
        unsafe { slice::from_raw_parts(self.data.get(), self.data_count) }
    }
}

impl<T: TensorType> DerefMut for TensorDataNoCRepr<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.unpack();
        // If the slice is modified, the tensor is stale.
        self.drop_tensor();
        unsafe { slice::from_raw_parts_mut(self.data.get(), self.data_count) }
    }
}

impl<T: TensorType> Clone for TensorDataNoCRepr<T> {
    fn clone(&self) -> Self {
        let dims = unsafe { tensor_dims(self.inner.get()) };
        let mut clone = TensorDataNoCRepr::new_inner(&dims);
        clone.deref_mut().clone_from_slice(self.deref());
        clone
    }
}

/// Holds a multi-dimensional array of elements of a single data type.
///
/// The data buffer stores elements in row major order.  E.g. if data is treated
/// as a vector of `T`:
///
/// ```text
///   element 0:   index (0, ..., 0)
///   element 1:   index (0, ..., 1)
///   ...
/// ```
#[derive(Debug, Clone, Eq)]
pub struct Tensor<T: TensorType> {
    inner: T::InnerType,
    dims: Vec<u64>,
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
        Tensor {
            inner: T::InnerType::new_inner(dims),
            dims: Vec::from(dims),
        }
    }

    /// Sets (copies) the tensor values to the provided ones.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// let a = Tensor::new(&[2, 2]).with_values(&[0_i32, 1, 2, 3]).unwrap();
    /// ```
    pub fn with_values(mut self, value: &[T]) -> Result<Self> {
        if self.len() != value.len() {
            return Err(invalid_arg!(
                "length of values array ({}) is not equal to tensor total elements ({})",
                value.len(),
                self.len()
            ));
        }
        for (e, v) in self.iter_mut().zip(value) {
            e.clone_from(v);
        }
        Ok(self)
    }

    /// Set one single value on the tensor.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// let mut a = Tensor::<i32>::new(&[3, 3, 3]);
    ///
    /// a.set(&[0, 0, 1], 10);
    /// assert_eq!(a[0 + 0 + 1], 10);
    ///
    /// a.set(&[2, 2, 0], 9);
    /// assert_eq!(a[2*9 + 2*3 + 0], 9);
    /// ```
    pub fn set(&mut self, indices: &[u64], value: T) {
        let index = self.get_index(indices);
        self[index] = value;
    }

    /// Get one single value from the Tensor.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// let mut a = Tensor::<i32>::new(&[2, 3, 5]);
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[u64]) -> T {
        let index = self.get_index(indices);
        self[index].clone()
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// let a = Tensor::<f32>::new(&[3, 3, 3]);
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[u64]) -> usize {
        assert!(self.dims.len() == indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index as usize
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Returns the tensor's dimensions as a Shape.
    pub fn shape(&self) -> Shape {
        Shape(Some(self.dims.iter().map(|d| Some(*d as i64)).collect()))
    }

    // Wraps a TF_Tensor. Returns None if types don't match.
    unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self> {
        let mut dims = Vec::with_capacity(tf::TF_NumDims(tensor) as usize);
        for i in 0..dims.capacity() {
            dims.push(tf::TF_Dim(tensor, i as c_int) as u64);
        }

        Some(Tensor {
            inner: T::InnerType::from_tf_tensor(tensor)?,
            dims,
        })
    }
}

impl<T: TensorType> AnyTensor for Tensor<T> {
    fn inner(&self) -> Result<*mut tf::TF_Tensor> {
        self.inner.as_mut_ptr(&self.dims)
    }

    fn data_type(&self) -> DataType {
        T::data_type()
    }
}

impl<T: TensorType> Deref for Tensor<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.inner.deref()
    }
}

impl<T: TensorType> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.inner.deref_mut()
    }
}

impl<T: TensorType> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        let mut tensor = Tensor::new(&[]);
        tensor[0] = value;
        tensor
    }
}

impl<'a, T: TensorType> From<&'a [T]> for Tensor<T> {
    fn from(value: &'a [T]) -> Self {
        let mut tensor: Tensor<T> = Tensor::new(&[value.len() as u64]);
        for (e, v) in tensor.iter_mut().zip(value) {
            e.clone_from(v);
        }
        tensor
    }
}

#[rustversion::since(1.51)]
impl<'a, T: TensorType, const N: usize> From<[T; N]> for Tensor<T> {
    fn from(data: [T; N]) -> Tensor<T> {
        Tensor::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<'a, T: TensorType, const N: usize> From<&[T; N]> for Tensor<T> {
    fn from(data: &[T; N]) -> Tensor<T> {
        Tensor::from(&data[..])
    }
}

#[cfg(feature = "ndarray")]
/// Convert any ndarray::ArrayBase type into a tensorflow::Tensor
impl<T, S, D> From<ArrayBase<S, D>> for Tensor<T>
where
    T: TensorType,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn from(value: ArrayBase<S, D>) -> Self {
        let dims: Vec<u64> = value.shape().iter().map(|x| *x as u64).collect();
        let mut tensor: Tensor<T> = Self::new(&dims);
        for (e, v) in tensor.iter_mut().zip(value.iter()) {
            e.clone_from(v);
        }
        tensor
    }
}

#[cfg(feature = "ndarray")]
/// Convert a tensorflow::Tensor into a dynamic dimensional ndarray::ArrayBase that owns its data
impl<T> From<Tensor<T>> for Array<T, Dim<IxDynImpl>>
where
    T: TensorType,
{
    fn from(value: Tensor<T>) -> Self {
        let dims: Vec<usize> = value.dims.iter().map(|x| *x as usize).collect();
        let dim = Dim(dims);
        let data: Vec<T> = value.iter().map(|x| x.clone()).collect();
        // We can safely unwrap this because we know that `data` will have the
        // correct number of elements to conform to `dim`.
        Array::from_shape_vec(dim, data).unwrap()
    }
}

impl<T: TensorType + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
    }
}

fn write_tensor_recursive<T: Display>(
    f: &mut Formatter<'_>,
    shape: &[u64],
    values: &[T],
) -> ::std::fmt::Result {
    if shape.is_empty() {
        // Handle special case of a scalar tensor.
        write!(f, "{}", values[0])
    } else {
        // Recur with values split into chunks of the next dims size,
        // Surround with brackets and separate with comma.
        write!(f, "[")?;

        if shape[0] > 0 {
            let chunk_size = values.len() / shape[0] as usize;

            for i in 0..shape[0] as usize {
                if i != 0 {
                    write!(f, ", ")?;
                }

                write_tensor_recursive(
                    f,
                    &shape[1..],
                    &values[i * chunk_size..(i + 1) * chunk_size],
                )?;
            }
        }

        write!(f, "]")
    }
}

impl<T: TensorType> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> ::std::fmt::Result {
        write_tensor_recursive(f, &self.dims, self)
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
    op_list: OpList,
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
            let buf = unsafe {
                let stack_buf = tf::TF_GetOpList(inner);
                let heap_buf = tf::TF_NewBuffer();
                (*heap_buf).data = stack_buf.data;
                (*heap_buf).length = stack_buf.length;
                Buffer::<u8>::from_c(heap_buf, true)
            };
            let op_proto: protos::op_def::OpList = protobuf::Message::parse_from_bytes(&buf)
                .map_err(|e| {
                    Status::new_set_lossy(
                        Code::InvalidArgument,
                        &format!("Invalid serialized OpList: {}", e),
                    )
                })?;

            let op_list = OpList::from_proto(&op_proto)?;
            Ok(Library { inner, op_list })
        }
    }

    /// Get the inner library OpList
    pub fn op_list(&self) -> &OpList {
        &self.op_list
    }
}

/// Collection of OpDefs exposed from an external plugin
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct OpList(Vec<OpDef>);

impl OpList {
    // We don't use Into, because we don't want this to be public API.
    #[allow(dead_code)]
    fn into_proto(self) -> protos::op_def::OpList {
        let mut proto = protos::op_def::OpList::new();
        let ops = self
            .0
            .into_iter()
            .map(|op| op.into_proto())
            .collect::<Vec<_>>();
        proto.op.clone_from_slice(&ops);
        proto
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::op_def::OpList) -> Result<Self> {
        let ops = proto
            .get_op()
            .iter()
            .map(|op| OpDef::from_proto(op))
            .collect::<Result<Vec<OpDef>>>()?;
        Ok(Self(ops))
    }
}

impl From<Vec<OpDef>> for OpList {
    fn from(ops: Vec<OpDef>) -> Self {
        Self(ops)
    }
}

impl Into<Vec<OpDef>> for OpList {
    fn into(self) -> Vec<OpDef> {
        self.0
    }
}

/// A Graph operation exposed from an external plugin
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpDef {
    name: String,
    input_arg: Vec<OpArgDef>,
    output_arg: Vec<OpArgDef>,
    attr: Vec<OpAttrDef>,
    summary: String,
    description: String,
    is_commutative: bool,
    is_aggregate: bool,
    is_stateful: bool,
    allows_uninitialized_input: bool,
}

impl OpDef {
    /// Returns the name of the Op
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the input arguments of the Op
    pub fn input_arg(&self) -> &Vec<OpArgDef> {
        &self.input_arg
    }

    /// Returns the output arguments of the Op
    pub fn output_arg(&self) -> &Vec<OpArgDef> {
        &self.output_arg
    }

    /// Returns the attributes of the Op
    pub fn attr(&self) -> &Vec<OpAttrDef> {
        &self.attr
    }

    /// Returns the summary of the Op
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Returns the description of the Op
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns true if the Op is commutative
    pub fn is_commutative(&self) -> bool {
        self.is_commutative
    }

    /// Returns true if the Op aggregates values
    pub fn is_aggregate(&self) -> bool {
        self.is_aggregate
    }

    /// Returns true if the Op maintains state
    pub fn is_stateful(&self) -> bool {
        self.is_stateful
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::op_def::OpDef {
        let input_arg: Vec<protos::op_def::OpDef_ArgDef> = self
            .input_arg
            .into_iter()
            .map(|arg| arg.into_proto())
            .collect();
        let output_arg: Vec<protos::op_def::OpDef_ArgDef> = self
            .output_arg
            .into_iter()
            .map(|arg| arg.into_proto())
            .collect();
        let attr: Vec<protos::op_def::OpDef_AttrDef> = self
            .attr
            .into_iter()
            .map(|attr| attr.into_proto())
            .collect();
        let mut proto = protos::op_def::OpDef::new();
        proto.set_name(self.name);
        proto.set_input_arg(input_arg.into());
        proto.set_output_arg(output_arg.into());
        proto.set_attr(attr.into());
        proto.set_summary(self.summary);
        proto.set_description(self.description);
        proto.set_is_commutative(self.is_commutative);
        proto.set_is_aggregate(self.is_aggregate);
        proto.set_is_stateful(self.is_stateful);
        proto.set_allows_uninitialized_input(self.allows_uninitialized_input);
        proto
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::op_def::OpDef) -> Result<Self> {
        let input_arg = proto
            .get_input_arg()
            .iter()
            .map(|arg| OpArgDef::from_proto(arg))
            .collect::<Result<Vec<OpArgDef>>>()?;
        let output_arg = proto
            .get_output_arg()
            .iter()
            .map(|arg| OpArgDef::from_proto(arg))
            .collect::<Result<Vec<OpArgDef>>>()?;
        let attr = proto
            .get_attr()
            .iter()
            .map(|attr| OpAttrDef::from_proto(attr))
            .collect::<Result<Vec<OpAttrDef>>>()?;
        Ok(Self {
            name: proto.get_name().to_string(),
            input_arg,
            output_arg,
            attr,
            summary: proto.get_summary().to_string(),
            description: proto.get_description().to_string(),
            is_commutative: proto.get_is_commutative(),
            is_aggregate: proto.get_is_aggregate(),
            is_stateful: proto.get_is_stateful(),
            allows_uninitialized_input: proto.get_allows_uninitialized_input(),
        })
    }
}

/// An argument definition for a graph operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpArgDef {
    name: String,
    description: String,
    field_type: DataType,
    type_attr: String,
    number_attr: String,
    type_list_attr: String,
    is_ref: bool,
    // TODO: Add "default_value" and "allowed_values" from OpDef_AttrDef proto
}

impl OpArgDef {
    /// Returns the name of the argument
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the description of the argument
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the data type of the argument
    pub fn field_type(&self) -> DataType {
        self.field_type
    }

    /// Returns the type attribute for this argument
    pub fn type_attr(&self) -> &str {
        &self.type_attr
    }

    /// Returns the number attribute for this argument
    pub fn number_attr(&self) -> &str {
        &self.number_attr
    }

    /// Returns the type list attribute for this argument
    pub fn type_list_attr(&self) -> &str {
        &self.type_list_attr
    }

    /// Returns true if this Arg is a ref
    pub fn is_ref(&self) -> bool {
        self.is_ref
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::op_def::OpDef_ArgDef {
        let mut proto = protos::op_def::OpDef_ArgDef::new();
        proto.set_name(self.name);
        proto.set_description(self.description);
        proto.set_field_type(self.field_type.into_proto());
        proto.set_type_attr(self.type_attr);
        proto.set_number_attr(self.number_attr);
        proto.set_type_list_attr(self.type_list_attr);
        proto.set_is_ref(self.is_ref);
        proto
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::op_def::OpDef_ArgDef) -> Result<Self> {
        Ok(Self {
            name: proto.get_name().to_string(),
            description: proto.get_description().to_string(),
            field_type: DataType::from_proto(proto.get_field_type()),
            type_attr: proto.get_type_attr().to_string(),
            number_attr: proto.get_number_attr().to_string(),
            type_list_attr: proto.get_type_list_attr().to_string(),
            is_ref: proto.get_is_ref(),
        })
    }
}

/// An attribute definition for a graph operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpAttrDef {
    name: String,
    field_type: String,
    description: String,
    has_minimum: bool,
    minimum: i64,
}

impl OpAttrDef {
    /// Returns the name of the attribute
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the description of the attribute
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns true if this attribute has a minimum
    pub fn has_minimum(&self) -> bool {
        self.has_minimum
    }

    /// Returns the minimum for this attribute
    pub fn minimum(&self) -> i64 {
        self.minimum
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::op_def::OpDef_AttrDef {
        let mut proto = protos::op_def::OpDef_AttrDef::new();
        proto.set_name(self.name);
        proto.set_field_type(self.field_type);
        proto.set_description(self.description);
        proto.set_has_minimum(self.has_minimum);
        proto.set_minimum(self.minimum);
        proto
    }

    // We don't use From, because we don't want this to be public API.
    fn from_proto(proto: &protos::op_def::OpDef_AttrDef) -> Result<Self> {
        Ok(Self {
            name: proto.get_name().to_string(),
            field_type: proto.get_field_type().to_string(),
            description: proto.get_description().to_string(),
            has_minimum: proto.get_has_minimum(),
            minimum: proto.get_minimum(),
        })
    }
}

////////////////////////

/// Returns a string describing version information of the
/// `TensorFlow` library. `TensorFlow` is using semantic versioning.
pub fn version() -> std::result::Result<String, Utf8Error> {
    unsafe {
        CStr::from_ptr(tf::TF_Version())
            .to_str()
            .map(|s| s.to_string())
    }
}

/// Returns a serialized KernelList protocol buffer containing KernelDefs for
/// all registered kernels.
pub fn get_all_registered_kernels() -> Result<Vec<u8>> {
    let mut status = Status::new();
    let buf = unsafe {
        let buf = tf::TF_GetAllRegisteredKernels(status.inner());
        if !status.is_ok() {
            return Err(status);
        }
        Buffer::<u8>::from_c(buf, true)
    };
    Ok(Vec::from(buf.as_ref()))
}

/// Returns a serialized KernelList protocol buffer containing KernelDefs for
/// all kernels registered for the operation named `name`.
pub fn get_registered_kernels_for_op(name: &str) -> Result<Vec<u8>> {
    let c_name = CString::new(name)?;
    let mut status = Status::new();
    let buf = unsafe {
        let buf = tf::TF_GetRegisteredKernelsForOp(c_name.as_ptr(), status.inner());
        if !status.is_ok() {
            return Err(status);
        }
        Buffer::<u8>::from_c(buf, true)
    };
    Ok(Vec::from(buf.as_ref()))
}

////////////////////////

/// A Shape is the shape of a tensor.  A Shape may be an unknown rank, or it may
/// have a known rank with each dimension being known or unknown.
#[derive(Debug, Eq, Ord, PartialEq, PartialOrd, Hash, Clone, Default)]
pub struct Shape(Option<Vec<Option<i64>>>);

impl Shape {
    /// Creates a new Shape.
    pub fn new(s: Option<Vec<Option<i64>>>) -> Shape {
        Shape(s)
    }

    /// Returns the number of dimensions if known, or None if unknown.
    pub fn dims(&self) -> Option<usize> {
        match *self {
            Shape(None) => None,
            Shape(Some(ref v)) => Some(v.len()),
        }
    }

    // We don't use Into, because we don't want this to be public API.
    fn into_proto(self) -> protos::tensor_shape::TensorShapeProto {
        match self.0 {
            None => {
                let mut shape = protos::tensor_shape::TensorShapeProto::new();
                shape.set_unknown_rank(true);
                shape
            }
            Some(v) => {
                let mut shape = protos::tensor_shape::TensorShapeProto::new();
                for in_dim in v {
                    shape.mut_dim().push({
                        let mut out_dim = protos::tensor_shape::TensorShapeProto_Dim::new();
                        out_dim.set_size(match in_dim {
                            None => -1,
                            Some(d) => d,
                        });
                        out_dim
                    });
                }
                shape
            }
        }
    }

    // We don't use Into, because we don't want this to be public API.
    fn from_proto(proto: &protos::tensor_shape::TensorShapeProto) -> Self {
        Shape(if proto.get_unknown_rank() {
            None
        } else {
            Some(
                proto
                    .get_dim()
                    .iter()
                    .map(|dim| {
                        if dim.get_size() == -1 {
                            None
                        } else {
                            Some(dim.get_size())
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
    }
}

impl From<Option<Vec<Option<i64>>>> for Shape {
    fn from(data: Option<Vec<Option<i64>>>) -> Shape {
        Shape(data)
    }
}

impl From<&[i32]> for Shape {
    fn from(data: &[i32]) -> Shape {
        Shape(Some(data.iter().map(|i| Some(*i as i64)).collect()))
    }
}

impl From<&[u32]> for Shape {
    fn from(data: &[u32]) -> Shape {
        Shape(Some(data.iter().map(|i| Some(*i as i64)).collect()))
    }
}

impl From<&[i64]> for Shape {
    fn from(data: &[i64]) -> Shape {
        Shape(Some(data.iter().map(|i| Some(*i)).collect()))
    }
}

impl From<&[u64]> for Shape {
    fn from(data: &[u64]) -> Shape {
        Shape(Some(data.iter().map(|i| Some(*i as i64)).collect()))
    }
}

macro_rules! shape_from_array {
    ($N:expr) => {
        impl From<[i32; $N]> for Shape {
            fn from(data: [i32; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<[u32; $N]> for Shape {
            fn from(data: [u32; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<[i64; $N]> for Shape {
            fn from(data: [i64; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<[u64; $N]> for Shape {
            fn from(data: [u64; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<&[i32; $N]> for Shape {
            fn from(data: &[i32; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<&[u32; $N]> for Shape {
            fn from(data: &[u32; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<&[i64; $N]> for Shape {
            fn from(data: &[i64; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }

        impl From<&[u64; $N]> for Shape {
            fn from(data: &[u64; $N]) -> Shape {
                Shape::from(&data[..])
            }
        }
    };
}

#[rustversion::not(since(1.51))]
shape_from_array!(0);
#[rustversion::not(since(1.51))]
shape_from_array!(1);
#[rustversion::not(since(1.51))]
shape_from_array!(2);
#[rustversion::not(since(1.51))]
shape_from_array!(3);
#[rustversion::not(since(1.51))]
shape_from_array!(4);
#[rustversion::not(since(1.51))]
shape_from_array!(5);
#[rustversion::not(since(1.51))]
shape_from_array!(6);
#[rustversion::not(since(1.51))]
shape_from_array!(7);

#[rustversion::since(1.51)]
impl<const N: usize> From<[i32; N]> for Shape {
    fn from(data: [i32; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<[u32; N]> for Shape {
    fn from(data: [u32; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<[i64; N]> for Shape {
    fn from(data: [i64; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<[u64; N]> for Shape {
    fn from(data: [u64; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<&[i32; N]> for Shape {
    fn from(data: &[i32; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<&[u32; N]> for Shape {
    fn from(data: &[u32; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<&[i64; N]> for Shape {
    fn from(data: &[i64; N]) -> Shape {
        Shape::from(&data[..])
    }
}

#[rustversion::since(1.51)]
impl<const N: usize> From<&[u64; N]> for Shape {
    fn from(data: &[u64; N]) -> Shape {
        Shape::from(&data[..])
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

////////////////////////

mod while_loop;
pub use crate::while_loop::*;

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
    fn test_tensor_native_type_zero() {
        let tensor = <Tensor<i32>>::new(&[1000]);

        // Checking against null-initialized slice/vector makes
        // the unit test succeed often on repeated runs.
        for v in tensor.as_ref() {
            assert_eq!(0, *v);
        }
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
            0x0a, 0x2a, 0x0a, 0x01, 0x78, 0x12, 0x0b, 0x50, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f,
            0x6c, 0x64, 0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12,
            0x02, 0x30, 0x01, 0x2a, 0x0b, 0x0a, 0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x02,
            0x3a, 0x00, 0x0a, 0x30, 0x0a, 0x03, 0x79, 0x2f, 0x79, 0x12, 0x05, 0x43, 0x6f, 0x6e,
            0x73, 0x74, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12, 0x02, 0x30,
            0x01, 0x2a, 0x15, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x12, 0x0c, 0x42, 0x0a,
            0x08, 0x01, 0x12, 0x00, 0x2a, 0x04, 0x00, 0x00, 0x00, 0x40, 0x0a, 0x19, 0x0a, 0x01,
            0x79, 0x12, 0x03, 0x4d, 0x75, 0x6c, 0x1a, 0x01, 0x78, 0x1a, 0x03, 0x79, 0x2f, 0x79,
            0x2a, 0x07, 0x0a, 0x01, 0x54, 0x12, 0x02, 0x30, 0x01,
        ];
        let (session, mut graph) = create_session();
        let opts = ImportGraphDefOptions::new();
        let status = graph.import_graph_def(&graph_proto, &opts);
        assert!(status.is_ok());
        let mut x = <Tensor<f32>>::new(&[2]);
        x[0] = 2.0;
        x[1] = 3.0;
        let mut step = SessionRunArgs::new();
        let x_op = graph.operation_by_name_required("x").unwrap();
        step.add_feed(&x_op, 0, &x);
        let y_op = graph.operation_by_name_required("y").unwrap();
        let output_ix = step.request_fetch(&y_op, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.fetch::<f32>(output_ix).unwrap();
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

    #[test]
    fn test_f16() {
        let data: Vec<f16> = vec![-1.0f32, 0.0, 1.0, 2.5]
            .into_iter()
            .map(|x| f16::from_f32(x))
            .collect();
        let tensor = <Tensor<f16>>::new(&[2, 2]).with_values(&data).unwrap();
        assert_eq!(&tensor[..], &data[..]);
    }

    #[test]
    fn test_strings() {
        let mut g = Graph::new();
        let x_op = {
            let mut nd = g.new_operation("Placeholder", "x").unwrap();
            nd.set_attr_type("dtype", DataType::String).unwrap();
            nd.set_attr_shape("shape", &Shape(Some(vec![]))).unwrap();
            nd.finish().unwrap()
        };
        let y_op = {
            let mut nd = g.new_operation("EncodeBase64", "y").unwrap();
            nd.add_input(x_op.clone());
            nd.finish().unwrap()
        };
        let options = SessionOptions::new();
        let session = Session::new(&options, &g).unwrap();
        let mut x = <Tensor<String>>::new(&[2]);
        x[0] = "This is a long string.".to_string();
        x[1] = "This is another long string.".to_string();
        let mut step = SessionRunArgs::new();
        step.add_feed(&x_op, 0, &x);
        let output_ix = step.request_fetch(&y_op, 0);
        session.run(&mut step).unwrap();
        let output_tensor = step.fetch::<String>(output_ix).unwrap();
        assert_eq!(output_tensor.len(), 2);
        assert_eq!(output_tensor[0], "VGhpcyBpcyBhIGxvbmcgc3RyaW5nLg");
        assert_eq!(output_tensor[1], "VGhpcyBpcyBhbm90aGVyIGxvbmcgc3RyaW5nLg");
    }

    #[test]
    fn tensor_clone() {
        let x = Tensor::<i32>::new(&[3]).with_values(&[1, 2, 3]).unwrap();
        let clone = x.clone();
        assert_eq!(x, clone);
    }

    #[test]
    fn tensor_eq() {
        let a = Tensor::<i32>::new(&[3]).with_values(&[1, 2, 3]).unwrap();
        let b = Tensor::<i32>::from(&[1, 2, 3][..]);
        let c = Tensor::<i32>::new(&[3]).with_values(&[1, 2, 4]).unwrap();
        let d = Tensor::<i32>::new(&[3, 1]).with_values(&[1, 2, 3]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[rustversion::since(1.51)]
    #[test]
    fn tensor_from_array() {
        let x = Tensor::<i32>::from([1, 2, 3]);
        assert_eq!(x.as_ref(), &[1, 2, 3]);
    }

    #[rustversion::since(1.51)]
    #[test]
    fn tensor_from_array_ref() {
        let x = Tensor::<i32>::from(&[1, 2, 3]);
        assert_eq!(x.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn tensor_display() {
        let tests = [
            ("1", &[][..], &[1][..]),
            ("[1]", &[1], &[1]),
            ("[1, 2]", &[2], &[1, 2]),
            ("[[1, 2], [3, 4]]", &[2, 2], &[1, 2, 3, 4]),
            ("[[[1], [2]], [[3], [4]]]", &[2, 2, 1], &[1, 2, 3, 4]),
            ("[[[1, 2]], [[3, 4]]]", &[2, 1, 2], &[1, 2, 3, 4]),
            ("[[[[], []]], [[[], []]]]", &[2, 1, 2, 0], &[]),
            ("[[], []]", &[2, 0], &[]),
            ("[[], []]", &[2, 0, 2], &[]),
            ("[]", &[0], &[]),
            ("[]", &[0, 0], &[]),
        ];

        for &(expected, shape, values) in tests.iter() {
            let tensor = Tensor::<i32>::new(shape).with_values(values).unwrap();
            assert_eq!(expected, format!("{}", tensor));
        }
    }

    #[cfg(feature = "ndarray")]
    macro_rules! ndarray_tests {
        ($($name:ident: $type:ty, $dim:expr, $values:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let tensor = Tensor::<$type>::new(&$dim).with_values(&$values).unwrap();
                    let dims = Dim($dim);
                    let array = Array::<$type, _>::from_shape_vec(dims, $values).unwrap();

                    let output_array = Array::from(tensor.clone());
                    assert_eq!(array, output_array);

                    let output_tensor = Tensor::from(array);
                    assert_eq!(tensor, output_tensor);
                }
            )*
        }
    }

    #[cfg(feature = "ndarray")]
    ndarray_tests! {
        test_ndarray_0: f64, vec![1], vec![0.0],
        test_ndarray_1: f32, vec![1, 2], vec![3.1, 4.4],
        test_ndarray_2: i64, vec![2, 2], vec![3, 20, -1, 4],
        test_ndarray_3: i32, vec![1, 3], vec![-4, 100, -200],
        test_ndarray_4: u8, vec![2, 2, 2], vec![1, 1, 2, 2, 3, 3, 4, 4],
        test_ndarray_5: u16, vec![3, 3], vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
    }

    #[test]
    fn test_get_all_registered_kernels() {
        assert!(get_all_registered_kernels().unwrap().len() > 0);
    }

    #[test]
    fn test_get_registered_kernels_for_op() {
        assert!(get_registered_kernels_for_op("Add").unwrap().len() > 0);
    }

    #[test]
    fn test_library_load() {
        // This test is not yet implemented for Windows
        let check_path = match std::env::consts::OS {
            "linux" => Some("test_resources/library/linux/test_op.so"),
            // TODO: The test op needs to be recompiled for macos.
            // "macos" => Some("test_resources/library/macos/test_op.so"),
            _ => None,
        };
        if let Some(path) = check_path {
            let lib = Library::load(path).unwrap();
            let ops: Vec<OpDef> = lib.op_list().clone().into();
            assert!(ops.len() == 1);
            let op = &ops[0];
            assert!(op.name() == "TestOpList");
        };
    }

    #[test]
    fn shape_from_none() {
        assert_eq!(Shape::from(None).dims(), None);
    }

    #[test]
    fn shape_from_array0() {
        let array: [i32; 0] = [];
        assert_eq!(Shape::from(array), Shape::from(&array[..]));
    }

    #[test]
    fn shape_from_array1() {
        assert_eq!(Shape::from([1]), Shape::from(&[1][..]));
    }

    #[test]
    fn shape_from_array1_ref() {
        assert_eq!(Shape::from(&[1]), Shape::from(&[1][..]));
    }

    #[rustversion::since(1.51)]
    #[test]
    fn shape_from_array8() {
        assert_eq!(
            Shape::from([1, 2, 3, 4, 5, 6, 7, 8]),
            Shape::from(&[1, 2, 3, 4, 5, 6, 7, 8][..])
        );
    }

    #[rustversion::since(1.51)]
    #[test]
    fn shape_from_array8_ref() {
        assert_eq!(
            Shape::from(&[1, 2, 3, 4, 5, 6, 7, 8]),
            Shape::from(&[1, 2, 3, 4, 5, 6, 7, 8][..])
        );
    }
}
