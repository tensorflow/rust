//! C API extensions to experiment with eager execution of kernels.
//!
//! WARNING: The underlying C-API for the eager execution is not guaranteed to be
//! stable and can be changed without notice, which could result in breaking.

mod context;
pub use context::*;

mod tensor_handle;
pub use tensor_handle::*;
