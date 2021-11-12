//! C API extensions to experiment with eager execution of kernels.
//! WARNING: Unlike tensorflow/c/c_api.h, the API here is not guaranteed to be
//! stable and can change without notice.

mod context;
pub use context::*;
