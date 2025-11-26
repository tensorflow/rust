#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[allow(deref_nullptr)]
// FIXME old bindgen code has undefined behaviour in tests, see https://github.com/rust-lang/rust-bindgen/pull/2055
mod c_api;
pub use c_api::*;

#[cfg(feature = "eager")]
mod eager;
#[cfg(feature = "eager")]
pub use eager::*;

#[cfg(feature = "experimental")]
mod c_api_experimental;
#[cfg(feature = "experimental")]
pub use c_api_experimental::*;

pub use crate::TF_AttrType::*;
pub use crate::TF_Code::*;
pub use crate::TF_DataType::*;
