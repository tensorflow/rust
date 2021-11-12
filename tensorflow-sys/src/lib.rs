#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[cfg(feature = "eager")]
mod eager;
#[cfg(feature = "eager")]
pub use eager::*;

include!("c_api.rs");

pub use crate::TF_AttrType::*;
pub use crate::TF_Code::*;
pub use crate::TF_DataType::*;
