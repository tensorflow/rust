#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;

include!("bindgen.rs");

pub use TF_Code::*;
pub use TF_DataType::*;
pub use TF_AttrType::*;
