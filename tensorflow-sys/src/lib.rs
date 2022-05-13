#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

#[cfg(feature = "eager")]
mod eager;
#[cfg(feature = "eager")]
pub use eager::*;

#[cfg(feature = "runtime_linking")]
mod runtime_linking;
#[cfg(feature = "runtime_linking")]
pub use runtime_linking::*;

#[cfg(not(feature = "runtime_linking"))]
include!("c_api.rs");

#[cfg(not(feature = "runtime_linking"))]
pub use crate::TF_AttrType::*;
pub use crate::TF_Code::*;
pub use crate::TF_DataType::*;

#[cfg(feature = "runtime_linking")]
pub mod library {
    use std::path::PathBuf;

    // Include the definition of `load` here. This allows hiding all of the "extra" linking-related
    // functions in the same place, without polluting the top-level namespace (which should only
    // contain foreign functions and types).
    #[doc(inline)]
    pub use super::runtime_linking::load;

    pub fn find() -> Option<PathBuf> {
        super::runtime_linking::find("tensorflow")
    }
}
