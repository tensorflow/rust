#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
include!("c_api.rs");
include!("types.rs");
include!("finder.rs");
pub use crate::TF_AttrType::*;
pub use crate::TF_Code::*;
pub use crate::TF_DataType::*;

pub mod library {
    use std::path::PathBuf;

    // Include the definition of `load` here. This allows hiding all of the "extra" linking-related
    // functions in the same place, without polluting the top-level namespace (which should only
    // contain foreign functions and types).
    #[doc(inline)]
    pub use super::load;

    pub fn find() -> Option<PathBuf> {
        super::find("tensorflow")
    }
}
