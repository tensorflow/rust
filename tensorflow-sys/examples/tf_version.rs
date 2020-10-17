use std::ffi::CStr;
use tensorflow_sys as ffi;

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() {
    println!("{}", unsafe {
        CStr::from_ptr(ffi::TF_Version())
            .to_string_lossy()
            .into_owned()
    },);
}
