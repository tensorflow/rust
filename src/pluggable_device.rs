use crate::{Result, Status};
use std::ffi::CString;
use tensorflow_sys as tf;

/// PluggableDeviceLibrary handler.
#[derive(Debug)]
pub struct PluggableDeviceLibrary {
    inner: *mut tf::TF_Library,
}

impl PluggableDeviceLibrary {
    /// Load the library specified by library_filename and register the pluggable
    /// device and related kernels present in that library. This function is not
    /// supported on embedded on mobile and embedded platforms and will fail if
    /// called.
    ///
    /// Pass "library_filename" to a platform-specific mechanism for dynamically
    /// loading a library. The rules for determining the exact location of the
    /// library are platform-specific and are not documented here.
    pub fn load(library_filename: &str) -> Result<PluggableDeviceLibrary> {
        let status = Status::new();
        let library_filename = CString::new(library_filename)?;
        let lib_handle =
            unsafe { tf::TF_LoadPluggableDeviceLibrary(library_filename.as_ptr(), status.inner) };
        status.into_result()?;

        Ok(PluggableDeviceLibrary { inner: lib_handle })
    }
}

impl Drop for PluggableDeviceLibrary {
    /// Frees the memory associated with the library handle.
    /// Does NOT unload the library.
    fn drop(&mut self) {
        unsafe {
            tf::TF_DeletePluggableDeviceLibraryHandle(self.inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[test]
    fn load_pluggable_device_library() {
        let library_filename = "path-to-library";
        let pluggable_divice_library = PluggableDeviceLibrary::load(library_filename);
        dbg!(&pluggable_divice_library);
        assert!((pluggable_divice_library.is_ok()));
    }
}
