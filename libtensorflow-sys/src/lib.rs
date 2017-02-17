#![allow(dead_code)]
#![allow(missing_copy_implementations)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]

extern crate libc;

use libc::size_t;

include!(concat!(env!("OUT_DIR"), "/ffi.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        unsafe {
            let session_options = TF_NewSessionOptions();
            let status = TF_NewStatus();
            let session = TF_NewSession(session_options, status);
            TF_DeleteSession(session, status);
            TF_DeleteStatus(status);
            TF_DeleteSessionOptions(session_options);
        }
    }
}
