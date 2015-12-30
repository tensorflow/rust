extern crate libtensorflow_sys;

#[test]
fn smoke() {
  use libtensorflow_sys::*;

  unsafe {
    let session_options = TF_NewSessionOptions();
    let status = TF_NewStatus();
    let session = TF_NewSession(session_options, status);
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
    TF_DeleteSessionOptions(session_options);
  }
}
