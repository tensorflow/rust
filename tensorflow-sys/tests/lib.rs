use tensorflow_sys as ffi;

#[test]
fn linkage() {
    unsafe {
        let buffer = ffi::TF_NewBuffer();
        assert!(!buffer.is_null());
        ffi::TF_DeleteBuffer(buffer);
    }
}
