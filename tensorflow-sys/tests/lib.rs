use tensorflow_sys as ffi;

#[test]
fn linkage() {
    unsafe {
        let buffer = ffi::TF_NewBuffer();
        assert!(!buffer.is_null());
        ffi::TF_DeleteBuffer(buffer);
    }
}

#[test]
fn eager_api() {
    unsafe {
        let buffer = ffi::TFE_NewContextOptions();
        assert!(!buffer.is_null());
        ffi::TFE_DeleteContextOptions(buffer);
    }
}
