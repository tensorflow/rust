use tensorflow_sys as ffi;

/// Test that the library is linked properly
#[test]
fn linkage() {
    unsafe {
        let buffer = ffi::TF_NewBuffer();
        assert!(!buffer.is_null());
        ffi::TF_DeleteBuffer(buffer);
    }
}

/// Test that the eager API works.
#[cfg(feature = "eager")]
#[test]
fn tfe_tensor_handle() {
    let data = vec![0.0f32; 100];
    let shape = [1, 10, 10];
    let num_elements = shape.iter().fold(1, |t, x| t * x) as usize;

    unsafe {
        let tf_tensor = ffi::TF_AllocateTensor(
            ffi::TF_FLOAT,
            shape.as_ptr(),
            shape.len() as i32,
            num_elements * std::mem::size_of::<f32>(),
        );
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            ffi::TF_TensorData(tf_tensor) as *mut f32,
            ffi::TF_TensorByteSize(tf_tensor) / std::mem::size_of::<f32>(),
        );
        let status = ffi::TF_NewStatus();
        let tfe_handle = ffi::TFE_NewTensorHandle(tf_tensor, status);

        ffi::TF_DeleteStatus(status);
        ffi::TFE_DeleteTensorHandle(tfe_handle);
        ffi::TF_DeleteTensor(tf_tensor);
    }
}
