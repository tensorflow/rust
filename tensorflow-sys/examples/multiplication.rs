extern crate libc;
extern crate tensorflow_sys as ffi;

use libc::{c_int, c_longlong, c_void, size_t};
use std::ffi::{CStr, CString};
use std::path::Path;

macro_rules! nonnull(
    ($pointer:expr) => ({
        let pointer = $pointer;
        assert!(!pointer.is_null());
        pointer
    });
);

macro_rules! ok(
    ($status:expr) => ({
        if ffi::TF_GetCode($status) != ffi::TF_OK {
            panic!(CStr::from_ptr(ffi::TF_Message($status)).to_string_lossy().into_owned());
        }
    });
);

fn main() {
    use std::mem::size_of;
    use std::ptr::{null, null_mut};
    use std::slice::from_raw_parts;

    unsafe {
        let options = nonnull!(ffi::TF_NewSessionOptions());
        let status = nonnull!(ffi::TF_NewStatus());
        let session = nonnull!(ffi::TF_NewSession(options, status));

        let graph = read("examples/assets/multiplication.pb"); // c = a * b
        ffi::TF_ExtendGraph(session, graph.as_ptr() as *const _, graph.len() as size_t, status);
        ok!(status);

        let mut input_names = vec![];
        let mut inputs = vec![];

        let name = CString::new("a").unwrap();
        let mut data = vec![1f32, 2.0, 3.0];
        let mut dims = vec![data.len() as c_longlong];
        let tensor = nonnull!(ffi::TF_NewTensor(ffi::TF_FLOAT, dims.as_mut_ptr(),
                                                dims.len() as c_int, data.as_mut_ptr() as *mut _,
                                                data.len() as size_t, Some(noop), null_mut()));

        input_names.push(name.as_ptr());
        inputs.push(tensor);

        let name = CString::new("b").unwrap();
        let mut data = vec![4f32, 5.0, 6.0];
        let mut dims = vec![data.len() as c_longlong];
        let tensor = nonnull!(ffi::TF_NewTensor(ffi::TF_FLOAT, dims.as_mut_ptr(),
                                                dims.len() as c_int, data.as_mut_ptr() as *mut _,
                                                data.len() as size_t, Some(noop), null_mut()));

        input_names.push(name.as_ptr());
        inputs.push(tensor);

        let mut output_names = vec![];
        let mut outputs = vec![];

        let name = CString::new("c").unwrap();

        output_names.push(name.as_ptr());
        outputs.push(null_mut());

        let mut target_names = vec![];

        ffi::TF_Run(session, null(), input_names.as_mut_ptr(), inputs.as_mut_ptr(),
                    input_names.len() as c_int, output_names.as_mut_ptr(), outputs.as_mut_ptr(),
                    output_names.len() as c_int, target_names.as_mut_ptr(),
                    target_names.len() as c_int, null_mut(), status);
        ok!(status);

        let tensor = nonnull!(outputs[0]);
        let data = nonnull!(ffi::TF_TensorData(tensor)) as *const f32;
        let data = from_raw_parts(data, ffi::TF_TensorByteSize(tensor) / size_of::<f32>());

        assert_eq!(data, &[1.0 * 4.0, 2.0 * 5.0, 3.0 * 6.0]);

        ffi::TF_CloseSession(session, status);

        ffi::TF_DeleteTensor(tensor);
        ffi::TF_DeleteSession(session, status);
        ffi::TF_DeleteStatus(status);
        ffi::TF_DeleteSessionOptions(options);
    }

    unsafe extern "C" fn noop(_: *mut c_void, _: size_t, _: *mut c_void) {}
}

fn read<T: AsRef<Path>>(path: T) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let mut buffer = vec![];
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    buffer
}
