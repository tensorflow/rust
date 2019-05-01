use libc::{c_int, int64_t, size_t};
use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::c_void;
use std::path::Path;
use tensorflow_sys as ffi;

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

#[cfg_attr(feature = "examples_system_alloc", global_allocator)]
#[cfg(feature = "examples_system_alloc")]
static ALLOCATOR: std::alloc::System = std::alloc::System;

fn main() {
    use std::mem::size_of;
    use std::ptr::{null, null_mut};
    use std::slice::from_raw_parts;

    unsafe {
        let options = nonnull!(ffi::TF_NewSessionOptions());
        let status = nonnull!(ffi::TF_NewStatus());
        let graph = nonnull!(ffi::TF_NewGraph());

        let graph_def = read("examples/assets/multiplication.pb"); // c = a * b
        let opts = nonnull!(ffi::TF_NewImportGraphDefOptions());
        let graph_def_buf = ffi::TF_Buffer {
            data: graph_def.as_ptr() as *const c_void,
            length: graph_def.len(),
            data_deallocator: None,
        };
        ffi::TF_GraphImportGraphDef(graph, &graph_def_buf, opts, status);
        ffi::TF_DeleteImportGraphDefOptions(opts);
        ok!(status);

        let session = nonnull!(ffi::TF_NewSession(graph, options, status));
        let mut inputs = vec![];
        let mut input_tensors: Vec<*const ffi::TF_Tensor> = vec![];

        let name = CString::new("a").unwrap();
        let mut data = vec![1f32, 2.0, 3.0];
        let dims = vec![data.len() as int64_t];
        let input_tensor1 = nonnull!(ffi::TF_NewTensor(
            ffi::TF_FLOAT,
            dims.as_ptr(),
            dims.len() as c_int,
            data.as_mut_ptr() as *mut _,
            data.len() as size_t * mem::size_of::<f32>(),
            Some(noop),
            null_mut()
        ));

        let input_op = nonnull!(ffi::TF_GraphOperationByName(graph, name.as_ptr()));
        inputs.push(ffi::TF_Output {
            oper: input_op,
            index: 0,
        });
        input_tensors.push(input_tensor1);

        let name = CString::new("b").unwrap();
        let mut data = vec![4f32, 5.0, 6.0];
        let dims = vec![data.len() as int64_t];
        let input_tensor2 = nonnull!(ffi::TF_NewTensor(
            ffi::TF_FLOAT,
            dims.as_ptr(),
            dims.len() as c_int,
            data.as_mut_ptr() as *mut _,
            data.len() as size_t as size_t * mem::size_of::<f32>(),
            Some(noop),
            null_mut()
        ));

        let input_op = nonnull!(ffi::TF_GraphOperationByName(graph, name.as_ptr()));
        inputs.push(ffi::TF_Output {
            oper: input_op,
            index: 0,
        });
        input_tensors.push(input_tensor2);

        let mut outputs = vec![];
        let mut output_tensors = vec![];

        let name = CString::new("c").unwrap();

        let output_op = nonnull!(ffi::TF_GraphOperationByName(graph, name.as_ptr()));
        outputs.push(ffi::TF_Output {
            oper: output_op,
            index: 0,
        });
        output_tensors.push(null_mut());

        let mut target_names = vec![];

        ffi::TF_SessionRun(
            session,
            null(),
            inputs.as_mut_ptr(),
            input_tensors.as_ptr(),
            inputs.len() as c_int,
            outputs.as_mut_ptr(),
            output_tensors.as_mut_ptr(),
            outputs.len() as c_int,
            target_names.as_mut_ptr(),
            target_names.len() as c_int,
            null_mut(),
            status,
        );
        ok!(status);

        let output_tensor = nonnull!(output_tensors[0]);
        let data = nonnull!(ffi::TF_TensorData(output_tensor)) as *const f32;
        let data = from_raw_parts(
            data,
            ffi::TF_TensorByteSize(output_tensor) / size_of::<f32>(),
        );

        assert_eq!(data, &[1.0 * 4.0, 2.0 * 5.0, 3.0 * 6.0]);

        ffi::TF_CloseSession(session, status);

        ffi::TF_DeleteSession(session, status);
        ffi::TF_DeleteGraph(graph);
        ffi::TF_DeleteTensor(output_tensor);
        ffi::TF_DeleteTensor(input_tensor1);
        ffi::TF_DeleteTensor(input_tensor2);
        ffi::TF_DeleteStatus(status);
        ffi::TF_DeleteSessionOptions(options);
    }

    unsafe extern "C" fn noop(
        _: *mut std::os::raw::c_void,
        _: size_t,
        _: *mut std::os::raw::c_void,
    ) {
    }
}

fn read<T: AsRef<Path>>(path: T) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let mut buffer = vec![];
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    buffer
}
