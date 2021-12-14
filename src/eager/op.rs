#![allow(dead_code)] // until raw_ops are implemented

use libc::c_float;
use libc::c_int;
use libc::c_uchar;
use libc::c_void;
use libc::size_t;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::mem::{self, ManuallyDrop};
use std::os::raw::c_void as std_c_void;
use std::ptr;

use crate::eager::{Context, TensorHandle};
use crate::{AnyTensor, Code, DataType, Result, Shape, Status};

use tensorflow_sys as tf;

/// Description of the TensorFlow op to execute, for the eager execution.
///
/// The lifetime of this Op is bounded by the provided 'ctx'. This requirement
/// comes from the underlying C-API implementation.
#[derive(Debug)]
pub(super) struct Op<'a> {
    pub(super) inner: *mut tf::TFE_Op,
    ctx: PhantomData<&'a Context>,
}

impl<'a> Drop for Op<'a> {
    fn drop(&mut self) {
        unsafe {
            tf::TFE_DeleteOp(self.inner);
        }
    }
}

/// Thin wrapper around a Context obtained from eager::Op.
///
/// Since the context taken from the Op is just a reference to the Context
/// under which the Op was created, this wrapper is needed to ensure that the
/// Context is not dropped here.
pub(super) struct OpContext<'a> {
    ctx: ManuallyDrop<Context>,
    lifetime: PhantomData<&'a Context>,
}

impl<'a> Op<'a> {
    pub(super) fn new(ctx: &'a Context, op_or_function_name: &str) -> Result<Self> {
        let status = Status::new();

        let c_op_or_function_name = CString::new(op_or_function_name)?;
        let inner =
            unsafe { tf::TFE_NewOp(ctx.inner, c_op_or_function_name.as_ptr(), status.inner) };
        if inner.is_null() || !status.is_ok() {
            return Err(status);
        }
        Ok(Self {
            inner,
            ctx: PhantomData,
        })
    }

    pub(super) fn get_name(&self) -> Result<&str> {
        let status = Status::new();

        let name = unsafe {
            let name = tf::TFE_OpGetName(self.inner, status.inner);
            CStr::from_ptr(name)
        };
        if status.is_ok() {
            return Ok(name.to_str()?);
        }
        Err(status)
    }

    /// Return the context in which this op will be executed.
    pub(super) fn get_context(&self) -> Result<OpContext<'a>> {
        let status = Status::new();
        let inner = unsafe { tf::TFE_OpGetContext(self.inner, status.inner) };
        if status.is_ok() {
            let ctx = ManuallyDrop::new(Context { inner });
            return Ok(OpContext {
                ctx,
                lifetime: PhantomData,
            });
        }
        Err(status)
    }

    /// Adds an input to this operation.
    pub(super) fn add_input(&mut self, input: &TensorHandle) -> Result<()> {
        let status = Status::new();
        unsafe {
            tf::TFE_OpAddInput(self.inner, input.inner, status.inner);
        };
        status.into_result()
    }

    /// Set the device where this operation is computed.
    pub(super) fn set_device(&mut self, device_name: &str) -> Result<()> {
        let status = Status::new();
        let c_device_name = CString::new(device_name)?;
        unsafe {
            tf::TFE_OpSetDevice(self.inner, c_device_name.as_ptr(), status.inner);
        }
        status.into_result()
    }

    /// Get the device where this operation is computed.
    pub(super) fn get_device(&self) -> Result<&str> {
        let status = Status::new();
        let device_name = unsafe {
            // The returned string remains valid throughout the lifetime of 'op'.
            let device_name = tf::TFE_OpGetDevice(self.inner, status.inner);
            CStr::from_ptr(device_name)
        };
        if status.is_ok() {
            return Ok(device_name.to_str()?);
        }
        Err(status)
    }

    /// Adds multiple inputs to this operation.
    pub(super) fn add_input_list(&mut self, inputs: &[TensorHandle]) -> Result<()> {
        let status = Status::new();
        unsafe {
            let mut inputs: Vec<*mut tf::TFE_TensorHandle> =
                inputs.iter().map(|v| v.inner).collect();
            tf::TFE_OpAddInputList(
                self.inner,
                inputs.as_mut_ptr(),
                inputs.len() as c_int,
                status.inner,
            );
        };
        status.into_result()
    }

    /// Sets the value of a string attribute.
    pub(super) fn set_attr_string(&mut self, attr_name: &str, value: &str) -> Result<()> {
        let attr_name = CString::new(attr_name)?;
        let c_value = value.as_bytes();
        unsafe {
            tf::TFE_OpSetAttrString(
                self.inner,
                attr_name.as_ptr(),
                c_value.as_ptr() as *const std_c_void,
                c_value.len() as size_t,
            );
        }
        Ok(())
    }

    /// Sets the value of an attribute which holds a list of strings.
    pub(super) fn set_attr_string_list<S: AsRef<str>>(
        &mut self,
        attr_name: &str,
        values: &[S],
    ) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let bytes: Vec<&[u8]> = values.iter().map(|x| x.as_ref().as_bytes()).collect();
        let ptrs: Vec<*const c_void> = bytes.iter().map(|x| x.as_ptr() as *const c_void).collect();
        let lens: Vec<size_t> = bytes.iter().map(|x| x.len() as size_t).collect();
        unsafe {
            tf::TFE_OpSetAttrStringList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr() as *const *const std_c_void,
                lens.as_ptr(),
                ptrs.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets an int-valued attribute.
    pub(super) fn set_attr_int(&mut self, attr_name: &str, value: i64) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of ints.
    pub(super) fn set_attr_int_list(&mut self, attr_name: &str, value: &[i64]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrIntList(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr(),
                value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a float-valued attribute.
    pub(super) fn set_attr_float(&mut self, attr_name: &str, value: f32) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of floats.
    pub(super) fn set_attr_float_list(&mut self, attr_name: &str, value: &[f32]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        // Allow trivial_numeric_casts here because f32 is not necessarily equal to c_float.
        let c_value: Vec<c_float> = value.iter().map(|x| *x as c_float).collect();
        unsafe {
            tf::TFE_OpSetAttrFloatList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a boolean-valued attribute.
    pub(super) fn set_attr_bool(&mut self, attr_name: &str, value: bool) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of booleans.
    pub(super) fn set_attr_bool_list(&mut self, attr_name: &str, value: &[bool]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<c_uchar> = value.iter().map(|x| if *x { 1 } else { 0 }).collect();
        unsafe {
            tf::TFE_OpSetAttrBoolList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets a type-valued attribute.
    pub(super) fn set_attr_type(&mut self, attr_name: &str, value: DataType) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of types.
    pub(super) fn set_attr_type_list(&mut self, attr_name: &str, value: &[DataType]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TFE_OpSetAttrTypeList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a shape-valued attribute.
    pub(super) fn set_attr_shape(&mut self, attr_name: &str, value: &Shape) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            match value.0 {
                None => tf::TFE_OpSetAttrShape(
                    self.inner,
                    c_attr_name.as_ptr(),
                    ptr::null(),
                    -1,
                    status.inner,
                ),
                Some(ref dims) => {
                    let c_dims: Vec<i64> = dims.iter().map(|x| (*x).unwrap_or(-1)).collect();
                    tf::TFE_OpSetAttrShape(
                        self.inner,
                        c_attr_name.as_ptr(),
                        c_dims.as_ptr(),
                        c_dims.len() as i32,
                        status.inner,
                    );
                }
            }
        }
        status.into_result()
    }

    /// Sets an attribute which holds an array of shapes.
    pub(super) fn set_attr_shape_list(&mut self, attr_name: &str, value: &[Shape]) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name)?;
        // Convert Option<i64> in each shape to i64 with None becoming -1.
        let c_dims: Vec<Option<Vec<i64>>> = value
            .iter()
            .map(|x| {
                x.0.as_ref()
                    .map(|dims| dims.iter().map(|x| (*x).unwrap_or(-1)).collect())
            })
            .collect();
        let mut ptrs: Vec<*const i64> = c_dims
            .iter()
            .map(|x| match *x {
                None => ptr::null(),
                Some(ref dims) => dims.as_ptr(),
            })
            .collect();
        let lens: Vec<c_int> = value
            .iter()
            .map(|x| match x.0 {
                None => -1,
                Some(ref dims) => dims.len() as c_int,
            })
            .collect();
        unsafe {
            tf::TFE_OpSetAttrShapeList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_mut_ptr(),
                lens.as_ptr(),
                ptrs.len() as c_int,
                status.inner,
            );
        }
        status.into_result()
    }

    /// Sets a tensor-valued attribute.
    pub(super) fn set_attr_any_tensor(
        &mut self,
        attr_name: &str,
        value: &dyn AnyTensor,
    ) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TFE_OpSetAttrTensor(
                self.inner,
                c_attr_name.as_ptr(),
                value.inner()?,
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Execute the operation defined by the `Op` and return hadndles to computed
    /// tensors.
    ///
    /// If async execution is enabled, the call may simply enqueue the execution
    /// and return "non-ready" handles. Note that any handles contained in the `Op`
    /// should not be mutated till the kernel execution actually finishes.
    ///
    /// For sync execution, if any of the inputs to `op` are not ready, this call
    /// will block till they become ready and then return when the kernel execution
    /// is done.
    pub(super) fn execute<const N: usize>(self, ctx: &'a Context) -> Result<[TensorHandle; N]> {
        let mut status = Status::new();

        let mut num_retvals = N as i32;
        let mut retvals: [*mut tf::TFE_TensorHandle; N] = [ptr::null_mut(); N];
        unsafe {
            // 'retvals' must point to a pre-allocated array of TFE_TensorHandle* and
            // '*num_retvals' should be set to the size of this array. It is an error if
            // the size of 'retvals' is less than the number of outputs.
            //
            // This call will update the *num_retvals to the number of outputs without raising an
            // error if it is larger than the number of outputs. However, here we treat that such
            // cases as errors and return an error status.
            tf::TFE_Execute(
                self.inner,
                retvals.as_mut_ptr(),
                &mut num_retvals,
                status.inner,
            );
        }
        if num_retvals != N as i32 {
            status.set_lossy(Code::InvalidArgument, "Invalid number of outputs");
            return Err(status);
        }
        if status.is_ok() {
            let mut handles_uninit: [mem::MaybeUninit<TensorHandle>; N] =
                unsafe { mem::MaybeUninit::uninit().assume_init() };

            for i in 0..N {
                let t = TensorHandle::from_tensor_handle(ctx, retvals[i]);
                handles_uninit[i].write(t);
            }
            // Transmute uninitialized handles to initialized handles. Ideally, we would use
            // `mem::transmute` here, but it is not stable yet for generic sized arrays.
            // ref : https://github.com/rust-lang/rust/issues/61956
            //
            // Following is a workaround for this issue:
            // Using &mut as an assertion of unique "ownership"
            let ptr = &mut handles_uninit as *mut _ as *mut [TensorHandle; N];
            let handles: [TensorHandle; N] = unsafe { ptr.read() };
            mem::forget(handles_uninit);

            return Ok(handles);
        }
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    use crate::eager::raw_ops::add;
    use crate::eager::{Context, ContextOptions, TensorHandle};
    use crate::Tensor;

    #[test]
    fn test_add() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let x = Tensor::new(&[2, 2]).with_values(&[1i32, 2, 3, 4]).unwrap();
        let h_x = TensorHandle::new(&ctx, &x).unwrap();
        let h_y = h_x.copy_sharing_tensor().unwrap();
        let h_z = add(&ctx, &h_x, &h_y).unwrap();
        let z: crate::Tensor<i32> = h_z.resolve().unwrap();
        assert_eq!(&z[..], &[2i32, 4, 6, 8]);
    }

    #[cfg(feature = "tensorflow_gpu")]
    #[test]
    #[ignore]
    fn test_add_gpu() {
        use raw_ops::Add;

        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();
        let devices = ctx.device_list().unwrap();
        let gpu_device = devices
            .iter()
            .find(|d| d.device_type == "GPU")
            .expect("No GPU device was found.");
        let target_device = &gpu_device.name;

        let x = Tensor::new(&[2, 2])
            .with_values(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let h = TensorHandle::new(&ctx, &x).unwrap();
        // Copy to GPU. This creates a new handle managed by the context `ctx`.
        let h_gpu = h.copy_to_device(&ctx, target_device).unwrap();

        let mut add = Add::new();
        add.set_device(target_device);

        let h_z_gpu = add.call(&ctx, &h, &h_gpu).unwrap();
        assert!(&h_z_gpu.device_name().unwrap() == target_device);

        let z: crate::Tensor<f32> = h_z_gpu.resolve().unwrap();
        let expected = [2.0f32, 4.0, 6.0, 8.0];
        for (v0, v1) in z.iter().zip(&expected) {
            assert!((v0 - v1).abs() < f32::EPSILON);
        }
    }
}
