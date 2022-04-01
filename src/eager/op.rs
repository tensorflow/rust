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

#[cfg(test)]
mod op_test_util;

#[allow(
    non_snake_case,
    clippy::too_many_arguments,
    clippy::derivable_impls,
    clippy::needless_lifetimes
)]
/// This module contains raw_ops that correspond to [`tf.raw_ops`](https://www.tensorflow.org/api_docs/python/tf/raw_ops).
pub mod raw_ops;

/// Description of the TensorFlow op to execute, for the eager execution.
///
/// The lifetime of this Op is bounded by the provided 'ctx'. This requirement
/// comes from the underlying C-API implementation.
#[derive(Debug)]
struct Op<'a> {
    inner: *mut tf::TFE_Op,
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
struct OpContext<'a> {
    ctx: ManuallyDrop<Context>,
    lifetime: PhantomData<&'a Context>,
}

impl<'a> Op<'a> {
    fn new(ctx: &'a Context, op_or_function_name: &str) -> Result<Self> {
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

    /// Returns the op or function name that this op will execute.
    fn get_name(&self) -> Result<&str> {
        let status = Status::new();

        // The returned string remains valid throughout the lifetime of 'op'.
        let c_name = unsafe { tf::TFE_OpGetName(self.inner, status.inner) };
        status.into_result()?;

        let name = unsafe { CStr::from_ptr(c_name).to_str()? };
        Ok(name)
    }

    /// Return the context in which this op will be executed.
    fn get_context(&self) -> Result<OpContext<'a>> {
        let status = Status::new();
        let inner = unsafe { tf::TFE_OpGetContext(self.inner, status.inner) };
        status.into_result()?;

        let ctx = ManuallyDrop::new(Context { inner });
        Ok(OpContext {
            ctx,
            lifetime: PhantomData,
        })
    }

    /// Adds an input to this operation.
    fn add_input(&mut self, input: &TensorHandle) -> Result<()> {
        let status = Status::new();
        unsafe {
            tf::TFE_OpAddInput(self.inner, input.inner, status.inner);
        };
        status.into_result()
    }

    /// Set the device where this operation is computed.
    fn set_device(&mut self, device_name: &str) -> Result<()> {
        let status = Status::new();
        let c_device_name = CString::new(device_name)?;
        unsafe {
            tf::TFE_OpSetDevice(self.inner, c_device_name.as_ptr(), status.inner);
        }
        status.into_result()
    }

    /// Get the device where this operation is computed.
    fn get_device(&self) -> Result<&str> {
        let status = Status::new();
        // The returned string remains valid throughout the lifetime of 'op'.
        let c_device_name = unsafe { tf::TFE_OpGetDevice(self.inner, status.inner) };
        status.into_result()?;
        let device_name = unsafe { CStr::from_ptr(c_device_name).to_str()? };
        Ok(device_name)
    }

    /// Adds multiple inputs to this operation.
    fn add_input_list(&mut self, inputs: &[TensorHandle]) -> Result<()> {
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
    fn set_attr_string(&mut self, attr_name: &str, value: &str) -> Result<()> {
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
    fn set_attr_string_list<S: AsRef<str>>(&mut self, attr_name: &str, values: &[S]) -> Result<()> {
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
    fn set_attr_int(&mut self, attr_name: &str, value: i64) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of ints.
    fn set_attr_int_list(&mut self, attr_name: &str, value: &[i64]) -> Result<()> {
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
    fn set_attr_float(&mut self, attr_name: &str, value: f32) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of floats.
    fn set_attr_float_list(&mut self, attr_name: &str, value: &[f32]) -> Result<()> {
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
    fn set_attr_bool(&mut self, attr_name: &str, value: bool) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of booleans.
    fn set_attr_bool_list(&mut self, attr_name: &str, value: &[bool]) -> Result<()> {
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
    fn set_attr_type(&mut self, attr_name: &str, value: DataType) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of types.
    fn set_attr_type_list(&mut self, attr_name: &str, value: &[DataType]) -> Result<()> {
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
    fn set_attr_shape(&mut self, attr_name: &str, value: &Shape) -> Result<()> {
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
    fn set_attr_shape_list(&mut self, attr_name: &str, value: &[Shape]) -> Result<()> {
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
    fn set_attr_any_tensor(&mut self, attr_name: &str, value: &dyn AnyTensor) -> Result<()> {
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
    fn execute<const N: usize>(self, ctx: &'a Context) -> Result<[TensorHandle; N]> {
        let status = Status::new();

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
        status.into_result()?;

        // If the 'num_retvals' was updated, we treat that as an error. See comment above.
        if num_retvals != N as i32 {
            for i in 0..num_retvals as usize {
                unsafe {
                    tf::TFE_DeleteTensorHandle(retvals[i]);
                }
            }
            let status = Status::new_set_lossy(
                Code::InvalidArgument,
                &format!("Expected {} outputs, got {}", N, num_retvals),
            );
            return Err(status);
        }

        let mut handles_uninit: [mem::MaybeUninit<TensorHandle>; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };

        for i in 0..N {
            let t = unsafe { TensorHandle::from_tensor_handle(ctx, retvals[i]) };
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

        Ok(handles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eager::{Context, ContextOptions, TensorHandle};
    use crate::Tensor;
    use op_test_util::add as add_ut;
    use raw_ops::{add, concat_v2};

    #[cfg(feature = "ndarray")]
    use ndarray::array;

    #[test]
    fn test_add_op() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let x = Tensor::new(&[2, 2])
            .with_values(&[1i32, 2, 3, 4])
            .unwrap()
            .freeze();
        let h_x = TensorHandle::new(&ctx, &x).unwrap();
        let h_y = h_x.copy_sharing_tensor().unwrap();

        let op_name = "Add";
        let mut op = Op::new(&ctx, op_name).unwrap();

        // Required input arguments
        op.add_input(&h_x).unwrap();
        op.add_input(&h_y).unwrap();

        // Execute Op
        const NUMBER_OF_OUTPUTS: usize = 1;
        let [h] = op.execute::<NUMBER_OF_OUTPUTS>(&ctx).unwrap();
        let z = h.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 4, 6, 8]).unwrap();
        assert_eq!(z, expected);
    }

    #[test]
    fn test_invalid_add() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let x = Tensor::new(&[2, 2])
            .with_values(&[1i32, 2, 3, 4])
            .unwrap()
            .freeze();
        let h_x = TensorHandle::new(&ctx, &x).unwrap();
        let h_y = h_x.copy_sharing_tensor().unwrap();

        let op_name = "Add";
        let mut op = Op::new(&ctx, op_name).unwrap();

        // Required input arguments
        op.add_input(&h_x).unwrap();
        op.add_input(&h_y).unwrap();

        // Execute Op
        const WRONG_NUMBER_OF_OUTPUTS: usize = 2;
        let res = op.execute::<WRONG_NUMBER_OF_OUTPUTS>(&ctx);
        assert!(res.is_err());
    }

    #[test]
    fn test_add_ut() {
        let values = [1i32, 2, 3, 4];
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let x = Tensor::new(&[2, 2]).with_values(&values).unwrap().freeze();
        let h_x = TensorHandle::new(&ctx, &x).unwrap();
        let h_y = h_x.copy_sharing_tensor().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 4, 6, 8]).unwrap();

        // tensor and tensor
        let h_z = add_ut(&ctx, &x, &x).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // tensor and handle
        let h_z = add_ut(&ctx, &x, &h_y).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // handle and tensor
        let h_z = add_ut(&ctx, &h_x, &x).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // handle and handle
        let h_z = add_ut(&ctx, &h_x, &h_y).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);
    }

    #[test]
    fn test_raw_ops_add() {
        let values = [1i32, 2, 3, 4];
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let x = Tensor::new(&[2, 2]).with_values(&values).unwrap().freeze();
        let h_x = TensorHandle::new(&ctx, &x).unwrap();
        let h_y = h_x.copy_sharing_tensor().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 4, 6, 8]).unwrap();

        // tensor and tensor
        let h_z = add(&ctx, &x, &x).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // tensor and handle
        let h_z = add(&ctx, &x, &h_y).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // handle and tensor
        let h_z = add(&ctx, &h_x, &x).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);

        // handle and handle
        let h_z = add(&ctx, &h_x, &h_y).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        assert_eq!(z, expected);
    }

    #[test]
    fn test_raw_ops_concat() {
        let values = [1i32, 2, 3, 4];
        let ctx = Context::new(ContextOptions::new()).unwrap();
        // h = [[1, 2],
        //      [3, 4]]
        let h = Tensor::new(&[2, 2])
            .with_values(&values)
            .unwrap()
            .into_handle(&ctx)
            .unwrap();

        // concat along axis 0
        let h_z = concat_v2(&ctx, &[&h, &h], &Tensor::from(0i32).freeze()).unwrap();
        // [[1, 2],
        //  [3, 4],
        //  [1, 2],
        //  [3, 4]]
        let z = h_z.resolve::<i32>().unwrap();

        let expected = Tensor::new(&[4, 2])
            .with_values(&[1i32, 2, 3, 4, 1, 2, 3, 4])
            .unwrap();
        assert_eq!(z, expected);

        // concat along axis 1
        let h_z = concat_v2(&ctx, &[&h, &h], &Tensor::from(1i32).freeze()).unwrap();
        // [[1, 2, 1, 2],
        //  [3, 4, 3, 4]]
        let z = h_z.resolve::<i32>().unwrap();

        let expected = Tensor::new(&[2, 4])
            .with_values(&[1i32, 2, 1, 2, 3, 4, 3, 4])
            .unwrap();
        assert_eq!(z, expected);
    }

    fn test_add_tensor_and_others() {
        let values = [1i32, 2, 3, 4];
        let ctx = Context::new(ContextOptions::new()).unwrap();

        // h = [[1, 2],
        //      [3, 4]]
        let h = Tensor::new(&[2, 2])
            .with_values(&values)
            .unwrap()
            .into_handle(&ctx)
            .unwrap();

        // tensor and scalar, braodcast
        //  [[2, 3],  = [[1, 2],  + 1
        //   [4, 5]]     [3, 4]]
        let h_z = add(&ctx, &h, &1).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 3, 4, 5]).unwrap();
        assert_eq!(z, expected);

        // tensor and array, broadcast
        //  [[2, 3],  = [[1, 2],  + [1]
        //   [4, 5]]     [3, 4]]
        let h_z = add(&ctx, &h, &[1]).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 3, 4, 5]).unwrap();
        assert_eq!(z, expected);

        // handle and array (horizontal vector), broadcst
        //  [[2, 4],  = [[1, 2],  + [1, 2]
        //   [4, 6]]     [3, 4]]
        let h_z = add(&ctx, &h, &[1, 2]).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 4, 4, 6]).unwrap();
        assert_eq!(z, expected);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_add_tensor_and_ndarray() {
        let values = [1i32, 2, 3, 4];
        let ctx = Context::new(ContextOptions::new()).unwrap();

        // h = [[1, 2],
        //      [3, 4]]
        let h = Tensor::new(&[2, 2])
            .with_values(&values)
            .unwrap()
            .into_handle(&ctx)
            .unwrap();

        // tensor and scalar, braodcast
        //  [[2, 3],  = [[1, 2],  + 1
        //   [4, 5]]     [3, 4]]
        let h_z = add(&ctx, &h, &array![1]).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 3, 4, 5]).unwrap();
        assert_eq!(z, expected);

        // tensor and array, broadcast
        //  [[2, 3],  = [[1, 2],  + 1
        //   [4, 5]]     [3, 4]]
        let h_z = add(&ctx, &h, &array![[1]]).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 3, 4, 5]).unwrap();
        assert_eq!(z, expected);

        // handle and array (horizontal vector), broadcst
        //  [[2, 4],  = [[1, 2],  + [1, 2]
        //   [4, 6]]     [3, 4]]
        let h_z = add(&ctx, &h, &array![1, 2]).unwrap();
        let z = h_z.resolve::<i32>().unwrap();
        let expected = Tensor::new(&[2, 2]).with_values(&[2i32, 4, 4, 6]).unwrap();
        assert_eq!(z, expected);
    }

    #[cfg(feature = "tensorflow_gpu")]
    #[test]
    #[ignore]
    fn test_add_gpu() {
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
            .unwrap()
            .freeze();
        let h = TensorHandle::new(&ctx, &x).unwrap();
        // Copy to GPU. This creates a new handle managed by the context `ctx`.
        let h_gpu = h.copy_to_device(&ctx, target_device).unwrap();

        let op_name = "Add";
        let mut op = Op::new(&ctx, op_name).unwrap();

        // Required input arguments
        op.add_input(&h).unwrap();
        op.add_input(&h_gpu).unwrap();
        op.set_device(target_device).unwrap();

        let [h_z_gpu] = op.execute(&ctx).unwrap();
        assert!(&h_z_gpu.device_name().unwrap() == target_device);

        let z = h_z_gpu.resolve::<f32>().unwrap();
        let expected = [2.0f32, 4.0, 6.0, 8.0];
        for (v0, v1) in z.iter().zip(&expected) {
            assert!((v0 - v1).abs() < f32::EPSILON);
        }
    }
}
