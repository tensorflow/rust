use std::ffi::{CStr, CString};
use std::marker::PhantomData;

use tensorflow_sys as tf;

use crate::eager::Context;
use crate::{AnyTensor, DataType, Result, Status, Tensor, TensorType};

/// A handle to a tensor on a device.
///
/// Constructing a TensorHandle requires a reference to an execute context so that the
/// generated handle will not out live the context.
/// ```
/// # use tensorflow::Tensor;
/// use tensorflow::eager::{ContextOptions, Context, TensorHandle};
///
/// let opts = ContextOptions::new();
/// let ctx = Context::new(opts).unwrap();
///
/// let t = Tensor::from(&[3i32]);
/// let h = TensorHandle::new(&ctx, &t).unwrap();
/// let v: Tensor<i32> = h.resolve().unwrap();
/// assert_eq!(&v[..], &[3i32]);
/// ```
///
/// Since TensorHandle cannot be alive beyond the lifetime of the context, the following
/// code will not compile.
/// ```compile_fail
/// # use tensorflow::Tensor;
/// use tensorflow::eager::{ContextOptions, Context, TensorHandle};
///
/// let h = {
///     let opts = ContextOptions::new();
///     let ctx = Context::new(opts).unwrap();
///
///     let t = Tensor::from(&[3i32]);
///     let h = TensorHandle::new(&ctx, &t).unwrap();
///     h
/// };
/// ```
#[derive(Debug)]
pub struct TensorHandle<'a> {
    inner: *mut tf::TFE_TensorHandle,
    // TensorHandle should not live longer than a given context.
    ctx: PhantomData<&'a Context>,
}

impl<'a> Drop for TensorHandle<'a> {
    fn drop(&mut self) {
        unsafe {
            tf::TFE_DeleteTensorHandle(self.inner);
        }
    }
}

impl<'a> TensorHandle<'a> {
    /// Crate a TensorHandle from Tensor
    pub fn new<T: TensorType>(_ctx: &'a Context, t: &Tensor<T>) -> Result<TensorHandle<'a>> {
        let status = Status::new();
        let inner = unsafe { tf::TFE_NewTensorHandle(t.inner()?, status.inner) };

        if inner.is_null() {
            Err(status)
        } else {
            Ok(TensorHandle {
                inner,
                ctx: PhantomData,
            })
        }
    }

    /// Return the DataType that corresponds to this type.
    pub fn data_type(&self) -> DataType {
        unsafe { DataType::from_c(tf::TFE_TensorHandleDataType(self.inner)) }
    }

    /// Return the number of dimensions.
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn num_dims(&self) -> Result<i32> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleNumDims(self.inner, status.inner) };
        if status.is_ok() {
            Ok(num_dims)
        } else {
            Err(status)
        }
    }

    /// Return the number of elements
    pub fn num_elements(&self) -> Result<i64> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleNumElements(self.inner, status.inner) };
        if status.is_ok() {
            Ok(num_dims)
        } else {
            Err(status)
        }
    }

    /// Return the number of elements for a given dim_index.
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn dim(&self, dim_index: i32) -> Result<i64> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleDim(self.inner, dim_index, status.inner) };
        if status.is_ok() {
            Ok(num_dims)
        } else {
            Err(status)
        }
    }

    /// Return the device of the operation that produced `h`. If `h` was produced by
    /// a copy, returns the destination device of the copy. Note that the returned
    /// device name is not always the device holding the tensor handle's memory. If
    /// you want the latter, use backing_device_name. This function will block till
    /// the operation that produces `h` has completed.
    pub fn device_name(&self) -> Result<String> {
        let status = Status::new();
        unsafe {
            let device_name = tf::TFE_TensorHandleDeviceName(self.inner, status.inner);
            if status.is_ok() {
                Ok(CStr::from_ptr(device_name).to_str()?.to_string())
            } else {
                Err(status)
            }
        }
    }

    /// Returns the name of the device in whose memory `h` resides.
    /// This function will block till the operation that produces `h` has completed.
    pub fn backing_device_name(&self) -> Result<String> {
        let status = Status::new();
        unsafe {
            let device_name = tf::TFE_TensorHandleBackingDeviceName(self.inner, status.inner);
            if status.is_ok() {
                Ok(CStr::from_ptr(device_name).to_str()?.to_string())
            } else {
                Err(status)
            }
        }
    }

    /// Return a new TensorHandle that shares the underlying tensor with `h`.
    pub fn copy_sharing_tensor(&self) -> Result<Self> {
        let status = Status::new();
        let inner = unsafe { tf::TFE_TensorHandleCopySharingTensor(self.inner, status.inner) };
        if status.is_ok() {
            Ok(Self {
                inner,
                ctx: self.ctx,
            })
        } else {
            Err(status)
        }
    }

    /// This function will block till the operation that produces `h` has
    /// completed. The memory returned might alias the internal memory used by
    /// TensorFlow. Hence, callers should not mutate this memory.
    pub fn resolve<T: TensorType>(&self) -> Result<Tensor<T>> {
        let mut status = Status::new();
        let tf_tensor = unsafe { tf::TFE_TensorHandleResolve(self.inner, status.inner) };
        if !status.is_ok() {
            return Err(status);
        }
        if self.data_type() != T::data_type() {
            let msg = format!(
                "The expected data type ({}) and underlying data type ({}) did not match.",
                T::data_type(),
                self.data_type()
            );

            status.set_lossy(crate::Code::InvalidArgument, &msg);
            return Err(status);
        }

        // Safely unwrap since data_type was checked beforehand.
        unsafe { Ok(Tensor::from_tf_tensor(tf_tensor).unwrap()) }
    }

    /// Create a new TensorHandle with the same contents as 'h' but placed
    /// in the memory of the device name 'device_name'.
    /// If source and destination are the same device, then this creates a new handle
    /// that shares the underlying buffer. Otherwise, it currently requires at least
    /// one of the source or destination devices to be CPU (i.e., for the source or
    /// destination tensor to be placed in host memory).
    /// If async execution is enabled, the copy may be enqueued and the call will
    /// return "non-ready" handle. Else, this function returns after the copy has
    /// been done.
    pub fn copy_to_device(&self, ctx: &Context, device_name: &str) -> Result<Self> {
        let status = Status::new();

        let device_name = CString::new(device_name)?;
        unsafe {
            let inner = tf::TFE_TensorHandleCopyToDevice(
                self.inner,
                ctx.inner,
                device_name.as_ptr(),
                status.inner,
            );

            if status.is_ok() {
                Ok(Self {
                    inner,
                    ctx: PhantomData,
                })
            } else {
                Err(status)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eager::ContextOptions;

    #[test]
    fn test_tensor_handle() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let t = Tensor::new(&[2, 3])
            .with_values(&[0_i32, 1, 2, 3, 4, 5])
            .unwrap();
        let h = TensorHandle::new(&ctx, &t).unwrap();

        assert_eq!(h.data_type(), DataType::Int32);
        assert_eq!(h.num_elements().unwrap(), 6);
        assert_eq!(h.num_dims().unwrap(), 2);
        assert_eq!(h.dim(0).unwrap(), 2);
        assert_eq!(h.dim(1).unwrap(), 3);
    }

    #[test]
    fn test_copy_sharing_tensor() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let t = Tensor::new(&[2, 3])
            .with_values(&[0_i32, 1, 2, 3, 4, 5])
            .unwrap();
        let h = TensorHandle::new(&ctx, &t).unwrap();
        let h_copy = h.copy_sharing_tensor().unwrap();
        let t2 = h_copy.resolve::<i32>().unwrap();

        // t and t2 may share the same memory, but it's difficuly to check
        // since the `resolve` does not guarantee that.
        assert_eq!(&t[..], &t2[..]);
    }

    #[cfg(feature = "tensorflow_gpu")]
    #[test]
    fn test_copy_to_device() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();
        let devices = ctx.device_list().unwrap();
        if !(devices.iter().any(|d| d.device_type == "GPU")) {
            eprintln!("Skipping test_copy_to_device because no GPU device is found");
            return;
        }

        let t = Tensor::new(&[2, 2]).with_values(&[0_i32, 1, 2, 3]).unwrap();
        let h = TensorHandle::new(&ctx, &t).unwrap();
        let target_device = "/job:localhost/replica:0/task:0/device:GPU:0";
        let h_gpu = h.copy_to_device(&ctx, target_device).unwrap();
        assert_eq!(h_gpu.device_name().unwrap(), target_device);
        let t2 = h_gpu.resolve::<i32>().unwrap();

        assert_eq!(&t[..], &t2[..]);
    }
}
