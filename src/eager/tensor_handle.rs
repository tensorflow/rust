use std::ffi::{CStr, CString};
use std::marker::PhantomData;

use tensorflow_sys as tf;

use crate::eager::{Context, ReadonlyTensor};
use crate::{AnyTensor, DataType, Result, Status, TensorType};

/// A handle to a tensor on a device.
///
/// Constructing a TensorHandle requires a reference to an execute context so that the
/// generated handle will not out live the context.
/// ```
/// # use tensorflow::{Result, Tensor};
/// use tensorflow::eager::*;
/// # fn main() -> Result<()> {
/// let opts = ContextOptions::new();
/// let ctx = Context::new(opts)?;
///
/// let t = Tensor::from(&[3i32]).freeze();
/// let h = TensorHandle::new(&ctx, &t)?;
/// let v = h.resolve::<i32>()?;
/// assert_eq!(&v[..], &[3i32]);
/// # Ok(())
/// # }
/// ```
///
/// TensorHandle manages the same buffer of the tensor. Users can destruct the Tensor
/// while leaving the TensorHandle. This is a valid use case for TensorHandle.
/// ```
/// # use tensorflow::{Result, Tensor};
/// use tensorflow::eager::*;
///
/// # fn main() -> Result<()> {
/// let opts = ContextOptions::new();
/// let ctx = Context::new(opts)?;
/// let h = {
///     let t = Tensor::from(&[3i32]).freeze();
///     TensorHandle::new(&ctx, &t)?
/// };
/// // At this point, the buffer is managed only by the handle.
/// # Ok(())
/// # }
/// ```
///
/// Since TensorHandle cannot be alive beyond the lifetime of the context, the following
/// code will not compile.
/// ```compile_fail
/// # use tensorflow::{Result, Tensor};
/// use tensorflow::eager::*;
///
/// # fn main() -> Result<()> {
/// let h = {
///     let opts = ContextOptions::new();
///     let ctx = Context::new(opts)?;
///
///     let t = Tensor::from(&[3i32]).freeze();
///     TensorHandle::new(&ctx, &t)?
/// };
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug)]
pub struct TensorHandle<'a> {
    pub(super) inner: *mut tf::TFE_TensorHandle,
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
    /// Create a TensorHandle from the input Tensor
    pub fn new<T: TensorType>(
        _ctx: &'a Context,
        t: &ReadonlyTensor<T>,
    ) -> Result<TensorHandle<'a>> {
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
    ///
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn num_dims(&self) -> Result<usize> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleNumDims(self.inner, status.inner) };
        if status.is_ok() {
            // num_dims >= 0 when the status is ok, so we can safely cast it to u64.
            Ok(num_dims as usize)
        } else {
            Err(status)
        }
    }

    /// Return the number of elements
    pub fn num_elements(&self) -> Result<u64> {
        let status = Status::new();
        let num_elements = unsafe { tf::TFE_TensorHandleNumElements(self.inner, status.inner) };
        if status.is_ok() {
            // num_elements >= 0 when the status is ok, so we can safely cast it to u64.
            Ok(num_elements as u64)
        } else {
            Err(status)
        }
    }

    /// Return the number of elements for a given dim_index.
    ///
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn dim(&self, dim_index: i32) -> Result<u64> {
        let status = Status::new();
        let dim = unsafe { tf::TFE_TensorHandleDim(self.inner, dim_index, status.inner) };
        if status.is_ok() {
            // dim >= 0 when the status is ok, so we can safely cast it to u64.
            Ok(dim as u64)
        } else {
            Err(status)
        }
    }

    /// Return the device of the operation that produced the current TensorHandle.
    ///
    /// If the TensorHandle was produced by a copy, returns the destination device of the copy.
    /// Note that the returned device name is not always the device holding the tensor handle's memory.
    /// If you want the latter, use backing_device_name.
    ///
    /// This function will block till the operation that produces the current TensorHandle has completed.
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

    /// Returns the name of the device in whose memory underlying the current TensorHandle resides.
    ///
    /// This function will block till the operation that produces the current TensorHandle has completed.
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

    /// Return a new TensorHandle that shares the underlying tensor with the current TensorHandle.
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

    /// This function will block till the operation that produces the current TensorHandle has completed.
    /// The memory returned might alias the internal memory used by TensorFlow.
    /// Hence, callers should not mutate this memory.
    pub fn resolve<T: TensorType>(&self) -> Result<ReadonlyTensor<T>> {
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
        unsafe { Ok(ReadonlyTensor::from_tf_tensor(tf_tensor).unwrap()) }
    }

    /// Create a new TensorHandle with the same contents as the current TensorHandle but placed
    /// in the memory of the device name 'device_name'.
    /// If source and destination are the same device, then this creates a new handle
    /// that shares the underlying buffer. Otherwise, it currently requires at least
    /// one of the source or destination devices to be CPU (i.e., for the source or
    /// destination tensor to be placed in host memory).
    /// If async execution is enabled, the copy may be enqueued and the call will
    /// return "non-ready" TensorHandle. Else, this function returns after the copy has
    /// been done.
    pub fn copy_to_device<'b>(
        &self,
        ctx: &'b Context,
        device_name: &str,
    ) -> Result<TensorHandle<'b>> {
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
                Ok(TensorHandle {
                    inner,
                    ctx: PhantomData,
                })
            } else {
                Err(status)
            }
        }
    }

    /// Convert the raw TFE_TensorHandle* into a TensorHandle.
    pub(super) unsafe fn from_tensor_handle(
        _ctx: &'a Context,
        inner: *mut tf::TFE_TensorHandle,
    ) -> Self {
        Self {
            inner,
            ctx: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eager::ContextOptions;
    use crate::Tensor;

    #[test]
    fn test_tensor_handle() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let t = Tensor::new(&[2, 3])
            .with_values(&[0_i32, 1, 2, 3, 4, 5])
            .unwrap()
            .freeze();
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
            .unwrap()
            .freeze();
        let h = TensorHandle::new(&ctx, &t).unwrap();
        let h_copy = h.copy_sharing_tensor().unwrap();
        let t2 = h_copy.resolve::<i32>().unwrap();

        // t and t2 may share the same memory, but it's difficuly to check
        // since the `resolve` does not guarantee that.
        assert_eq!(t, t2);
    }

    /// Following tests are disabled by default because it requires a GPU and some setup.
    ///
    /// To run this test, you need to pass the `-- --ignored` argument to cargo test.
    /// ```sh
    /// cargo test --features "eager tensorflow_gpu" -- --ignored
    /// ```
    #[cfg(feature = "tensorflow_gpu")]
    mod gpu {
        use super::*;
        use crate::eager::ContextOptions;

        #[test]
        #[ignore]
        fn test_copy_to_device() {
            let values = [0_i32, 1, 2, 3];

            let opts = ContextOptions::new();
            let ctx = Context::new(opts).unwrap();
            let devices = ctx.device_list().unwrap();
            let gpu_device = devices
                .iter()
                .find(|d| d.device_type == "GPU")
                .expect("No GPU device was found.");
            let target_device = &gpu_device.name;

            let t = Tensor::new(&[2, 2]).with_values(&values).unwrap().freeze();
            let h = TensorHandle::new(&ctx, &t).unwrap();
            let h_gpu = TensorHandle::copy_to_device(&h, &ctx, target_device).unwrap();
            assert_eq!(&h_gpu.device_name().unwrap(), target_device);
            let t2 = h_gpu.resolve::<i32>().unwrap();

            assert_eq!(&t[..], &t2[..]);
        }

        #[test]
        #[ignore]
        fn test_copy_to_device_lifetime() {
            let values = [0_i32, 1, 2, 3];

            let opts = ContextOptions::new();
            let ctx = Context::new(opts).unwrap();
            let devices = ctx.device_list().unwrap();
            let gpu_device = devices
                .iter()
                .find(|d| d.device_type == "GPU")
                .expect("No GPU device was found.");
            let target_device = &gpu_device.name;

            let h_gpu = {
                // Create a temporal Context
                let opts = ContextOptions::new();
                let ctx2 = Context::new(opts).unwrap();
                let t = Tensor::new(&[2, 2]).with_values(&values).unwrap().freeze();

                // Create a TensorHandle managed by the context `ctx2`.
                let h = TensorHandle::new(&ctx2, &t).unwrap();

                // Copy to GPU. This creates a new handle managed by the context `ctx`.
                h.copy_to_device(&ctx, target_device).unwrap()
            };
            assert_eq!(&h_gpu.device_name().unwrap(), target_device);
            let t2 = h_gpu.resolve::<i32>().unwrap();

            assert_eq!(&values[..], &t2[..]);
        }
    }
}
