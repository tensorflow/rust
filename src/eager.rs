//! C API extensions to experiment with eager execution of kernels.
//!
//! WARNING: The underlying C-API for the eager execution is not guaranteed to be
//! stable and can be changed without notice, which could result in breaking.

mod context;
pub use context::*;

mod readonly_tensor;
pub use readonly_tensor::*;

mod tensor_handle;
pub use tensor_handle::*;

mod op;

pub use op::raw_ops;

use crate::{Result, Tensor, TensorType};

impl<T: TensorType> Tensor<T> {
    /// Convert a Tensor to a readonly Tensor for use in eager execution.
    pub fn freeze(self) -> ReadonlyTensor<T> {
        ReadonlyTensor {
            inner: self.inner,
            dims: self.dims,
        }
    }
}

/// A helper trait to convert a Tensor or some other types into a TensorHandle
/// for use in eager execution.
pub trait ToTensorHandle<'a> {
    /// Convert a Tensor or values into a new TensorHandle.
    ///
    /// _Warning_ : This function may create multiple handles to the same
    /// underlying tensor. Users should be careful not to modify the tensor
    /// after converting it to a TensorHandle. Also, users should be careful
    /// not to modify the Tensor generated from the TensorHandle's `resolve`
    /// method.
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>>;
}

impl<'a, T> ToTensorHandle<'a> for ReadonlyTensor<T>
where
    T: TensorType,
{
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        TensorHandle::new(ctx, self)
    }
}

impl<'a> ToTensorHandle<'a> for TensorHandle<'a> {
    fn to_handle(&self, _: &'a Context) -> Result<TensorHandle<'a>> {
        self.copy_sharing_tensor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eager::{Context, ContextOptions};
    use crate::Tensor;

    #[cfg(feature = "ndarray")]
    use ndarray::array;

    #[test]
    fn tensor_to_handle() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]).freeze();
        let handle = tensor.to_handle(&ctx).unwrap();
        assert_eq!(handle.num_dims().unwrap(), 4);
        assert_eq!(handle.dim(0).unwrap(), 1);
        assert_eq!(handle.dim(1).unwrap(), 2);
        assert_eq!(handle.dim(2).unwrap(), 3);
        assert_eq!(handle.dim(3).unwrap(), 4);

        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]).freeze();
        let handle = &tensor.to_handle(&ctx).unwrap();
        assert_eq!(handle.num_dims().unwrap(), 4);
        assert_eq!(handle.dim(0).unwrap(), 1);
        assert_eq!(handle.dim(1).unwrap(), 2);
        assert_eq!(handle.dim(2).unwrap(), 3);
        assert_eq!(handle.dim(3).unwrap(), 4);
    }

    #[test]
    fn handle_to_handle() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]).freeze();
        let handle = tensor.to_handle(&ctx).unwrap();
        let handle2 = handle.to_handle(&ctx).unwrap();
        let tensor2 = handle2.resolve::<i32>().unwrap();
        assert_eq!(tensor, tensor2);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn ndarray_to_handle() {
        let a = array![[1i32, 2], [3, 4]];

        let ctx = Context::new(ContextOptions::new()).unwrap();
        let handle = a.to_handle(&ctx).unwrap();
        let tensor: Tensor<i32> = handle.resolve().unwrap();
        assert_eq!(a.as_slice().unwrap(), &tensor[..]);

        let at = Tensor::from(a);
        assert_eq!(at, tensor);
    }
}
