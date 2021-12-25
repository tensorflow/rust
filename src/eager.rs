//! C API extensions to experiment with eager execution of kernels.
//!
//! WARNING: The underlying C-API for the eager execution is not guaranteed to be
//! stable and can be changed without notice, which could result in breaking.

mod context;
pub use context::*;

mod tensor_handle;
pub use tensor_handle::*;

mod op;

pub use op::raw_ops;

use crate::{Result, Tensor, TensorType};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, Data, Dimension};

/// Simple helper trait to convert a Tensor into a TensorHandle for use in eager
/// execution.
pub trait ToTensorHandle<'a> {
    /// Convert the Tensor or TensorHandle into a new TensorHandle.
    ///
    /// _Warning_ : This function may create multiple handles to the same
    /// underlying tensor. Users should be careful not to modify the tensor
    /// after converting it to a handle. Also, users should be careful not
    /// to modify the tensor obtained via the TensorHandle's `resolve` method.
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>>;
}

impl<'a, T> ToTensorHandle<'a> for Tensor<T>
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

impl<'a, T: TensorType> ToTensorHandle<'a> for T {
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        let mut tensor = Tensor::<T>::new(&[]);
        tensor[0] = self.clone();
        TensorHandle::new(ctx, &tensor)
    }
}

impl<'a, T: TensorType> ToTensorHandle<'a> for [T] {
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        let mut tensor = Tensor::<T>::new(&[self.len() as u64]);
        for (e, v) in tensor.iter_mut().zip(self) {
            e.clone_from(v);
        }
        TensorHandle::new(ctx, &tensor)
    }
}

impl<'a, T: TensorType, const N: usize> ToTensorHandle<'a> for [T; N] {
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        let tensor = Tensor::from(&self[..]);
        TensorHandle::new(ctx, &tensor)
    }
}

#[cfg(feature = "ndarray")]
/// Convert any ndarray::ArrayBase type into a TensorHandle
impl<'a, T, S, D> ToTensorHandle<'a> for ArrayBase<S, D>
where
    T: TensorType,
    S: Data<Elem = T>,
    D: Dimension,
{
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        let dims: Vec<u64> = self.shape().iter().map(|x| *x as u64).collect();
        let mut tensor: Tensor<T> = Tensor::new(&dims);
        for (e, v) in tensor.iter_mut().zip(self.iter()) {
            e.clone_from(v);
        }
        TensorHandle::new(ctx, &tensor)
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
        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]);
        let handle = tensor.to_handle(&ctx).unwrap();
        assert_eq!(handle.num_dims().unwrap(), 4);
        assert_eq!(handle.dim(0).unwrap(), 1);
        assert_eq!(handle.dim(1).unwrap(), 2);
        assert_eq!(handle.dim(2).unwrap(), 3);
        assert_eq!(handle.dim(3).unwrap(), 4);

        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]);
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
        let tensor = Tensor::<i32>::new(&[1, 2, 3, 4]);
        let handle = tensor.to_handle(&ctx).unwrap();
        let handle2 = handle.to_handle(&ctx).unwrap();
        let tensor2: Tensor<i32> = handle2.resolve().unwrap();
        assert_eq!(&tensor[..], &tensor2[..]);
    }

    #[test]
    fn tensortype_to_handle() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let v = 1i32;
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<i32> = handle.resolve().unwrap();
        assert_eq!(&[v], &tensor[..]);

        let v = 1i64;
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<i64> = handle.resolve().unwrap();
        assert_eq!(&[v], &tensor[..]);

        let v = 1f32;
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<f32> = handle.resolve().unwrap();
        assert_eq!(&[v], &tensor[..]);

        let v = 1f64;
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<f64> = handle.resolve().unwrap();
        assert_eq!(&[v], &tensor[..]);
    }

    #[test]
    fn tarray_to_handle() {
        let ctx = Context::new(ContextOptions::new()).unwrap();
        let v = [1i32, 2, 3, 4];
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<i32> = handle.resolve().unwrap();
        assert_eq!(&v, &tensor[..]);

        let v = [1i64, 2, 3, 4];
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<i64> = handle.resolve().unwrap();
        assert_eq!(&v, &tensor[..]);

        let v = [1f32, 2.0, 3.0, 4.0];
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<f32> = handle.resolve().unwrap();
        assert_eq!(&v, &tensor[..]);

        let v = [1f64, 2.0, 3.0, 4.0];
        let handle = v.to_handle(&ctx).unwrap();
        let tensor: Tensor<f64> = handle.resolve().unwrap();
        assert_eq!(&v, &tensor[..]);
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
