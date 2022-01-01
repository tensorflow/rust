use libc::c_int;
use std::ops::Deref;
use tensorflow_sys as tf;

use crate::{AnyTensor, DataType, Result, Shape, Tensor, TensorInner, TensorType};

/// A read-only tensor.
///
/// This ReadonlyTensor is a wrapper around a `Tensor` that is read-only.
/// Some operations that mutate the underlying Tensor are not supported.
#[derive(Debug, Clone, Eq)]
pub struct ReadonlyTensor<T: TensorType> {
    pub(super) inner: T::InnerType,
    pub(super) dims: Vec<u64>,
}

impl<T: TensorType> AnyTensor for ReadonlyTensor<T> {
    fn inner(&self) -> Result<*mut tf::TF_Tensor> {
        self.inner.as_mut_ptr(&self.dims)
    }

    fn data_type(&self) -> DataType {
        T::data_type()
    }
}

impl<T: TensorType> Deref for ReadonlyTensor<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.inner.deref()
    }
}

impl<T: TensorType + PartialEq> PartialEq for ReadonlyTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
    }
}

impl<T: TensorType + PartialEq> PartialEq<Tensor<T>> for ReadonlyTensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
    }
}

impl<T: TensorType> ReadonlyTensor<T> {
    /// Get one single value from the Tensor.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// # use tensorflow::eager::ReadonlyTensor;
    /// let mut a = Tensor::<i32>::new(&[2, 3, 5]);
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// let a = a.freeze();
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[u64]) -> T {
        let index = self.get_index(indices);
        self[index].clone()
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// # use tensorflow::Tensor;
    /// # use tensorflow::eager::ReadonlyTensor;
    /// let a = Tensor::<f32>::new(&[3, 3, 3]).freeze();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[u64]) -> usize {
        assert!(self.dims.len() == indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index as usize
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Returns the tensor's dimensions as a Shape.
    pub fn shape(&self) -> Shape {
        Shape(Some(self.dims.iter().map(|d| Some(*d as i64)).collect()))
    }

    // Wraps a TF_Tensor. Returns None if types don't match.
    pub(super) unsafe fn from_tf_tensor(tensor: *mut tf::TF_Tensor) -> Option<Self> {
        let mut dims = Vec::with_capacity(tf::TF_NumDims(tensor) as usize);
        for i in 0..dims.capacity() {
            dims.push(tf::TF_Dim(tensor, i as c_int) as u64);
        }

        Some(Self {
            inner: T::InnerType::from_tf_tensor(tensor)?,
            dims,
        })
    }

    /// Convert back to a Tensor.
    ///
    /// Safety: This is unsafe because modifying the returned Tensor will modify the underlying memory,
    /// which may affect other Tensors that share the same memory.
    ///
    /// ```
    /// # use tensorflow::{Tensor, Result};
    /// # use tensorflow::eager::*;
    /// # fn main() -> Result<()> {
    /// let ctx = Context::new(ContextOptions::new())?;
    ///
    /// let a_readonly = Tensor::<i32>::new(&[2, 3, 5]).freeze();
    /// let h = a_readonly.to_handle(&ctx)?;
    /// let b_readonly = h.resolve::<i32>()?;
    ///
    /// let mut b = unsafe { b_readonly.into_tensor() };
    /// b[1*15 + 1*5 + 1] = 5;
    ///
    /// // Since a and b share the same memory, modifying b will also modify a.
    /// assert_eq!(a_readonly, b);
    /// # Ok(())
    /// # }
    /// ```
    pub unsafe fn into_tensor(self) -> Tensor<T> {
        Tensor {
            inner: self.inner,
            dims: self.dims,
        }
    }
}
