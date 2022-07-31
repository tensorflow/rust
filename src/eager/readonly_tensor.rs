use crate::{
    write_tensor_recursive, AnyTensor, DataType, Result, Shape, Tensor, TensorInner, TensorType,
};
use core::fmt;
use fmt::{Debug, Formatter};
use libc::c_int;
use std::{fmt::Display, ops::Deref};
use tensorflow_sys as tf;

/// A read-only tensor.
///
/// ReadonlyTensor is a [`Tensor`](Tensor) that does not support mutation.
#[derive(Clone, Eq)]
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

impl<T: TensorType> Display for ReadonlyTensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> ::std::fmt::Result {
        let mut counter: i64 = match std::env::var("TF_RUST_DISPLAY_MAX") {
            Ok(e) => e.parse().unwrap_or(-1),
            Err(_) => -1,
        };
        write_tensor_recursive(f, self, self.dims(), &mut counter)
    }
}

impl<T: TensorType> Debug for ReadonlyTensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        crate::format_tensor(self, "ReadonlyTensor", self.dims(), f)
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
    /// let a: ReadonlyTensor<_> = a.freeze();
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
    /// let a: ReadonlyTensor<_> = Tensor::<f32>::new(&[3, 3, 3]).freeze();
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
        Shape::from(&self.dims[..])
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
    /// # Safety
    ///
    /// This is unsafe because modifying the returned Tensor will modify the underlying memory,
    /// which may affect other Tensors that share the same memory.
    ///
    /// ```
    /// # use tensorflow::{Tensor, Result};
    /// # use tensorflow::eager::*;
    /// # fn main() -> Result<()> {
    /// let ctx = Context::new(ContextOptions::new()).unwrap();
    /// let tensor = Tensor::from(0i32).freeze();
    /// let h = tensor.to_handle(&ctx).unwrap();
    ///
    /// let t0 = h.resolve::<i32>().unwrap();
    /// assert_eq!(t0[0], 0i32);
    ///
    /// // Manipulating the Tensor will affect the Tensor that shares underlying buffer.
    /// {
    ///     // Getting multiple times should return the same Tensor.
    ///     let t1 = h.resolve::<i32>().unwrap();
    ///
    ///     // Convert back from a TensorHandle to a Tensor.
    ///     let mut t1 = unsafe { t1.into_tensor() };
    ///     t1[0] = 5;
    /// }
    ///
    /// // Check that t0 shares the same underlying buffer with t1.
    /// // This is why we need to use unsafe.
    /// assert_eq!(t0[0], 5);
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
