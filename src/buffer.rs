// -*- tab-width: 2 -*-

extern crate tensorflow_sys as tf;

use libc;
use libc::c_void;
use libc::size_t;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::cmp;
use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;
use std::ops::RangeFrom;
use std::ops::RangeFull;
use std::ops::RangeTo;
use std::ptr;
use std::slice;
use super::BufferTrait;
use super::TensorType;

/// Fixed-length heap-allocated vector.
/// This is basically a `Box<[T]>`, except that that type can't actually be constructed.
/// Furthermore, `[T; N]` can't be constructed if N is not a compile-time constant.
#[derive(Debug)]
pub struct Buffer<T: TensorType> {
  inner: *mut tf::TF_Buffer,
  owned: bool,
  phantom: PhantomData<T>,
}

impl<T: TensorType> Buffer<T> {
  /// Creates a new buffer initialized to zeros.
  ///
  /// `len` is the number of elements.
  pub fn new(len: usize) -> Self {
    let mut b = unsafe {
      Buffer::new_uninitialized(len)
    };
    // TODO: Use libc::memset for primitives once we have impl specialization and memset is included
    for i in 0..len {
      b[i] = T::default();
    }
    b
  }

  /// Creates a new uninitialized buffer.
  ///
  /// `len` is the number of elements.
  /// The caller is responsible for initializing the data.
  pub unsafe fn new_uninitialized(len: usize) -> Self {
    let inner = tf::TF_NewBuffer();
    let elem_size = mem::size_of::<T>();
    let alloc_size = len * elem_size;
    let align = cmp::max(mem::align_of::<T>(), mem::size_of::<*const c_void>());
    // posix_memalign requires the alignment to be at least sizeof(void*).
    // TODO: Use alloc::heap::allocate once it's stable, or at least libc::aligned_alloc once it exists
    let mut ptr = ptr::null::<c_void>() as *mut c_void;
    let err = libc::posix_memalign(&mut ptr, align, alloc_size);
    if err != 0 {
      let c_msg = libc::strerror(err);
      let msg = CStr::from_ptr(c_msg);
      panic!("Failed to allocate: {}", msg.to_str().unwrap());
    }
    (*inner).data = ptr;
    (*inner).length = len;
    (*inner).data_deallocator = Some(deallocator);
    Buffer {
      inner: inner,
      owned: true,
      phantom: PhantomData,
    }
  }

  /// Creates a buffer from data owned by the C API.
  ///
  /// `len` is the number of elements.
  /// The underlying data is *not* freed when the buffer is destroyed.
  pub unsafe fn from_ptr(ptr: *mut T, len: usize) -> Self {
    let inner = tf::TF_NewBuffer();
    (*inner).data = ptr as *const c_void;
    (*inner).length = len;
    Buffer {
      inner: inner,
      owned: false,
      phantom: PhantomData,
    }
  }

  /// Consumes the buffer and returns the data.
  ///
  /// The caller is responsible for freeing the data.
  pub unsafe fn into_ptr(mut self) -> (*mut T, usize) {
    // TODO: remove
    // This flag is used by drop.
    self.owned = false;
    (self.data_mut(), self.length())
  }

  /// Returns a buffer with a null pointer.
  pub unsafe fn null() -> Self {
    Buffer {
      inner: ptr::null::<tf::TF_Buffer>() as *mut _,
      owned: false,
      phantom: PhantomData,
    }
  }

  #[inline]
  fn data(&self) -> *const T {
    unsafe {
      (*self.inner).data as *const T
    }
  }

  #[inline]
  fn data_mut(&mut self) -> *mut T {
    unsafe {
      (*self.inner).data as *mut T
    }
  }

  #[inline]
  fn length(&self) -> usize {
    unsafe {
      (*self.inner).length
    }
  }

  /// Creates a buffer from data owned by the C API.
  ///
  /// `len` is the number of elements.
  /// The underlying data is freed when the buffer is destroyed if `owned` is true.
  pub unsafe fn from_c(buf: *mut tf::TF_Buffer, owned: bool) -> Self {
    Buffer {
      inner: buf,
      owned: owned,
      phantom: PhantomData,
    }
  }
}

impl<T: TensorType> BufferTrait for Buffer<T> {
  fn is_owned(&self) -> bool {
    self.owned
  }

  fn set_owned(&mut self, owned: bool) {
    self.owned = owned;
  }

  fn inner(&self) -> *const tf::TF_Buffer {
    self.inner
  }

  fn inner_mut(&mut self) -> *mut tf::TF_Buffer {
    self.inner
  }
}

unsafe extern "C" fn deallocator(data: *mut c_void, _length: size_t) {
  libc::free(data);
}

impl<T: TensorType> Drop for Buffer<T> {
  fn drop(&mut self) {
    if self.owned {
      unsafe {
        tf::TF_DeleteBuffer(self.inner);
      }
    }
  }
}

impl<T: TensorType> AsRef<[T]> for Buffer<T> {
  #[inline]
  fn as_ref(&self) -> &[T] {
    unsafe {
      slice::from_raw_parts(self.data(), (*self.inner).length)
    }
  }
}

impl<T: TensorType> AsMut<[T]> for Buffer<T> {
  #[inline]
  fn as_mut(&mut self) -> &mut [T] {
    unsafe {
      slice::from_raw_parts_mut(self.data_mut(), (*self.inner).length)
    }
  }
}

impl<T: TensorType> Deref for Buffer<T> {
  type Target = [T];

  #[inline]
  fn deref(&self) -> &[T] {
    self.as_ref()
  }
}

impl<T: TensorType> DerefMut for Buffer<T> {
  #[inline]
  fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
    self.as_mut()
  }
}

impl<T: TensorType> Borrow<[T]> for Buffer<T> {
  #[inline]
  fn borrow(&self) -> &[T] {
    self.as_ref()
  }
}

impl<T: TensorType> BorrowMut<[T]> for Buffer<T> {
  #[inline]
  fn borrow_mut(&mut self) -> &mut [T] {
    self.as_mut()
  }
}

impl<T: TensorType> Clone for Buffer<T> where T: Clone {
  #[inline]
  fn clone(&self) -> Buffer<T> {
    let mut b = unsafe {
      Buffer::new_uninitialized((*self.inner).length)
    };
    // TODO: Use std::ptr::copy for primitives once we have impl specialization
    for i in 0..self.length() {
      b[i] = self[i].clone();
    }
    b
  }

  #[inline]
  fn clone_from(&mut self, other: &Buffer<T>) {
    assert!(self.length() == other.length(), "self.length() = {}, other.length() = {}", self.length(), other.length());
    // TODO: Use std::ptr::copy for primitives once we have impl specialization
    for i in 0..self.length() {
      self[i] = other[i].clone();
    }
  }
}

impl<T: TensorType> Index<usize> for Buffer<T> {
  type Output = T;

  #[inline]
  fn index(&self, index: usize) -> &T {
    assert!(index < self.length(), "index = {}, length = {}", index, self.length());
    unsafe {
      &*self.data().offset(index as isize)
    }
  }
}

impl<T: TensorType> IndexMut<usize> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut T {
    assert!(index < self.length(), "index = {}, length = {}", index, self.length());
    unsafe {
      &mut *self.data_mut().offset(index as isize)
    }
  }
}

impl<T: TensorType> Index<Range<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: Range<usize>) -> &[T] {
    assert!(index.start <= index.end, "index.start = {}, index.end = {}", index.start, index.end);
    assert!(index.end <= self.length(), "index.end = {}, length = {}", index.end, self.length());
    unsafe {
      slice::from_raw_parts(&*self.data().offset(index.start as isize), index.len())
    }
  }
}

impl<T: TensorType> IndexMut<Range<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
    assert!(index.start <= index.end, "index.start = {}, index.end = {}", index.start, index.end);
    assert!(index.end <= self.length(), "index.end = {}, length = {}", index.end, self.length());
    unsafe {
      slice::from_raw_parts_mut(&mut *self.data_mut().offset(index.start as isize), index.len())
    }
  }
}

impl<T: TensorType> Index<RangeTo<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: RangeTo<usize>) -> &[T] {
    assert!(index.end <= self.length(), "index.end = {}, length = {}", index.end, self.length());
    unsafe {
      slice::from_raw_parts(&*self.data(), index.end)
    }
  }
}

impl<T: TensorType> IndexMut<RangeTo<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
    assert!(index.end <= self.length(), "index.end = {}, length = {}", index.end, self.length());
    unsafe {
      slice::from_raw_parts_mut(&mut *self.data_mut(), index.end)
    }
  }
}

impl<T: TensorType> Index<RangeFrom<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: RangeFrom<usize>) -> &[T] {
    assert!(index.start <= self.length(), "index.start = {}, length = {}", index.start, self.length());
    unsafe {
      slice::from_raw_parts(&*self.data().offset(index.start as isize), self.length() - index.start)
    }
  }
}

impl<T: TensorType> IndexMut<RangeFrom<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
    assert!(index.start <= self.length(), "index.start = {}, length = {}", index.start, self.length());
    unsafe {
      slice::from_raw_parts_mut(&mut *self.data_mut().offset(index.start as isize), self.length() - index.start)
    }
  }
}

impl<T: TensorType> Index<RangeFull> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, _: RangeFull) -> &[T] {
    unsafe {
      slice::from_raw_parts(&*self.data(), self.length())
    }
  }
}

impl<T: TensorType> IndexMut<RangeFull> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
    unsafe {
      slice::from_raw_parts_mut(&mut *self.data_mut(), self.length())
    }
  }
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn basic() {
    let mut buf = Buffer::new(10);
    assert_eq!(buf.len(), 10);
    buf[0] = 1;
    assert_eq!(buf[0], 1);
  }
}
