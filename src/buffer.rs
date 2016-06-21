// -*- tab-width: 2 -*-

use libc;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::cmp;
use std::ffi::CStr;
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

/// Fixed-length heap-allocated vector.
/// This is basically a `Box<[T]>`, except that that type can't actually be constructed.
/// Furthermore, `[T; N]` can't be constructed if N is not a compile-time constant.
#[derive(Debug)]
pub struct Buffer<T> {
  ptr: *mut T,
  length: usize,
  owned: bool,
}

impl<T: Default> Buffer<T> {
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
}

impl<T> Buffer<T> {
  /// Creates a new uninitialized buffer.
  ///
  /// `len` is the number of elements.
  /// The caller is responsible for initializing the data.
  pub unsafe fn new_uninitialized(len: usize) -> Self {
    let elem_size = mem::size_of::<T>();
    let alloc_size = len * elem_size;
    let align = cmp::max(mem::align_of::<T>(), mem::size_of::<*const libc::c_void>());
    // posix_memalign requires the alignment to be at least sizeof(void*).
    // TODO: Use alloc::heap::allocate once it's stable, or at least libc::aligned_alloc once it exists
    let mut ptr = ptr::null::<libc::c_void>() as *mut libc::c_void;
    let err = libc::posix_memalign(&mut ptr, align, alloc_size);
    if err != 0 {
      let c_msg = libc::strerror(err);
      let msg = CStr::from_ptr(c_msg);
      panic!("Failed to allocate: {}", msg.to_str().unwrap());
    }
    Buffer {
      ptr: ptr as *mut T,
      length: len,
      owned: true,
    }
  }

  /// Creates a buffer from data owned by the C API.
  ///
  /// `len` is the number of elements.
  /// The underlying data is *not* freed when the buffer is destroyed.
  pub unsafe fn from_ptr(ptr: *mut T, len: usize) -> Self {
    Buffer {
      ptr: ptr,
      length: len,
      owned: false,
    }
  }
}

impl<T> Drop for Buffer<T> {
  fn drop(&mut self) {
    if self.owned {
      // It would be nice to skip the loop entirely if std::intrinsics::needs_drop() returns false,
      // but it's not stable, and the docs say it probably never will be.
      // Note that checking to see if T implements Drop is not sufficient,
      // because T may not implement Drop, but may contain a type that does.
      unsafe {
        for x in self.iter_mut() {
          // TODO: use std::ptr::drop_in_place
          mem::drop(mem::replace(x, mem::uninitialized()));
        }
        libc::free(self.ptr as *mut libc::c_void);
      }
    }
  }
}

impl<T> AsRef<[T]> for Buffer<T> {
  #[inline]
  fn as_ref(&self) -> &[T] {
    unsafe {
      slice::from_raw_parts(self.ptr, self.length)
    }
  }
}

impl<T> AsMut<[T]> for Buffer<T> {
  #[inline]
  fn as_mut(&mut self) -> &mut [T] {
    unsafe {
      slice::from_raw_parts_mut(self.ptr, self.length)
    }
  }
}

impl<T> Deref for Buffer<T> {
  type Target = [T];

  #[inline]
  fn deref(&self) -> &[T] {
    self.as_ref()
  }
}

impl<T> DerefMut for Buffer<T> {
  #[inline]
  fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
    self.as_mut()
  }
}

impl<T> Borrow<[T]> for Buffer<T> {
  #[inline]
  fn borrow(&self) -> &[T] {
    self.as_ref()
  }
}

impl<T> BorrowMut<[T]> for Buffer<T> {
  #[inline]
  fn borrow_mut(&mut self) -> &mut [T] {
    self.as_mut()
  }
}

impl<T: Clone> Clone for Buffer<T> where T: Clone {
  #[inline]
  fn clone(&self) -> Buffer<T> {
    let mut b = unsafe {
      Buffer::new_uninitialized(self.length)
    };
    // TODO: Use std::ptr::copy for primitives once we have impl specialization
    for i in 0..self.length {
      b[i] = self[i].clone();
    }
    b
  }

  #[inline]
  fn clone_from(&mut self, other: &Buffer<T>) {
    assert!(self.length == other.length, "self.length = {}, other.length = {}", self.length, other.length);
    // TODO: Use std::ptr::copy for primitives once we have impl specialization
    for i in 0..self.length {
      self[i] = other[i].clone();
    }
  }
}

impl<T> Index<usize> for Buffer<T> {
  type Output = T;

  #[inline]
  fn index(&self, index: usize) -> &T {
    assert!(index < self.length, "index = {}, length = {}", index, self.length);
    unsafe {
      &*self.ptr.offset(index as isize)
    }
  }
}

impl<T> IndexMut<usize> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut T {
    assert!(index < self.length, "index = {}, length = {}", index, self.length);
    unsafe {
      &mut *self.ptr.offset(index as isize)
    }
  }
}

impl<T> Index<Range<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: Range<usize>) -> &[T] {
    assert!(index.start <= index.end, "index.start = {}, index.end = {}", index.start, index.end);
    assert!(index.end <= self.length, "index.end = {}, length = {}", index.end, self.length);
    unsafe {
      slice::from_raw_parts(&*self.ptr.offset(index.start as isize), index.len())
    }
  }
}

impl<T> IndexMut<Range<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
    assert!(index.start <= index.end, "index.start = {}, index.end = {}", index.start, index.end);
    assert!(index.end <= self.length, "index.end = {}, length = {}", index.end, self.length);
    unsafe {
      slice::from_raw_parts_mut(&mut *self.ptr.offset(index.start as isize), index.len())
    }
  }
}

impl<T> Index<RangeTo<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: RangeTo<usize>) -> &[T] {
    assert!(index.end <= self.length, "index.end = {}, length = {}", index.end, self.length);
    unsafe {
      slice::from_raw_parts(&*self.ptr, index.end)
    }
  }
}

impl<T> IndexMut<RangeTo<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
    assert!(index.end <= self.length, "index.end = {}, length = {}", index.end, self.length);
    unsafe {
      slice::from_raw_parts_mut(&mut *self.ptr, index.end)
    }
  }
}

impl<T> Index<RangeFrom<usize>> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, index: RangeFrom<usize>) -> &[T] {
    assert!(index.start <= self.length, "index.start = {}, length = {}", index.start, self.length);
    unsafe {
      slice::from_raw_parts(&*self.ptr.offset(index.start as isize), self.length - index.start)
    }
  }
}

impl<T> IndexMut<RangeFrom<usize>> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
    assert!(index.start <= self.length, "index.start = {}, length = {}", index.start, self.length);
    unsafe {
      slice::from_raw_parts_mut(&mut *self.ptr.offset(index.start as isize), self.length - index.start)
    }
  }
}

impl<T> Index<RangeFull> for Buffer<T> {
  type Output = [T];

  #[inline]
  fn index(&self, _: RangeFull) -> &[T] {
    unsafe {
      slice::from_raw_parts(&*self.ptr, self.length)
    }
  }
}

impl<T> IndexMut<RangeFull> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
    unsafe {
      slice::from_raw_parts_mut(&mut *self.ptr, self.length)
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
