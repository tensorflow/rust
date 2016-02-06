// -*- tab-width: 2 -*-

use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;
use std::ops::RangeFrom;
use std::ops::RangeFull;
use std::ops::RangeTo;

/// Fixed-length heap-allocated vector.
/// This is basically a Box<[T]>, except that that type can't actually be constructed.
/// Furthermore, [T; N] can't be constructed if N is not a compile-time constant.
#[derive(Debug)]
pub struct Buffer<T> {
  // alloc::raw_vec::RawVec would be nice to use internally, but it's unstable.
  data: Vec<T>,
}

impl<T: Default + Clone> Buffer<T> {
  pub fn new(len: usize) -> Self {
    Buffer {
      data: vec![T::default(); len],
    }
  }
}

impl<T> AsRef<[T]> for Buffer<T> {
  #[inline]
  fn as_ref(&self) -> &[T] {
    &self.data
  }
}

impl<T> AsMut<[T]> for Buffer<T> {
  #[inline]
  fn as_mut(&mut self) -> &mut [T] {
    &mut self.data
  }
}

impl<T> Deref for Buffer<T> {
  type Target = [T];

  #[inline]
  fn deref(&self) -> &[T] {
    &self.data
  }
}

impl<T> DerefMut for Buffer<T> {
  #[inline]
  fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
    &mut self.data
  }
}

impl<T> Borrow<[T]> for Buffer<T> {
  #[inline]
  fn borrow(&self) -> &[T] {
    &self.data
  }
}

impl<T> BorrowMut<[T]> for Buffer<T> {
  #[inline]
  fn borrow_mut(&mut self) -> &mut [T] {
    &mut self.data
  }
}

impl<T> Clone for Buffer<T> where T: Clone {
  #[inline]
  fn clone(&self) -> Buffer<T> {
    Buffer {
      data: self.data.clone(),
    }
  }

  #[inline]
  fn clone_from(&mut self, other: &Buffer<T>) {
    self.data = other.data.clone()
  }
}

impl<T> Index<usize> for Buffer<T> {
  type Output = T;

  #[inline]
  fn index(&self, index: usize) -> &T {
    &self.data[index]
  }
}

impl<T> IndexMut<usize> for Buffer<T> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut T {
    &mut self.data[index]
  }
}

macro_rules! impl_range_index {
  ($index_type:ty) => {
    impl<T> Index<$index_type> for Buffer<T> {
      type Output = [T];

      #[inline]
      fn index(&self, index: $index_type) -> &[T] {
        &self.data[index]
      }
    }

    impl<T> IndexMut<$index_type> for Buffer<T> {
      #[inline]
      fn index_mut(&mut self, index: $index_type) -> &mut [T] {
        &mut self.data[index]
      }
    }
  }
}

impl_range_index!(Range<usize>);
impl_range_index!(RangeTo<usize>);
impl_range_index!(RangeFrom<usize>);
impl_range_index!(RangeFull);

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
