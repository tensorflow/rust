use super::TensorType;
use libc::size_t;
use std::alloc;
use std::borrow::Borrow;
use std::borrow::BorrowMut;
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
use std::os::raw::c_void as std_c_void;
use std::process;
use std::slice;
#[cfg(feature = "default")]
use tensorflow_sys as tf;
#[cfg(feature = "tensorflow_runtime_linking")]
use tensorflow_sys_runtime as tf;

/// Fixed-length heap-allocated vector.
/// This is basically a `Box<[T]>`, except that that type can't actually be constructed.
/// Furthermore, `[T; N]` can't be constructed if N is not a compile-time constant.
#[derive(Debug)]
pub(crate) struct Buffer<T: TensorType> {
    inner: *mut tf::TF_Buffer,
    owned: bool,
    phantom: PhantomData<T>,
}

impl<T: TensorType> Buffer<T> {
    /// Creates a new buffer initialized to zeros.
    ///
    /// `len` is the number of elements.
    pub fn new(len: usize) -> Self {
        let mut b = unsafe { Buffer::new_uninitialized(len) };
        // TODO: Use libc::memset for primitives once we have impl specialization and
        // memset is included
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
        let align = mem::align_of::<T>();
        let size = mem::size_of::<T>();
        let ptr = alloc::alloc(alloc::Layout::from_size_align(size * len, align).unwrap());
        assert!(!ptr.is_null(), "allocation failure");

        (*inner).data_deallocator = Some(deallocator::<T>);
        (*inner).data = ptr as *mut std_c_void;
        (*inner).length = len;
        Buffer {
            inner,
            owned: true,
            phantom: PhantomData,
        }
    }

    /// Creates a new buffer with no memory allocated.
    pub unsafe fn new_unallocated() -> Self {
        Buffer {
            inner: tf::TF_NewBuffer(),
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
        (*inner).data = ptr as *const std_c_void;
        (*inner).length = len;
        Buffer {
            inner,
            owned: true,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn data(&self) -> *const T {
        unsafe { (*self.inner).data as *const T }
    }

    #[inline]
    fn data_mut(&mut self) -> *mut T {
        unsafe { (*self.inner).data as *mut T }
    }

    #[inline]
    fn length(&self) -> usize {
        unsafe { (*self.inner).length }
    }

    /// Creates a buffer from data owned by the C API.
    ///
    /// `len` is the number of elements.
    /// The underlying data is freed when the buffer is destroyed if `owned`
    /// is true and the `buf` has a data deallocator.
    pub unsafe fn from_c(buf: *mut tf::TF_Buffer, owned: bool) -> Self {
        Buffer {
            inner: buf,
            owned,
            phantom: PhantomData,
        }
    }

    pub fn inner(&self) -> *const tf::TF_Buffer {
        self.inner
    }

    pub fn inner_mut(&mut self) -> *mut tf::TF_Buffer {
        self.inner
    }
}

unsafe extern "C" fn deallocator<T>(data: *mut std_c_void, length: size_t) {
    let align = mem::align_of::<T>();
    let size = mem::size_of::<T>();
    let layout = alloc::Layout::from_size_align(size * length, align).unwrap_or_else(|_| {
        eprintln!("internal error: failed to construct layout");
        // make sure not to unwind
        process::abort();
    });
    alloc::dealloc(data as *mut _, layout);
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
        unsafe { slice::from_raw_parts(self.data(), (*self.inner).length) }
    }
}

impl<T: TensorType> AsMut<[T]> for Buffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data_mut(), (*self.inner).length) }
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
    fn deref_mut(&mut self) -> &mut [T] {
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

impl<T: TensorType> Clone for Buffer<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Buffer<T> {
        let mut b = unsafe { Buffer::new_uninitialized((*self.inner).length) };
        // TODO: Use std::ptr::copy for primitives once we have impl specialization
        for i in 0..self.length() {
            b[i] = self[i].clone();
        }
        b
    }

    #[inline]
    fn clone_from(&mut self, other: &Buffer<T>) {
        assert!(
            self.length() == other.length(),
            "self.length() = {}, other.length() = {}",
            self.length(),
            other.length()
        );
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
        assert!(
            index < self.length(),
            "index = {}, length = {}",
            index,
            self.length()
        );
        unsafe { &*self.data().add(index) }
    }
}

impl<T: TensorType> IndexMut<usize> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(
            index < self.length(),
            "index = {}, length = {}",
            index,
            self.length()
        );
        unsafe { &mut *self.data_mut().add(index) }
    }
}

impl<T: TensorType> Index<Range<usize>> for Buffer<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<usize>) -> &[T] {
        assert!(
            index.start <= index.end,
            "index.start = {}, index.end = {}",
            index.start,
            index.end
        );
        assert!(
            index.end <= self.length(),
            "index.end = {}, length = {}",
            index.end,
            self.length()
        );
        unsafe { slice::from_raw_parts(&*self.data().add(index.start), index.len()) }
    }
}

impl<T: TensorType> IndexMut<Range<usize>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
        assert!(
            index.start <= index.end,
            "index.start = {}, index.end = {}",
            index.start,
            index.end
        );
        assert!(
            index.end <= self.length(),
            "index.end = {}, length = {}",
            index.end,
            self.length()
        );
        unsafe { slice::from_raw_parts_mut(&mut *self.data_mut().add(index.start), index.len()) }
    }
}

impl<T: TensorType> Index<RangeTo<usize>> for Buffer<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeTo<usize>) -> &[T] {
        assert!(
            index.end <= self.length(),
            "index.end = {}, length = {}",
            index.end,
            self.length()
        );
        unsafe { slice::from_raw_parts(&*self.data(), index.end) }
    }
}

impl<T: TensorType> IndexMut<RangeTo<usize>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
        assert!(
            index.end <= self.length(),
            "index.end = {}, length = {}",
            index.end,
            self.length()
        );
        unsafe { slice::from_raw_parts_mut(&mut *self.data_mut(), index.end) }
    }
}

impl<T: TensorType> Index<RangeFrom<usize>> for Buffer<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeFrom<usize>) -> &[T] {
        assert!(
            index.start <= self.length(),
            "index.start = {}, length = {}",
            index.start,
            self.length()
        );
        unsafe {
            slice::from_raw_parts(&*self.data().add(index.start), self.length() - index.start)
        }
    }
}

impl<T: TensorType> IndexMut<RangeFrom<usize>> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
        assert!(
            index.start <= self.length(),
            "index.start = {}, length = {}",
            index.start,
            self.length()
        );
        unsafe {
            slice::from_raw_parts_mut(
                &mut *self.data_mut().add(index.start),
                self.length() - index.start,
            )
        }
    }
}

impl<T: TensorType> Index<RangeFull> for Buffer<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _: RangeFull) -> &[T] {
        unsafe { slice::from_raw_parts(&*self.data(), self.length()) }
    }
}

impl<T: TensorType> IndexMut<RangeFull> for Buffer<T> {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(&mut *self.data_mut(), self.length()) }
    }
}

impl<'a, T: TensorType> From<&'a [T]> for Buffer<T> {
    fn from(data: &'a [T]) -> Buffer<T> {
        let mut buffer = Buffer::new(data.len());
        buffer.clone_from_slice(data);
        buffer
    }
}

impl<'a, T: TensorType> From<&'a Vec<T>> for Buffer<T> {
    #[allow(trivial_casts)]
    fn from(data: &'a Vec<T>) -> Buffer<T> {
        Buffer::from(data as &[T])
    }
}

impl<T: TensorType> From<Buffer<T>> for Vec<T> {
    fn from(buffer: Buffer<T>) -> Vec<T> {
        let mut vec = Vec::with_capacity(buffer.len());
        vec.extend_from_slice(&buffer);
        vec
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
