//! C API extensions to experiment with eager execution of kernels.
//! WARNING: Unlike tensorflow/c/c_api.h, the API here is not guaranteed to be
//! stable and can change without notice.
use std::ffi::{CStr, CString};
use std::ptr;

use crate::Shape;
use libc::c_float;
use libc::c_int;
use libc::c_uchar;
use libc::c_void;
use libc::size_t;
use std::marker::PhantomData;
use std::os::raw::c_void as std_c_void;

use tensorflow_sys as tf;

use crate::{AnyTensor, DataType, Device, Result, Status, Tensor, TensorType};

pub mod raw_ops;

/// Options that can be passed during context creation.
#[derive(Debug)]
pub struct ContextOptions {
    inner: *mut tf::TFE_ContextOptions,
}
impl_new!(
    ContextOptions,
    TFE_NewContextOptions,
    "Creates a blank set of context options."
);
impl_drop!(ContextOptions, TFE_DeleteContextOptions);

impl ContextOptions {
    /// Set the config.
    ///
    /// `config` should be a serialized [`ConfigProto` proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto).
    /// Returns an error if config was not parsed successfully as a `ConfigProto`.
    pub fn set_config(&mut self, config: &[u8]) -> Result<()> {
        let mut status = Status::new();
        unsafe {
            tf::TFE_ContextOptionsSetConfig(
                self.inner,
                config.as_ptr() as *const _,
                config.len(),
                status.inner(),
            );
        }
        status.into_result()
    }

    /// Sets the default execution mode (sync/async).
    pub fn set_async(&mut self, enable: bool) {
        unsafe {
            tf::TFE_ContextOptionsSetAsync(self.inner, enable as u8);
        }
    }
}

/// "Context" under which operations/functions are executed.
#[derive(Debug)]
pub struct Context {
    inner: *mut tf::TFE_Context,
}
impl_drop!(Context, TFE_DeleteContext);

impl Context {
    /// Create a "Context"
    pub fn new(opts: ContextOptions) -> Result<Self> {
        let status = Status::new();

        let inner = unsafe { tf::TFE_NewContext(opts.inner, status.inner) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Context { inner })
        }
    }

    /// Lists all devices in a context.
    pub fn device_list(&self) -> Result<Vec<Device>> {
        let status = Status::new();
        unsafe {
            let list = tf::TFE_ContextListDevices(self.inner, status.inner);
            if !status.is_ok() {
                return Err(status);
            }
            let result = (|| {
                let n = tf::TF_DeviceListCount(list);
                let mut devices = Vec::with_capacity(n as usize);
                for i in 0..n {
                    let c_name = tf::TF_DeviceListName(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let c_type = tf::TF_DeviceListType(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let bytes = tf::TF_DeviceListMemoryBytes(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    let incarnation = tf::TF_DeviceListIncarnation(list, i, status.inner);
                    if !status.is_ok() {
                        return Err(status);
                    }
                    devices.push(Device {
                        name: CStr::from_ptr(c_name).to_str()?.to_string(),
                        device_type: CStr::from_ptr(c_type).to_str()?.to_string(),
                        memory_bytes: bytes,
                        incarnation,
                    });
                }
                Ok(devices)
            })();
            tf::TF_DeleteDeviceList(list);
            result
        }
    }

    /// Clears the internal caches in the context.
    pub fn clear_cashes(&mut self) {
        unsafe {
            tf::TFE_ContextClearCaches(self.inner);
        }
    }
}

unsafe impl std::marker::Send for Context {}
unsafe impl std::marker::Sync for Context {}

/// A handle to a tensor on a device.
///
/// Constructing a TensorHandle requires a reference to an execute context so that the
/// generated handle will not out live the context.
/// ```
/// # use tensorflow::Tensor;
/// use tensorflow::eager::{ContextOptions, Context, ToHandle};
///
/// let opts = ContextOptions::new();
/// let ctx = Context::new(opts).unwrap();
///
/// let t = Tensor::from(3i32);                // <- Tensor
/// let h = t.to_handle(&ctx).unwrap();        // <- TensorHandle
/// let v: Tensor<i32> = h.resolve().unwrap(); // <- Tensor
/// assert_eq!(v[0], 3i32);
/// ```
///
/// Since TensorHandle cannot be alive beyond the lifetime of the context, the following
/// code will not compile.
/// ```compile_fail
/// # use tensorflow::Tensor;
/// use tensorflow::eager::{ContextOptions, Context, ToHandle};
///
/// let h = {
///     let opts = ContextOptions::new();
///     let ctx = Context::new(opts).unwrap();
///
///     let t = Tensor::from(3);
///     let h = t.to_handle(&ctx).unwrap();
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
        let inner = unsafe { tf::TFE_NewTensorHandle(t.inner().unwrap(), status.inner) };

        if inner.is_null() {
            Err(status)
        } else {
            Ok(TensorHandle {
                inner,
                ctx: PhantomData,
            })
        }
    }

    /// Returns the DataType that corresponds to this type.
    pub fn data_type(&self) -> DataType {
        unsafe { DataType::from_c(tf::TFE_TensorHandleDataType(self.inner)) }
    }

    /// Return the number of dimensions.
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn num_dims(&self) -> Result<i32> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleNumDims(self.inner, status.inner) };
        if !status.is_ok() {
            Err(status)
        } else {
            Ok(num_dims)
        }
    }

    /// Return the number of elements
    pub fn num_elements(&self) -> Result<i64> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleNumElements(self.inner, status.inner) };
        if !status.is_ok() {
            Err(status)
        } else {
            Ok(num_dims)
        }
    }

    /// Return a number elements for a given dim_index.
    /// This function will block till the operation that produces the TensorHandle has completed.
    pub fn dim(&self, dim_index: i32) -> Result<i64> {
        let status = Status::new();
        let num_dims = unsafe { tf::TFE_TensorHandleDim(self.inner, dim_index, status.inner) };
        if !status.is_ok() {
            Err(status)
        } else {
            Ok(num_dims)
        }
    }

    /// Returns the device of the operation that produced `h`. If `h` was produced by
    /// a copy, returns the destination device of the copy. Note that the returned
    /// device name is not always the device holding the tensor handle's memory. If
    /// you want the latter, use TFE_TensorHandleBackingDeviceName. This function
    /// will block till the operation that produces `h` has completed.
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
    ///
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

    /// Return a pointer to a new TFE_TensorHandle that shares the underlying tensor
    /// with `h`. On success, `status` is set to OK. On failure, `status` reflects
    /// the error and a nullptr is returned.
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
    /// TensorFlow. Hence, callers should not mutate this memory (for example by
    /// modifying the memory region pointed to by TF_TensorData() on the returned
    /// TF_Tensor).
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

        // Safely unwrap since data_type was checked beforehand
        unsafe { Ok(Tensor::from_tf_tensor(tf_tensor).unwrap()) }
    }

    ///
    pub fn debug_info(&self) -> Result<DebugInfo> {
        let status = Status::new();
        let inner = unsafe { tf::TFE_TensorHandleTensorDebugInfo(self.inner, status.inner) };
        if !status.is_ok() {
            Err(status)
        } else {
            Ok(DebugInfo { inner })
        }
    }

    ///
    pub(crate) unsafe fn from_tensor_handle(
        _ctx: &'a Context,
        h: *mut tf::TFE_TensorHandle,
    ) -> Self {
        Self {
            inner: h,
            ctx: PhantomData,
        }
    }
}

/// Debugging/Profiling information for TFE_TensorHandle
///
/// TFE_TensorDebugInfo contains information useful for debugging and
/// profiling tensors.
#[derive(Debug)]
pub struct DebugInfo {
    inner: *mut tf::TFE_TensorDebugInfo,
}
impl_drop!(DebugInfo, TFE_DeleteTensorDebugInfo);

impl DebugInfo {
    /// Returns the number of dimensions used to represent the tensor on its device.
    /// The number of dimensions used to represent the tensor on device can be
    /// different from the number returned by TFE_TensorHandleNumDims.
    /// The return value was current at the time of TFE_TensorDebugInfo creation.
    pub fn on_device_num_dims(&self) -> i32 {
        unsafe { tf::TFE_TensorDebugInfoOnDeviceNumDims(self.inner) }
    }

    /// Returns the number of elements in dimension `dim_index`.
    /// Tensor representation on device can be transposed from its representation
    /// on host. The data contained in dimension `dim_index` on device
    /// can correspond to the data contained in another dimension in on-host
    /// representation. The dimensions are indexed using the standard TensorFlow
    /// major-to-minor order (slowest varying dimension first),
    /// not the XLA's minor-to-major order.
    /// On-device dimensions can be padded. TFE_TensorDebugInfoOnDeviceDim returns
    /// the number of elements in a dimension after padding.
    /// The return value was current at the time of TFE_TensorDebugInfo creation.
    pub fn on_device_dim(&self, dim_index: i32) -> i64 {
        unsafe { tf::TFE_TensorDebugInfoOnDeviceDim(self.inner, dim_index) }
    }
}

struct Op {
    inner: *mut tf::TFE_Op,
}

impl Op {
    fn new(ctx: &Context, op_or_function_name: &str) -> Result<Op> {
        let status = Status::new();

        let op_or_function_name = CString::new(op_or_function_name).unwrap();
        let inner = unsafe { tf::TFE_NewOp(ctx.inner, op_or_function_name.as_ptr(), status.inner) };
        if inner.is_null() {
            return Err(status);
        }
        Ok(Self { inner })
    }

    #[allow(dead_code)]
    fn get_name(&self) -> Result<&str> {
        let status = Status::new();

        let name = unsafe {
            let name = tf::TFE_OpGetName(self.inner, status.inner);
            CStr::from_ptr(name)
        };
        if status.is_ok() {
            return Ok(name.to_str().unwrap());
        }
        Err(status)
    }

    /// Context may not be outlive over the lifetime of `op'
    #[allow(dead_code)]
    fn get_context(&self) -> &Context {
        unimplemented!()
    }

    /// Adds an input to this operation.
    fn add_input(&mut self, input: &TensorHandle) -> Result<()> {
        let status = Status::new();
        unsafe {
            tf::TFE_OpAddInput(self.inner, input.inner, status.inner);
        };
        if status.is_ok() {
            return Ok(());
        }
        Err(status)
    }

    /// Adds multiple inputs to this operation.
    #[allow(dead_code)]
    fn add_input_list(&mut self, inputs: &[TensorHandle]) -> Result<()> {
        let status = Status::new();
        unsafe {
            let mut inputs: Vec<*mut tf::TFE_TensorHandle> =
                inputs.iter().map(|v| v.inner).collect();
            tf::TFE_OpAddInputList(
                self.inner,
                inputs.as_mut_ptr(),
                inputs.len() as c_int,
                status.inner,
            );
        };
        if status.is_ok() {
            return Ok(());
        }
        Err(status)
    }

    /// Sets the value of a string attribute.
    fn set_attr_string(&mut self, attr_name: &str, value: &str) {
        let attr_name = CString::new(attr_name).unwrap();
        let c_value = value.as_bytes();
        unsafe {
            tf::TFE_OpSetAttrString(
                self.inner,
                attr_name.as_ptr(),
                c_value.as_ptr() as *const std_c_void,
                c_value.len() as size_t,
            );
        }
    }

    /// Sets the value of an attribute which holds a list of strings.
    fn set_attr_string_list<S: AsRef<str>>(&mut self, attr_name: &str, values: &[S]) {
        let c_attr_name = CString::new(attr_name).unwrap();
        let bytes: Vec<&[u8]> = values.iter().map(|x| x.as_ref().as_bytes()).collect();
        let ptrs: Vec<*const c_void> = bytes.iter().map(|x| x.as_ptr() as *const c_void).collect();
        let lens: Vec<size_t> = bytes.iter().map(|x| x.len() as size_t).collect();
        unsafe {
            tf::TFE_OpSetAttrStringList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_ptr() as *const *const std_c_void,
                lens.as_ptr(),
                ptrs.len() as c_int,
            );
        }
    }

    /// Sets an int-valued attribute.
    fn set_attr_int(&mut self, attr_name: &str, value: i64) {
        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            tf::TFE_OpSetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
    }

    /// Sets an attribute which holds an array of ints.
    fn set_attr_int_list(&mut self, attr_name: &str, value: &[i64]) {
        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            tf::TFE_OpSetAttrIntList(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr(),
                value.len() as i32,
            );
        }
    }

    /// Sets a float-valued attribute.
    fn set_attr_float(&mut self, attr_name: &str, value: f32) {
        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            tf::TFE_OpSetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
    }

    /// Sets an attribute which holds an array of floats.
    fn set_attr_float_list(&mut self, attr_name: &str, value: &[f32]) {
        let c_attr_name = CString::new(attr_name).unwrap();
        // Allow trivial_numeric_casts here because f32 is not necessarily equal to c_float.
        let c_value: Vec<c_float> = value.iter().map(|x| *x as c_float).collect();
        unsafe {
            tf::TFE_OpSetAttrFloatList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
    }

    /// Sets a boolean-valued attribute.
    fn set_attr_bool(&mut self, attr_name: &str, value: bool) {
        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            tf::TFE_OpSetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
    }

    /// Sets an attribute which holds an array of booleans.
    #[allow(dead_code)]
    fn set_attr_bool_list(&mut self, attr_name: &str, value: &[bool]) {
        let c_attr_name = CString::new(attr_name).unwrap();
        let c_value: Vec<c_uchar> = value.iter().map(|x| if *x { 1 } else { 0 }).collect();
        unsafe {
            tf::TFE_OpSetAttrBoolList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as c_int,
            );
        }
    }

    /// Sets a type-valued attribute.
    fn set_attr_type(&mut self, attr_name: &str, value: DataType) {
        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            tf::TFE_OpSetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
    }

    /// Sets an attribute which holds an array of types.
    fn set_attr_type_list(&mut self, attr_name: &str, value: &[DataType]) {
        let c_attr_name = CString::new(attr_name).unwrap();
        let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TFE_OpSetAttrTypeList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
    }

    /// Sets a shape-valued attribute.
    fn set_attr_shape(&mut self, attr_name: &str, value: &Shape) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name).unwrap();
        unsafe {
            match value.0 {
                None => tf::TFE_OpSetAttrShape(
                    self.inner,
                    c_attr_name.as_ptr(),
                    ptr::null(),
                    -1,
                    status.inner,
                ),
                Some(ref dims) => {
                    let c_dims: Vec<i64> = dims.iter().map(|x| (*x).unwrap_or(-1)).collect();
                    tf::TFE_OpSetAttrShape(
                        self.inner,
                        c_attr_name.as_ptr(),
                        c_dims.as_ptr(),
                        c_dims.len() as i32,
                        status.inner,
                    );
                }
            }
        }
        if status.is_ok() {
            return Ok(());
        }
        Err(status)
    }

    /// Sets an attribute which holds an array of shapes.
    fn set_attr_shape_list(&mut self, attr_name: &str, value: &[Shape]) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name).unwrap();
        // Convert Option<i64> in each shape to i64 with None becoming -1.
        let c_dims: Vec<Option<Vec<i64>>> = value
            .iter()
            .map(|x| {
                x.0.as_ref()
                    .map(|dims| dims.iter().map(|x| (*x).unwrap_or(-1)).collect())
            })
            .collect();
        let mut ptrs: Vec<*const i64> = c_dims
            .iter()
            .map(|x| match *x {
                None => ptr::null(),
                Some(ref dims) => dims.as_ptr(),
            })
            .collect();
        let lens: Vec<c_int> = value
            .iter()
            .map(|x| match x.0 {
                None => -1,
                Some(ref dims) => dims.len() as c_int,
            })
            .collect();
        unsafe {
            tf::TFE_OpSetAttrShapeList(
                self.inner,
                c_attr_name.as_ptr(),
                ptrs.as_mut_ptr(),
                lens.as_ptr(),
                ptrs.len() as c_int,
                status.inner,
            );
        }

        if status.is_ok() {
            return Ok(());
        }
        Err(status)
    }

    /// Sets a tensor-valued attribute.
    fn set_attr_any_tensor(&mut self, attr_name: &str, value: &dyn AnyTensor) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let mut status = Status::new();
        unsafe {
            tf::TFE_OpSetAttrTensor(
                self.inner,
                c_attr_name.as_ptr(),
                value.inner()?,
                status.inner(),
            );
        }
        status.into_result()
    }
}

impl Drop for Op {
    fn drop(&mut self) {
        unsafe { tf::TFE_DeleteOp(self.inner) };
    }
}

///
pub trait ToHandle<'a> {
    ///
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>>;
}

impl<'a, T> ToHandle<'a> for Tensor<T>
where
    T: TensorType,
{
    fn to_handle(&self, ctx: &'a Context) -> Result<TensorHandle<'a>> {
        TensorHandle::new(ctx, self)
    }
}

impl<'a> ToHandle<'a> for TensorHandle<'a> {
    fn to_handle(&self, _: &'a Context) -> Result<TensorHandle<'a>> {
        self.copy_sharing_tensor()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::Tensor;

    #[test]
    fn create() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let mut x = Tensor::new(&[1]);
        x[0] = 2i32;

        let x_handle = x.to_handle(&ctx).unwrap();

        let y = unsafe {
            let status = Status::new();
            let tf_tensor = tf::TFE_TensorHandleResolve(x_handle.inner, status.inner);
            Tensor::<i32>::from_tf_tensor(tf_tensor).unwrap()
        };

        assert_eq!(y[0], 2i32);
    }

    #[test]
    fn add_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let x = Tensor::from(2);
        let y = x.clone();

        let h = raw_ops::add(&ctx, x, y).unwrap();
        let z: Result<Tensor<i32>> = h.resolve();
        let z = z.unwrap();
        assert_eq!(z[0], 4i32);

        let h = raw_ops::add(&ctx, z.clone(), z.clone()).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();
        assert_eq!(z[0], 8i32);

        let h1 = z.to_handle(&ctx).unwrap();
        let h2 = z.to_handle(&ctx).unwrap();
        let h = raw_ops::add(&ctx, h1, h2).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();
        assert_eq!(z[0], 16i32);

        let h1 = z.to_handle(&ctx).unwrap();
        let h2 = z.to_handle(&ctx).unwrap();
        let h = raw_ops::add(&ctx, h1, h2).unwrap();

        let h1 = z.to_handle(&ctx).unwrap();
        let h = raw_ops::add(&ctx, h1, h).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();

        assert_eq!(z[0], 48i32);
    }

    #[test]
    fn read_file_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let filename: Tensor<String> =
            Tensor::from(String::from("test_resources/io/sample_text.txt"));

        let h = raw_ops::read_file(&ctx, filename).unwrap();
        let z: Tensor<String> = h.resolve().unwrap();
        assert_eq!(&z[..], &["This is a sample text for unittest.".to_string()]);
    }

    #[test]
    fn decode_png_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let filename: Tensor<String> = Tensor::from(String::from("test_resources/sample.png"));

        let h = raw_ops::read_file(&ctx, filename).unwrap();
        let h = raw_ops::DecodePng::new()
            .channels(3)
            .dtype(DataType::UInt8)
            .call(&ctx, h)
            .unwrap();
        let z: Tensor<u8> = h.resolve().unwrap();
        assert_eq!(z.len(), 224 * 224 * 3);
    }

    #[test]
    fn top_kv2_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        // 2 rows x 10 cols
        let mut t: Tensor<i64> = Tensor::new(&[2, 10]);
        for i in 0..20 {
            t[i] = i as i64;
        }
        let k = Tensor::from(3);
        let [values, indices] = raw_ops::TopKV2::new()
            .sorted(true)
            .call(&ctx, t, k.clone())
            .unwrap();
        let values: Tensor<i64> = values.resolve().unwrap();
        let indices: Tensor<i32> = indices.resolve().unwrap();

        // 1st row
        assert_eq!(values[0], 9);
        assert_eq!(values[1], 8);
        assert_eq!(values[2], 7);

        assert_eq!(indices[0], 9);
        assert_eq!(indices[1], 8);
        assert_eq!(indices[2], 7);

        // 2nd row
        assert_eq!(values[3], 19);
        assert_eq!(values[4], 18);
        assert_eq!(values[5], 17);

        assert_eq!(indices[3], 9);
        assert_eq!(indices[4], 8);
        assert_eq!(indices[5], 7);
    }

    #[test]
    fn context_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let devices = ctx.device_list().unwrap();
        assert!(devices.len() > 0);
        for d in &devices {
            assert_ne!(String::from(""), d.name);
        }
    }

    #[test]
    fn type_mismatch_test() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let t = Tensor::from(0);
        let h = t.to_handle(&ctx).unwrap();
        let v: Result<Tensor<f32>> = h.resolve();

        if let Err(status) = v {
            eprintln!("{}", status);
        } else {
            assert!(false);
        }
    }
}
