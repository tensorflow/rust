use crate::Result;
use crate::{AnyTensor, Status, Tensor, TensorType};
use std::ffi::CString;
use tensorflow_sys as tf;
use tf::TFE_ContextListDevices;

///
#[derive(Debug)]
pub struct ContextOptions {
    inner: *mut tf::TFE_ContextOptions,
}

impl ContextOptions {
    ///
    pub fn new() -> Self {
        let inner = unsafe { tf::TFE_NewContextOptions() };
        ContextOptions { inner }
    }
}

impl Drop for ContextOptions {
    fn drop(&mut self) {
        unsafe {
            tf::TFE_DeleteContextOptions(self.inner);
        }
    }
}

///
#[derive(Debug)]
pub struct Context {
    inner: *mut tf::TFE_Context,
}

impl Context {
    ///
    pub fn new() -> Result<Self> {
        let status = Status::new();
        let opts = ContextOptions::new();

        let inner = unsafe { tf::TFE_NewContext(opts.inner, status.inner) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Context { inner })
        }
    }

    ///
    pub fn new_with_options(opts: ContextOptions) -> Result<Self> {
        let status = Status::new();

        let inner = unsafe { tf::TFE_NewContext(opts.inner, status.inner) };
        if inner.is_null() {
            Err(status)
        } else {
            Ok(Context { inner })
        }
    }

    ///
    pub fn list_devices(&self) -> *mut tf::TF_DeviceList {
        let status = Status::new();
        unsafe { TFE_ContextListDevices(self.inner, status.inner) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            tf::TFE_DeleteContext(self.inner);
        }
    }
}

///
#[derive(Debug)]
pub struct TensorHandle {
    inner: *mut tf::TFE_TensorHandle,
}

impl TensorHandle {
    pub fn resolve<T: TensorType>(self) -> Result<Tensor<T>> {
        let status = Status::new();
        unsafe {
            let tf_tensor = tf::TFE_TensorHandleResolve(self.inner, status.inner);
            Ok(Tensor::from_tf_tensor(tf_tensor).unwrap())
        }
    }
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        unsafe {
            tf::TFE_DeleteTensorHandle(self.inner);
        }
    }
}

///
pub trait ToHandle {
    ///
    fn to_handle(self) -> Result<TensorHandle>;
}

impl<T> ToHandle for Tensor<T>
where
    T: TensorType,
{
    fn to_handle(self) -> Result<TensorHandle> {
        let mut status = Status::new();
        let inner = unsafe { tf::TFE_NewTensorHandle(self.inner().unwrap(), status.inner()) };

        if inner.is_null() {
            Err(status)
        } else {
            Ok(TensorHandle { inner })
        }
    }
}

impl ToHandle for TensorHandle {
    fn to_handle(self) -> Result<TensorHandle> {
        Ok(self)
    }
}

/// add
pub fn add<T1, T2>(x: T1, y: T2) -> Result<TensorHandle>
where
    T1: ToHandle,
    T2: ToHandle,
{
    unsafe {
        let add = CString::new("Add").unwrap();
        let status = Status::new();
        let opts = ContextOptions::new();
        let context = Context::new_with_options(opts).unwrap();
        let op = tf::TFE_NewOp(context.inner, add.as_ptr(), status.inner);
        tf::TFE_OpAddInput(op, x.to_handle()?.inner, status.inner);
        tf::TFE_OpAddInput(op, y.to_handle()?.inner, status.inner);

        let mut num_output = 1;
        let mut res = [std::ptr::null_mut::<tf::TFE_TensorHandle>()];
        tf::TFE_Execute(
            op,
            res.as_mut_ptr(),
            (&mut num_output) as *mut i32,
            status.inner,
        );
        Ok(TensorHandle { inner: res[0] })
    }
}

///
pub fn read_file<T>(filename: T) -> Result<Tensor<String>>
where
    T: ToHandle,
{
    unsafe {
        let add = CString::new("ReadFile").unwrap();
        let status = Status::new();
        let opts = ContextOptions::new();
        let context = Context::new_with_options(opts).unwrap();
        let op = tf::TFE_NewOp(context.inner, add.as_ptr(), status.inner);
        tf::TFE_OpAddInput(op, filename.to_handle()?.inner, status.inner);

        let mut num_output = 1;
        let mut res = [std::ptr::null_mut::<tf::TFE_TensorHandle>()];
        tf::TFE_Execute(
            op,
            res.as_mut_ptr(),
            (&mut num_output) as *mut i32,
            status.inner,
        );
        let tf_tensor = tf::TFE_TensorHandleResolve(res[0], status.inner);
        if tf_tensor.is_null() {
            Err(status)
        } else {
            Ok(Tensor::from_tf_tensor(tf_tensor).unwrap())
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::Tensor;

    #[test]
    fn create() {
        let mut x = Tensor::new(&[1]);
        x[0] = 2i32;

        let x_handle = x.to_handle().unwrap();

        let y = unsafe {
            let status = Status::new();
            let tf_tensor = tf::TFE_TensorHandleResolve(x_handle.inner, status.inner);
            Tensor::<i32>::from_tf_tensor(tf_tensor).unwrap()
        };

        assert_eq!(y[0], 2i32);
    }

    #[test]
    fn add_test() {
        let mut x = Tensor::new(&[1]);
        x[0] = 2i32;
        let y = x.clone();

        let h = add(x, y).unwrap();
        let z: Result<Tensor<i32>> = h.resolve();
        assert!(z.is_ok());
        let z = z.unwrap();
        assert_eq!(z[0], 4i32);

        let h = add(z.clone(), z.clone()).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();
        assert_eq!(z[0], 8i32);

        let h1 = z.clone().to_handle().unwrap();
        let h2 = z.clone().to_handle().unwrap();
        let h = add(h1, h2).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();
        assert_eq!(z[0], 16i32);

        let h1 = z.clone().to_handle().unwrap();
        let h2 = z.clone().to_handle().unwrap();
        let h = add(h1, h2).unwrap();

        let h1 = z.clone().to_handle().unwrap();
        let h = add(h1, h).unwrap();
        let z: Tensor<i32> = h.resolve().unwrap();

        assert_eq!(z[0], 48i32);
    }

    #[test]
    fn read_file_test() {
        let filename: Tensor<String> =
            Tensor::from(String::from("test_resources/io/sample_text.txt"));

        let z: Result<Tensor<String>> = read_file(filename);
        assert!(z.is_ok());
        let z = z.unwrap();
        assert_eq!(z.len(), 1);
        assert_eq!(z[0].len(), 32);
        assert_eq!(z[0], "This a sample text for unittest.")
    }

    #[test]
    fn context() {
        use std::ffi::CStr;

        let opts = ContextOptions::new();
        let ctx = Context::new_with_options(opts).unwrap();

        let devices = ctx.list_devices();
        let num_devices = unsafe { tf::TF_DeviceListCount(devices) };
        assert!(num_devices > 0);
        let status = Status::new();
        for i in 0..num_devices {
            let name_raw = unsafe { tf::TF_DeviceListName(devices, i, status.inner) };
            let _name = unsafe { CStr::from_ptr(name_raw) };
            assert!(status.is_ok());
        }

        unsafe {
            tf::TF_DeleteDeviceList(devices);
        }
    }
}
