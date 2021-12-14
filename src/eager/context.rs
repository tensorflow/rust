use std::ffi::CStr;

use tensorflow_sys as tf;

use crate::{Device, Result, Status};

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

/// Context under which operations/functions are executed.
#[derive(Debug)]
pub struct Context {
    pub(crate) inner: *mut tf::TFE_Context,
}
impl_drop!(Context, TFE_DeleteContext);

impl Context {
    /// Create a Context
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
    pub fn clear_caches(&mut self) {
        unsafe {
            tf::TFE_ContextClearCaches(self.inner);
        }
    }
}

unsafe impl std::marker::Send for Context {}
unsafe impl std::marker::Sync for Context {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_create_context() {
        let opts = ContextOptions::new();
        Context::new(opts).unwrap();
    }

    #[test]
    fn test_create_async_context() {
        let mut opts = ContextOptions::new();
        opts.set_async(true);
        Context::new(opts).unwrap();
    }

    #[test]
    fn test_context_set_config() {
        use crate::protos::config::{ConfigProto, GPUOptions};
        use protobuf::Message;

        let gpu_options = GPUOptions {
            per_process_gpu_memory_fraction: 0.5,
            allow_growth: true,
            ..Default::default()
        };
        let mut config = ConfigProto::new();
        config.set_gpu_options(gpu_options);

        let mut buf = vec![];
        config.write_to_writer(&mut buf).unwrap();

        let mut opts = ContextOptions::new();
        opts.set_config(&buf).unwrap();
        Context::new(opts).unwrap();
    }

    #[test]
    fn test_device_list() {
        let opts = ContextOptions::new();
        let ctx = Context::new(opts).unwrap();

        let devices = ctx.device_list().unwrap();
        for d in &devices {
            assert_ne!(String::from(""), d.name);
        }
    }
}
