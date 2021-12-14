/// Tentative implementation of raw_ops for unit testing.
use crate::eager::TensorHandle;
use crate::Result;
use tensorflow_sys as tf;

use super::op::Op;

/// Add
#[derive(::std::fmt::Debug)]
pub struct Add {
    T: ::std::option::Option<crate::DataType>,
    device_name: ::std::option::Option<String>,
}

impl ::std::default::Default for Add {
    fn default() -> Self {
        Self {
            T: None,
            device_name: None,
        }
    }
}

impl Add {
    /// Creates a new `Add`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the `T` attribute.
    pub fn T<ArgType: ::std::convert::Into<crate::DataType>>(mut self, value: ArgType) -> Self {
        self.T = ::std::option::Option::Some(value.into());
        self
    }

    /// Set the `device_name` where in the Op is executed.
    pub fn set_device(&mut self, device_name: &str) {
        self.device_name = ::std::option::Option::Some(device_name.to_string());
    }

    /// Execute add.
    pub fn call<'a>(
        &self,
        ctx: &'a crate::eager::Context,
        x: &TensorHandle,
        y: &TensorHandle,
    ) -> Result<TensorHandle<'a>> {
        let status = crate::Status::new();

        // Define Op

        let op_name = "Add";
        let mut op = Op::new(ctx, op_name)?;

        // Required input arguments
        op.add_input(&x)?;
        op.add_input(&y)?;

        // Attributes
        if let ::std::option::Option::Some(value) = &self.T {
            let attr_name = "T";
            op.set_attr_type(attr_name, *value)?;
        }

        // Device
        if let ::std::option::Option::Some(device_name) = &self.device_name {
            op.set_device(device_name)?;
        }

        // Execute Op
        let mut num_output = 1;
        let mut res = [std::ptr::null_mut::<tensorflow_sys::TFE_TensorHandle>(); 1];
        unsafe {
            tf::TFE_Execute(
                op.inner,
                res.as_mut_ptr(),
                (&mut num_output) as *mut i32,
                status.inner,
            );
        };
        if status.is_ok() {
            let ret = TensorHandle::from_tensor_handle(ctx, res[0]);
            return Ok(ret);
        }
        Err(status)
    }
}

/// add with default options.
pub fn add<'a>(
    ctx: &'a crate::eager::Context,
    x: &TensorHandle,
    y: &TensorHandle,
) -> Result<TensorHandle<'a>> {
    let op = Add::new();
    op.call(ctx, x, y)
}
