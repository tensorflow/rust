#![allow(non_snake_case)]
/// Code for Op's ut that mimics raw_opw.
use crate::eager::TensorHandle;
use crate::Result;

use super::Op;

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
        let [h] = op.execute::<1>(ctx)?;
        Ok(h)
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
