#![allow(dead_code)] // until raw_ops are implemented

use libc::c_float;
use libc::c_int;
use libc::c_uchar;
use libc::c_void;
use libc::size_t;
use std::ffi::{CStr, CString};
use std::os::raw::c_void as std_c_void;
use std::ptr;

use crate::eager::{Context, TensorHandle};
use crate::{AnyTensor, DataType, Result, Shape, Status};

use tensorflow_sys as tf;

/// Description of the TensorFlow op to execute.
struct Op {
    inner: *mut tf::TFE_Op,
}
impl_drop!(Op, TFE_DeleteOp);

impl Op {
    fn new(ctx: &Context, op_or_function_name: &str) -> Result<Op> {
        let status = Status::new();

        let op_or_function_name = CString::new(op_or_function_name)?;
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
            return Ok(name.to_str()?);
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
        status.into_result()
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
        status.into_result()
    }

    /// Sets the value of a string attribute.
    fn set_attr_string(&mut self, attr_name: &str, value: &str) -> Result<()> {
        let attr_name = CString::new(attr_name)?;
        let c_value = value.as_bytes();
        unsafe {
            tf::TFE_OpSetAttrString(
                self.inner,
                attr_name.as_ptr(),
                c_value.as_ptr() as *const std_c_void,
                c_value.len() as size_t,
            );
        }
        Ok(())
    }

    /// Sets the value of an attribute which holds a list of strings.
    fn set_attr_string_list<S: AsRef<str>>(&mut self, attr_name: &str, values: &[S]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
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
        Ok(())
    }

    /// Sets an int-valued attribute.
    fn set_attr_int(&mut self, attr_name: &str, value: i64) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrInt(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of ints.
    fn set_attr_int_list(&mut self, attr_name: &str, value: &[i64]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrIntList(
                self.inner,
                c_attr_name.as_ptr(),
                value.as_ptr(),
                value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a float-valued attribute.
    fn set_attr_float(&mut self, attr_name: &str, value: f32) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrFloat(self.inner, c_attr_name.as_ptr(), value);
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of floats.
    fn set_attr_float_list(&mut self, attr_name: &str, value: &[f32]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
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
        Ok(())
    }

    /// Sets a boolean-valued attribute.
    fn set_attr_bool(&mut self, attr_name: &str, value: bool) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrBool(self.inner, c_attr_name.as_ptr(), if value { 1 } else { 0 });
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of booleans.
    fn set_attr_bool_list(&mut self, attr_name: &str, value: &[bool]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<c_uchar> = value.iter().map(|x| if *x { 1 } else { 0 }).collect();
        unsafe {
            tf::TFE_OpSetAttrBoolList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as c_int,
            );
        }
        Ok(())
    }

    /// Sets a type-valued attribute.
    fn set_attr_type(&mut self, attr_name: &str, value: DataType) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        unsafe {
            tf::TFE_OpSetAttrType(self.inner, c_attr_name.as_ptr(), value.to_c());
        }
        Ok(())
    }

    /// Sets an attribute which holds an array of types.
    fn set_attr_type_list(&mut self, attr_name: &str, value: &[DataType]) -> Result<()> {
        let c_attr_name = CString::new(attr_name)?;
        let c_value: Vec<tf::TF_DataType> = value.iter().map(|x| x.to_c()).collect();
        unsafe {
            tf::TFE_OpSetAttrTypeList(
                self.inner,
                c_attr_name.as_ptr(),
                c_value.as_ptr(),
                c_value.len() as i32,
            );
        }
        Ok(())
    }

    /// Sets a shape-valued attribute.
    fn set_attr_shape(&mut self, attr_name: &str, value: &Shape) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name)?;
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
        status.into_result()
    }

    /// Sets an attribute which holds an array of shapes.
    fn set_attr_shape_list(&mut self, attr_name: &str, value: &[Shape]) -> Result<()> {
        let status = Status::new();

        let c_attr_name = CString::new(attr_name)?;
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
        status.into_result()
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
