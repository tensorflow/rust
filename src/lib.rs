extern crate libc;
extern crate libtensorflow_sys;

use std::ffi::CStr;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Drop;

use libtensorflow_sys as tf;

////////////////////////

fn check_not_null<T>(p: *mut T) -> *mut T {
  assert!(!p.is_null());
  p
}

////////////////////////

macro_rules! impl_new {
  ($name: ident, $call:ident) => {
    impl $name {
      pub fn new() -> Self {
        unsafe {
          $name {
            inner: check_not_null(tf::$call()),
          }
        }
      }
    }
  }
}

////////////////////////

macro_rules! impl_drop {
  ($name: ident, $call:ident) => {
    impl Drop for $name {
      fn drop(&mut self) {
        unsafe {
          tf::$call(self.inner);
        }
      }
    }
  }
}

////////////////////////

macro_rules! c_enum {
  ($enum_name:ident { $($name:ident = $num:expr),* }) => {
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug)]
    pub enum $enum_name {
      UnrecognizedEnumValue(::libc::c_uint),
      $($name),*
    }

    impl $enum_name {
      #[allow(dead_code)]
      fn from_int(value: ::libc::c_uint) -> $enum_name {
        match value {
          $($num => $enum_name::$name,)*
          c => $enum_name::UnrecognizedEnumValue(c),
        }
      }

      #[allow(dead_code)]
      fn to_int(&self) -> ::libc::c_uint {
        match self {
          &$enum_name::UnrecognizedEnumValue(c) => c,
          $(&$enum_name::$name => $num),*
        }
      }
    }

    impl ::std::fmt::Display for $enum_name {
      fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match self {
          $(&$enum_name::$name => f.write_str(stringify!($name)),)*
          &$enum_name::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
      }
    }
  };
  ($enum_name:ident { $($name:ident = $num:expr,)* }) => {
    c_enum!($enum_name { $($name = $num),* });
  }
}

////////////////////////

c_enum!(Code {
  Ok = 0,
  Cancelled = 1,
  Unknown = 2,
  InvalidArgument = 3,
  DeadlineExceeded = 4,
  NotFound = 5,
  AlreadyExists = 6,
  PermissionDenied = 7,
  ResourceExhausted = 8,
  FailedPrecondition = 9,
  Aborted = 10,
  OutOfRange = 11,
  Unimplemented = 12,
  Internal = 13,
  Unavailable = 14,
  DataLoss = 15,
  Unauthenticated = 16,
});

////////////////////////

pub struct Status {
  inner: *mut tf::TF_Status,
}

impl_new!(Status, TF_NewStatus);
impl_drop!(Status, TF_DeleteStatus);

impl Status {
  pub fn code(&self) -> Code {
    unsafe {
      Code::from_int(tf::TF_GetCode(self.inner))
    }
  }
}

impl Display for Status {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    unsafe {
      try!(write!(f, "{}: ", self.code()));
      let msg = match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      };
      f.write_str(msg)
    }
  }
}

////////////////////////

pub struct SessionOptions {
  inner: *mut tf::TF_SessionOptions,
}

impl_new!(SessionOptions, TF_NewSessionOptions);
impl_drop!(SessionOptions, TF_DeleteSessionOptions);

////////////////////////

pub struct Session {
  inner: *mut tf::TF_Session,
}

impl Session {
  pub fn new(options: &SessionOptions) -> (Option<Self>, Status) {
    let status = Status::new();
    let inner = unsafe { tf::TF_NewSession(options.inner, status.inner) };
    let session = if inner.is_null() {
      None
    } else {
      Some(Session {
        inner: inner,
      })
    };
    (session, status)
  }
}

impl Drop for Session {
  fn drop(&mut self) {
    let status = Status::new();
    unsafe {
      tf::TF_DeleteSession(self.inner, status.inner);
    }
    // TODO: What do we do with the status?
  }
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn smoke() {
    let options = SessionOptions::new();
    match Session::new(&options) {
      (Some(_), status) => assert_eq!(status.code(), Code::Ok),
      (None, status) => panic!("Creating session failed with status: {}", status),
    }
  }
}
