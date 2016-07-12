// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
//! This module builds computation graphs.
//!
//! This module is unfinished.

use std::convert::From;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Error;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops;
use std::rc::Rc;
use super::Buffer;
use super::TensorType;

/// Denotes operator precedence.
/// Used for displaying expressions as strings.
#[derive(Ord,PartialOrd,Eq,PartialEq,Debug)]
pub enum OpLevel {
  Add,
  Mul,
  Unary,
  Atom,
}

////////////////////////

/// A node in an expression tree, which is a thin wrapper around an ExprImpl.
///
/// This is separate from ExprImpl because we want expressions to be wrapped in an Rc,
/// and we can't directly implement std::ops::Add, etc., for Rc<E: ExprImpl<T>>.
#[derive(Debug)]
pub struct Expr<T: TensorType> {
  expr: Rc<ExprImpl<T>>,
}

impl<T: TensorType> Display for Expr<T> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    Display::fmt(&self.expr, f)
  }
}

impl<T: TensorType> From<T> for Expr<T> {
  fn from(value: T) -> Self {
    Expr {
      expr: Rc::new(value),
    }
  }
}

////////////////////////

/// Trait implemented by all expression types.
/// Most users will want to store an Expr instead.
pub trait ExprImpl<T: TensorType>: Display + Debug {
  /// Returns the precedence level for this operator.
  fn op_level(&self) -> OpLevel;
}

impl<T: TensorType> ExprImpl<T> for T {
  fn op_level(&self) -> OpLevel {
    OpLevel::Atom
  }
}

////////////////////////

macro_rules! impl_bin_op {
  ($name:ident, $fn_name:ident, $op:expr, $op_level:ident, $assoc:expr, $doc:expr) => {
    #[doc = $doc]
    #[derive(Debug)]
    pub struct $name<T: TensorType> {
      left: Expr<T>,
      right: Expr<T>,
    }

    impl<T: TensorType> ops::$name for Expr<T> {
      type Output = Expr<T>;

      fn $fn_name(self, rhs: Expr<T>) -> Expr<T> {
        Expr {
          expr: Rc::new($name {
            left: self,
            right: rhs,
          }),
        }
      }
    }

    impl<T: TensorType> ops::$name<T> for Expr<T> {
      type Output = Expr<T>;

      fn $fn_name(self, rhs: T) -> Expr<T> {
        Expr {
          expr: Rc::new($name {
            left: self,
            right: Expr::from(rhs),
          }),
        }
      }
    }

    impl<T: TensorType> Display for $name<T> {
      fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        if self.left.expr.op_level() < OpLevel::$op_level {
          try!(write!(f, "({})", self.left));
        } else {
          try!(write!(f, "{}", self.left));
        }
        try!(write!(f, concat!(" ", $op, " ")));
        let paren = if $assoc {
          self.right.expr.op_level() < OpLevel::$op_level
        } else {
          self.right.expr.op_level() <= OpLevel::$op_level
        };
        if paren {
          write!(f, "({})", self.right)
        } else {
          write!(f, "{}", self.right)
        }
      }
    }

    impl<T: TensorType> ExprImpl<T> for $name<T> {
      fn op_level(&self) -> OpLevel {
        OpLevel::$op_level
      }
    }
  }
}

impl_bin_op!(Add, add, "+", Add, true, "Expression resulting from adding two subexpressions.");
impl_bin_op!(Sub, sub, "-", Add, false, "Expression resulting from subtracting two subexpressions.");
impl_bin_op!(Mul, mul, "*", Mul, true, "Expression resulting from multiplying two subexpressions.");
impl_bin_op!(Div, div, "/", Mul, false, "Expression resulting from dividing two subexpressions.");
impl_bin_op!(Rem, rem, "%", Mul, false, "Expression resulting from taking a modulus.");

////////////////////////

/// Expression resulting from negation of an expression.
#[derive(Debug)]
pub struct Neg<T: TensorType> {
  expr: Expr<T>,
}

impl<T: TensorType> ops::Neg for Expr<T> {
  type Output = Expr<T>;

  fn neg(self) -> Expr<T> {
    Expr {
      expr: Rc::new(Neg {
        expr: self,
      }),
    }
  }
}

impl<T: TensorType> Display for Neg<T> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    try!(write!(f, "-"));
    if self.expr.expr.op_level() <= OpLevel::Unary {
      write!(f, "({})", self.expr)
    } else {
      write!(f, "{}", self.expr)
    }
  }
}

impl<T: TensorType> ExprImpl<T> for Neg<T> {
  fn op_level(&self) -> OpLevel {
    OpLevel::Unary
  }
}

////////////////////////

/// Expression for a variable.
#[derive(Debug)]
pub struct Variable<T: TensorType> {
  initial_value: Buffer<T>,
  shape: Vec<u64>,
  name: String,
}

impl<T: TensorType> Variable<T> {
  pub fn new(initial_value: Buffer<T>, shape: &[u64], name: &str) -> Self {
    Variable {
      initial_value: initial_value,
      shape: Vec::from(shape),
      name: name.to_string(),
    }
  }
}

impl<T: TensorType> Display for Variable<T> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    write!(f, "{}", self.name)
  }
}

impl<T: TensorType> ExprImpl<T> for Variable<T> {
  fn op_level(&self) -> OpLevel {
    OpLevel::Atom
  }
}

////////////////////////

/// Expression for a placeholder.
#[derive(Debug)]
pub struct Placeholder<T: TensorType> {
  shape: Vec<u64>,
  name: String,
  phantom: PhantomData<T>,
}

impl<T: TensorType> Placeholder<T> {
  pub fn new(shape: &[u64], name: &str) -> Self {
    Placeholder {
      shape: Vec::from(shape),
      name: name.to_string(),
      phantom: PhantomData,
    }
  }
}

impl<T: TensorType> Display for Placeholder<T> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
    write!(f, "{}", self.name)
  }
}

impl<T: TensorType> ExprImpl<T> for Placeholder<T> {
  fn op_level(&self) -> OpLevel {
    OpLevel::Atom
  }
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;
  use super::super::Buffer;

  #[test]
  fn test_display() {
    assert_eq!("1 + 2 + 3", format!("{}", (Expr::from(1) + 2) + 3));
    assert_eq!("1 + 2 + 3", format!("{}", Expr::from(1) + (Expr::from(2) + 3)));
    assert_eq!("1 + 2 - 3", format!("{}", (Expr::from(1) + 2) - 3));
    assert_eq!("1 - (2 + 3)", format!("{}", Expr::from(1) - (Expr::from(2) + 3)));

    assert_eq!("(1 + 2) * 3", format!("{}", (Expr::from(1) + 2) * 3));
    assert_eq!("1 * (2 + 3)", format!("{}", Expr::from(1) * (Expr::from(2) + 3)));
    assert_eq!("1 * 2 * 3", format!("{}", (Expr::from(1) * 2) * 3));
    assert_eq!("1 * 2 * 3", format!("{}", Expr::from(1) * (Expr::from(2) * 3)));

    assert_eq!("(1 + 2) / 3", format!("{}", (Expr::from(1) + 2) / 3));
    assert_eq!("1 / (2 + 3)", format!("{}", Expr::from(1) / (Expr::from(2) + 3)));
    assert_eq!("1 * 2 / 3", format!("{}", (Expr::from(1) * 2) / 3));
    assert_eq!("1 / (2 * 3)", format!("{}", Expr::from(1) / (Expr::from(2) * 3)));

    assert_eq!("(1 + 2) % 3", format!("{}", (Expr::from(1) + 2) % 3));
    assert_eq!("1 % (2 + 3)", format!("{}", Expr::from(1) % (Expr::from(2) + 3)));
    assert_eq!("1 * 2 % 3", format!("{}", (Expr::from(1) * 2) % 3));
    assert_eq!("1 % (2 * 3)", format!("{}", Expr::from(1) % (Expr::from(2) * 3)));

    assert_eq!("-1", format!("{}", -Expr::from(1)));
    assert_eq!("-(-1)", format!("{}", -(-Expr::from(1))));
    assert_eq!("-(1 + 2)", format!("{}", -(Expr::from(1) + 2)));

    let buf = Buffer::new(6);
    assert_eq!("x", format!("{}", <Variable<f32>>::new(buf, &vec![2, 3], "x")));

    assert_eq!("x", format!("{}", <Placeholder<f32>>::new(&vec![2, 3], "x")));
  }
}
