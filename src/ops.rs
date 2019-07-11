//! This module exposes functions for building standard operations.
//!
//! This module currently requires the `experimental_training` feature.
//!
//! Each operation has a struct which can be used as a builder and allows
//! setting optional attributes:
//!
//! ```ignore
//! MatMul::new().transpose_a(true).build(&mut scope, a, b)?;
//! ```
//!
//! and a function which is shorter when no attributes need to be set:
//!
//! ```ignore
//! mat_mul(&mut scope, a, b)
//! ```

use tensorflow_macros::define_op;

mod array_ops;
pub use array_ops::*;

mod math_ops;
pub use math_ops::*;

mod random_ops;
pub use random_ops::*;

mod state_ops;
pub use state_ops::*;

define_op!(no_op, NoOp, "NoOp");
