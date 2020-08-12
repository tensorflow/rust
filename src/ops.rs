//! This module exposes functions for building standard operations.
//!
//! Each operation has a struct which can be used as a builder and allows
//! setting optional attributes:
//!
//! ```ignore
//! MatMul::new().transpose_a(true).build(a, b, &mut scope)?;
//! ```
//!
//! and a function which is shorter when no attributes need to be set:
//!
//! ```ignore
//! mat_mul(a, b, &mut scope)
//! ```
//!
//! Note that for some ops, the builder may always be required, because
//! the op has required attributes with no default specified.

mod math_ops;
pub use math_ops::*;

mod random_ops;
pub use random_ops::*;

#[allow(
    clippy::double_parens,
    clippy::too_many_arguments,
    clippy::wrong_self_convention
)]
mod ops_impl;
pub use ops_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device() {
        let op = no_op(&mut crate::Scope::new_root_scope().with_device("foo")).unwrap();
        assert_eq!(op.device().unwrap(), "foo");
    }
}
