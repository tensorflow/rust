# Changelog

## Release 0.14.0

### Additions

- Support for high-level graph building in pure Rust
  - Adds Scope, ops module, etc.
  - Currently guarded by experimental_training feature
  - Includes a basic xor example
- Support requesting run metadata from Session::run
- Implement TensorType for half::f16
- Add From<&[i64]> and From<&[u64]> for Shape
- Add Tensor::shape
- Add Shape::new

### Changes

- Change return type of Graph::add_gradients to return optional gradients

### Fixes

- Fix memory initialization bug in Operation::output_consumers
