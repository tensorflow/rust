# Changelog

## Release 0.17.0

### Additions

- Support Tensor::from and Shape::from for arrays (requires Rust 1.51)
- Add dtype and shape accessors to TensorInfo
- Implement Scope::with_xla_cluster
- Add Scope::with_kernel_label

### Changes

- Upgrade to TensorFlow 2.5
- Deprecate Session::from_saved_model in favor of SavedModelBundle::load

## Release 0.16.1

### Fixes

- Fix Windows build

## Release 0.16.0

### Additions

- Stabilize new graph generation code (removed experimental_training feature)
- Add Scope::{with_device, with_control_dependencies}
- Add optional support for Tensor conversions to/from ndarray::Array
- Add Library::op_list
- Allow tensorflow-sys to download prebuilt windows releases

### Changes

- Improve ergonomics for graph building
  - Allow conversions for arguments to generated ops
  - Implement From for arrays for Shape
  - Allow VariableBuilder::shape to take Into<Shape>

### Fixes

- Fix memory safety bug in Operation::control_inputs
- Allow 0 colons in output names, default to index 0
- Fix docs.rs (hopefully)

## Release 0.15.0

### Additions

- Add generated code for all standard ops
  - Currently guarded by experimental_training feature
- Add RecordReader for TFRecords
- Add support for creating saved models
- Document that BFloat16 is not an IEEE-754 16-bit float
- Implement Send and Sync for Status
- Add Tensor::get and Tensor::set

### Changes

- Use std::alloc instead of aligned_alloc

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
