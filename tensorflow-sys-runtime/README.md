# tensorflow-sys-runtime [![Version][version-icon]][version-page]

The crate provides runtime bindings to [TensorFlow][tensorflow]. Using runtime bindings allows you to avoid
pulling in additional package dependencies into your project. Users will need to call tensorflow::library::load()
before any other calls so that the linking is completed before use.

## NOTE
This crate is meant to be used by [Rust language bindings for Tensorflow][crates-tf]. It is not meant to be used on it's own.

## Requirements

To use these bindings you must have the Tensorflow C libraries installed. See  [install steps][tensorflow-setup]
for detailed instructions.

[tensorflow]: https://www.tensorflow.org
[tensorflow-setup]: https://www.tensorflow.org/install/lang_c
[crates-tf]: https://crates.io/crates/tensorflow
[version-icon]: https://img.shields.io/crates/v/tensorflow-sys-runtime.svg
[version-page]: https://crates.io/crates/tensorflow-sys-runtime
