#TensorFlow Rust

TensorFlow Rust provides [Rust](https://www.rust-lang.org) language bindings for
[TensorFlow](http://tensorflow.org).

This project is still under active development and not yet ready for widespread use.

## Building

Install [TensorFlow from source](https://www.tensorflow.org/versions/master/get_started/os_setup.html#source).
The Python/pip steps are not necesary, but building `tensorflow:libtensorflow.so` is.

In short:

1. [Install Bazel](http://bazel.io/docs/install.html)
1. `git clone --recurse-submodules https://github.com/tensorflow/tensorflow`
1. `cd tensorflow`
1. `./configure`
1. `bazel build -c opt tensorflow:libtensorflow.so`

Copy $TENSORFLOW_SRC/tensorflow/core/public/tensor_c_api.h to /usr/local/include,
and copy $TENSORFLOW_SRC/bazel-bin/tensorflow/libtensorflow.so to /usr/local/lib.
If this is not possible, add $TENSORFLOW_SRC/tensorflow/core/public to CPATH
and $TENSORFLOW_SRC/bazel-bin/tensorflow to LD_LIBRARY_PATH.

You may need to run `ldconfig` to reset `ld`'s cache after copying libtensorflow.so.

Now run `cargo build` as usual.

## Other

This project is not directly affiliated with the TensorFlow project, although we
do intend to communicate and cooperate with them.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute code.

This is not an official Google product.

##For more information

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)
