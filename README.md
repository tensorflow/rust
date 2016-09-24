#TensorFlow Rust
[![Version](https://img.shields.io/crates/v/tensorflow.svg)](https://crates.io/crates/tensorflow)
[![Status](https://travis-ci.org/google/tensorflow-rust.svg?branch=master)](https://travis-ci.org/google/tensorflow-rust)
[![Chat on irc] (https://img.shields.io/badge/chat-on%20mozilla%20rust--machine--learning-brightgreen.svg)](irc://irc.mozilla.org/rust-machine-learning)
TensorFlow Rust provides [Rust](https://www.rust-lang.org) language bindings for
[TensorFlow](http://tensorflow.org).

This project is still under active development and not yet ready for widespread use.

## Building

Install [TensorFlow from source](https://www.tensorflow.org/versions/master/get_started/os_setup.html#source).
The Python/pip steps are not necessary, but building `tensorflow:libtensorflow.so` is.

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
To include the especially unstable API (which is currently the `expr` module),
use `--features tensorflow_unstable`.

## RFCs
RFCs are [issues tagged with RFC](https://github.com/google/tensorflow-rust/labels/rfc).
Check them out and comment. Discussions are welcome. After all, thats what a Request For Comment is for!

## FAQs

#### Why does the compiler say that parts of the API don't exist?
The especially unstable parts of the API (which is currently the `expr` modul) are
feature-gated behind the feature `tensorflow_unstable` to prevent accidental
use. See http://doc.crates.io/manifest.html#the-features-section.
(We would prefer using an `#[unstable]` attribute, but that
[doesn't exist](https://github.com/rust-lang/rfcs/issues/1491) yet.)

## Other

This project is not directly affiliated with the TensorFlow project, although we
do intend to communicate and cooperate with them.

Developers and users are welcome to join
[#tensorflow-rust](http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23tensorflow-rust)
on irc.mozilla.org.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute code.

This is not an official Google product.

##For more information

* [API docs](https://google.github.io/tensorflow-rust)
* [TensorFlow website](http://tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)
