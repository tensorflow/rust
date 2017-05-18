# <img alt="TensorFlow" src="https://www.tensorflow.org/images/tf_logo_transp.png" width="170"/> Rust Binding
[![Version](https://img.shields.io/crates/v/tensorflow.svg)](https://crates.io/crates/tensorflow)
[![Status](https://travis-ci.org/tensorflow/rust.svg?branch=master)](https://travis-ci.org/tensorflow/rust)

TensorFlow Rust provides idiomatic [Rust](https://www.rust-lang.org) language
bindings for [TensorFlow](http://tensorflow.org).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

* [Documentation](https://tensorflow.github.io/rust/tensorflow/)
* [TensorFlow website](http://tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)

## Getting Started
Since this crate depends on the TensorFlow C API, it needs to be compiled first. This crate will
automatically compile TensorFlow for you, but it is also possible to manually install TensorFlow
and the crate will pick it up accordingly.

### Prerequisites
The following dependencies are needed to compile and build this crate (assuming TensorFlow itself
  should also be compiled transparently):

 - git
 - [bazel](https://bazel.build/)
 - Python Dependencies `numpy`, `dev`, `pip` and `wheel`
 - Optionally, CUDA packages to support GPU-based processing

The TensorFlow website provides detailed instructions on how to obtain and install said dependencies,
so if you are unsure please [check out the docs](https://www.tensorflow.org/install/install_sources)
 for further details.

### Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
tensorflow = "0.4.0"
```

and this to your crate root:

```rust
extern crate tensorflow;
```

Then run `cargo build -j 1`. Since TensorFlow is built during this process, and
the TensorFlow build is very memory intensive, we recommend using the `-j 1`
flag which tells cargo to use only one task, which in turn tells TensorFlow to
build with only one task. Of course, if you have a lot of RAM, you can use a
higher value.

To include the especially unstable API (which is currently the `expr` module),
use `--features tensorflow_unstable`.

For now, please see the [Examples](https://github.com/tensorflow/rust/tree/master/examples) for more
details on how to use this binding.

## Manual TensorFlow Compilation
If you don't want to build TensorFlow after every `cargo clean` or you want to work against
unreleased/unsupported TensorFlow versions, manual compilation is the way to go.

See [TensorFlow from source](https://www.tensorflow.org/install/install_sources) first.
The Python/pip steps are not necessary, but building `tensorflow:libtensorflow.so` is.

In short:

1. Install [SWIG](http://www.swig.org) and [NumPy](http://www.numpy.org).  The
   version from your distro's package manager should be fine for these two.
2. [Install Bazel](http://bazel.io/docs/install.html), which you may need to do
   from source.
3. `git clone https://github.com/tensorflow/tensorflow`
4. `cd tensorflow`
5. `./configure`
6. `bazel build --compilation_mode=opt --copt=-march=native --jobs=1 tensorflow:libtensorflow.so`

   Using `--jobs=1` is recommended unless you have a lot of RAM, because
   TensorFlow's build is very memory intensive.

Copy `$TENSORFLOW_SRC/bazel-bin/tensorflow/libtensorflow.so` to `/usr/local/lib`.
If this is not possible, add `$TENSORFLOW_SRC/bazel-bin/tensorflow` to
`LD_LIBRARY_PATH`.

You may need to run `ldconfig` to reset `ld`'s cache after copying `libtensorflow.so`.

**OSX Note**: If you are running on OSX, there is a
[Homebrew PR](https://github.com/Homebrew/homebrew-core/pull/10273) in process which, once merged,
will make it easy to install `libtensorflow` wihout hassle. In the meantime, you can take a look at
[snipsco/tensorflow-build](https://github.com/snipsco/tensorflow-build) which provides a homebrew
tap that does essentially the same.

## FAQ's

### Why does the compiler say that parts of the API don't exist?
The especially unstable parts of the API (which is currently the `expr` modul) are
feature-gated behind the feature `tensorflow_unstable` to prevent accidental
use. See http://doc.crates.io/manifest.html#the-features-section.
(We would prefer using an `#[unstable]` attribute, but that
[doesn't exist](https://github.com/rust-lang/rfcs/issues/1491) yet.)

## Contributing
Developers and users are welcome to join
[#tensorflow-rust](http://chat.mibbit.com/?server=irc.mozilla.org&channel=%23tensorflow-rust)
on irc.mozilla.org.

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute code.

This is not an official Google product.

RFCs are [issues tagged with RFC](https://github.com/tensorflow/rust/labels/rfc).
Check them out and comment. Discussions are welcome. After all, thats what a Request For
Comment is for!

## License
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/tensorflow/rust/blob/master/LICENSE).
