# <img alt="SIG Rust TensorFlow" src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGRust.png" width="340"/>
[![Version](https://img.shields.io/crates/v/tensorflow.svg)](https://crates.io/crates/tensorflow)
[![Status](https://travis-ci.org/tensorflow/rust.svg?branch=master)](https://travis-ci.org/tensorflow/rust)

TensorFlow Rust provides idiomatic [Rust](https://www.rust-lang.org) language
bindings for [TensorFlow](https://www.tensorflow.org).

**Notice:** This project is still under active development and not guaranteed to have a
stable API. This is especially true because the underlying TensorFlow C API has not yet
been stabilized as well.

* [Documentation](https://tensorflow.github.io/rust/tensorflow/)
* [TensorFlow Rust Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/rust)
* [TensorFlow website](https://www.tensorflow.org)
* [TensorFlow GitHub page](https://github.com/tensorflow/tensorflow)

## Getting Started
Since this crate depends on the TensorFlow C API, it needs to be downloaded or compiled first. This
crate will automatically download or compile the TensorFlow shared libraries for you, but it is also
possible to manually install TensorFlow and the crate will pick it up accordingly.

### Prerequisites
If the TensorFlow shared libraries can already be found on your system, they will be used.  If your
system is x86-64 Linux or Mac, a prebuilt binary will be downloaded, and no special prerequisites
are needed.

Otherwise, the following dependencies are needed to compile and build this crate, which involves
compiling TensorFlow itself:

 - git
 - [bazel](https://bazel.build/)
 - Python Dependencies `numpy`, `dev`, `pip` and `wheel`
 - Optionally, CUDA packages to support GPU-based processing

The TensorFlow website provides detailed instructions on how to obtain and install said dependencies,
so if you are unsure please [check out the docs](https://www.tensorflow.org/install/install_sources)
 for further details.

Some of the examples use TensorFlow code written in Python and require a full TensorFlow
intallation.

### Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
tensorflow = "0.13.0"
```

and this to your crate root:

```rust
extern crate tensorflow;
```

Then run `cargo build -j 1`. The tensorflow-sys crate's 
[`build.rs`](https://github.com/tensorflow/rust/blob/f204b39/tensorflow-sys/build.rs#L44-L52)
now either downloads a pre-built, basic CPU only binary
([the default](https://github.com/tensorflow/rust/pull/65))
or compiles TensorFlow if forced to by an environment variable. If TensorFlow
is compiled during this process, since the full compilation is very memory
intensive, we recommend using the `-j 1` flag which tells cargo to use only one
task, which in turn tells TensorFlow to build with only one task. Though, if
you have a lot of RAM, you can obviously use a higher value.

To include the especially unstable API (which is currently the `expr` module),
use `--features tensorflow_unstable`.

For now, please see the [Examples](https://github.com/tensorflow/rust/tree/master/examples) for more
details on how to use this binding.

## GPU Support

To enable GPU support, use the `tensorflow_gpu` feature in your Cargo.toml:

```
[dependencies]
tensorflow = { version = "0.13.0", features = ["tensorflow_gpu"] }
```

## Manual TensorFlow Compilation

If you want to work against unreleased/unsupported TensorFlow versions or use a build optimized for
your machine, manual compilation is the way to go.

See [tensorflow-sys/README.md](tensorflow-sys/README.md) for details.

## FAQ's

### Why does the compiler say that parts of the API don't exist?
The especially unstable parts of the API (which is currently the `expr` module) are
feature-gated behind the feature `tensorflow_unstable` to prevent accidental
use. See http://doc.crates.io/manifest.html#the-features-section.
(We would prefer using an `#[unstable]` attribute, but that
[doesn't exist](https://github.com/rust-lang/rfcs/issues/1491) yet.)

### How do I...?
Try the [documentation](https://tensorflow.github.io/rust/tensorflow/) first, and see if it answers
your question.  If not, take a look at the examples folder.  Note that there may not be an example
for your exact question, but it may be answered by an example demonstrating something else.

If none of the above help, you can ask your question on
[TensorFlow Rust Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/rust).

## Contributing
Developers and users are welcome to join the
[TensorFlow Rust Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/rust).

Developers and users are also welcome to join
[#tensorflow-rust](https://chat.mibbit.com/?server=irc.mozilla.org&channel=%23tensorflow-rust)
on irc.mozilla.org, although the Google Group is more likely to provide a response.

Please read the [contribution guidelines](CONTRIBUTING.md) on how to contribute code.

This is not an official Google product.

RFCs are [issues tagged with RFC](https://github.com/tensorflow/rust/labels/rfc).
Check them out and comment. Discussions are welcomed. After all, that is the purpose of
Request For Comment!

## License
This project is licensed under the terms of the [Apache 2.0 license](LICENSE).
