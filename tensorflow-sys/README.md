# tensorflow-sys [![Version][version-icon]][version-page]

The package provides bindings to [TensorFlow][tensorflow].

## Requirements

The build prerequisites can be found on the [corresponding
page][tensorflow-setup] of TensorFlow’s documentation. In particular,
[Bazel][bazel], [NumPy][numpy], and [SWIG][swig] are assumed to be installed.

## GPU Support

To enable GPU support, use the `tensorflow_gpu` feature in your Cargo.toml:

```
[dependencies]
tensorflow-sys = { version = "0.20.0", features = ["tensorflow_gpu"] }
```

## Manual TensorFlow Compilation

If you want to work against unreleased/unsupported TensorFlow versions or use a build optimized for
your machine, manual compilation is the way to go.

See [TensorFlow from source](https://www.tensorflow.org/install/install_sources) first.
The Python/pip steps are not necessary, but building `tensorflow:libtensorflow.so` is.

In short:

1. Install [SWIG](http://www.swig.org) and [NumPy](http://www.numpy.org).  The
   version from your distro's package manager should be fine for these two.
2. [Install Bazel](https://bazel.io/docs/install.html), which you may need to do
   from source.  You will likely need an up-to-date version.
3. `git clone https://github.com/tensorflow/tensorflow`
4. `cd tensorflow`
5. `./configure`
6. `bazel build --compilation_mode=opt --copt=-march=native --jobs=1 tensorflow:libtensorflow.so`

   Using `--jobs=1` is recommended unless you have a lot of RAM, because
   TensorFlow's build is very memory intensive.

Copy `$TENSORFLOW_SRC/bazel-bin/tensorflow/libtensorflow.so` and `libtensorflow_framework.so` to
`/usr/local/lib`.  You may need to run `ldconfig` to reset `ld`'s cache after copying
`libtensorflow.so`.

Generate tensorflow.pc by running the following (where $TENSORFLOW_VERSION is the version number of
TensorFlow that you compiled, *not* the version number of the Rust crate):

```
$TENSORFLOW_SRC/tensorflow/c/generate-pc.sh --prefix=/usr/local --version=$TENSORFLOW_VERSION
```

This generates tensorflow.pc in the current folder. Copy this to your PKG_CONFIG_PATH (may be
`/usr/lib/pkgconfig`).  To verify that the library is installed correctly, run
`pkg-config --libs tensorflow`.

If you previously compiled this crate, you may need to run `cargo clean` before the manually
compiled library will be picked up.

**macOS Note**: Via [Homebrew](https://brew.sh/), you can just run
`brew install libtensorflow`.

## Resources

[bazel]: http://www.bazel.io
[numpy]: http://www.numpy.org
[swig]: http://www.swig.org
[tensorflow]: https://www.tensorflow.org
[tensorflow-configure]: https://github.com/tensorflow/tensorflow/blob/r0.9/configure
[tensorflow-setup]: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
[version-icon]: https://img.shields.io/crates/v/tensorflow-sys.svg
[version-page]: https://crates.io/crates/tensorflow-sys
