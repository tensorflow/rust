# tensorflow-sys [![Version][version-icon]][version-page]

The package provides bindings to [TensorFlow][tensorflow].

## Requirements

The build prerequisites can be found on the [corresponding
page][tensorflow-setup] of TensorFlow’s documentation. In particular,
[Bazel][bazel], [NumPy][numpy], and [SWIG][swig] are assumed to be installed.

## Configuration

The compilation process is configured via a number of environment variables, all
of which can be found in TensorFlow’s [configure][tensorflow-configure] script.
In particular, `TF_NEED_CUDA` is used to indicate if GPU support is needed.

[bazel]: http://www.bazel.io
[numpy]: http://www.numpy.org
[swig]: http://www.swig.org
[tensorflow]: https://www.tensorflow.org
[tensorflow-configure]: https://github.com/tensorflow/tensorflow/blob/r0.9/configure
[tensorflow-setup]: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
[version-icon]: https://img.shields.io/crates/v/tensorflow-sys.svg
[version-page]: https://crates.io/crates/tensorflow-sys
