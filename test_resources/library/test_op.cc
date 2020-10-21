/*
A test Library for `test_library_load` TF_GetOpList functionality in lib.rs

To compile, after running `pip install tensorflow==2.3.0`:
  TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
  g++ -std=c++11 -shared test_op.cc -o test_op.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("TestOpList").Doc(R"doc(Used to test TF_GetOpList)doc");

}  // namespace tensorflow
