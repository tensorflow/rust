/*
A test Library for `test_library_load` TF_GetOpList functionality in lib.rs

To compile, see instructions in RELEASING.md.
*/
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("TestOpList").Doc(R"doc(Used to test TF_GetOpList)doc");

}  // namespace tensorflow
