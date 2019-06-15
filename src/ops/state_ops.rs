use crate::DataType;
use crate::Operation;
use crate::Output;
use crate::Result;
use crate::Scope;
use crate::Shape;
use crate::Tensor;
use crate::TensorType;
use tensorflow_macros::define_op;

define_op!(assign, Assign, "Assign", args {a, b});

define_op!(placeholder, Placeholder, "Placeholder", attrs {
    data_type: DataType => "dtype",
    shape: Shape => "shape",
});
