use crate::DataType;
use tensorflow_internal_macros::define_op;

define_op!(random_normal, RandomNormal, "RandomStandardNormal", args{x}, attrs {
    dtype: DataType => "dtype",
    seed?: i64 => "seed",
    seed2?: i64 => "seed2",
});
