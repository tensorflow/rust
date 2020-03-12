use crate::DataType;
use tensorflow_internal_macros::define_op;

#[deprecated(note = "Use random_standard_normal instead.", since = "0.15.0")]
define_op!(random_normal, RandomNormal, "RandomStandardNormal", args{x}, attrs {
    dtype: DataType => "dtype",
    seed?: i64 => "seed",
    seed2?: i64 => "seed2",
});
