use tensorflow_macros::define_op;

define_op!(zeros_like, ZerosLike, "ZerosLike", args { x });
