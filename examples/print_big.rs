use tensorflow::eager::{raw_ops, Context, ContextOptions};
use tensorflow::Tensor;

fn main() {
    let x: Vec<u64> = (0..100000).collect();

    let ctx = Context::new(ContextOptions::new()).unwrap();
    let t = Tensor::new(&[2, 50000])
        .with_values(&x)
        .unwrap()
        .into_handle(&ctx)
        .unwrap()
        .resolve::<u64>();

    dbg!(t.unwrap());
}
