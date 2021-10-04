use tensorflow::eager::{raw_ops, Context, ContextOptions};
use tensorflow::{DataType, Result, Tensor};

fn main() -> Result<()> {
    let opts = ContextOptions::new();
    let ctx = &Context::new(opts).unwrap();

    let mut v = vec![0u8; 600 * 800 * 3];
    for i in 0..v.len() / 3 {
        v[3 * i + 0] = (i % 256) as u8;
        v[3 * i + 1] = 255 - (i % 256) as u8;
    }
    let img = Tensor::new(&[600, 800, 3]).with_values(&v)?;
    let buf = raw_ops::encode_png(ctx, img.clone())?;
    raw_ops::write_file(ctx, Tensor::from(String::from("input.png")), buf)?;

    let images = raw_ops::expand_dims(ctx, img, Tensor::from(&[0]))?;
    let size = Tensor::from([224, 224]);
    let cast_to_float = raw_ops::Cast::new().DstT(DataType::Float);
    let scale = raw_ops::div(
        ctx,
        cast_to_float.call(ctx, size.clone())?,
        Tensor::from([600.0f32, 800.0f32]),
    )?;
    let translation = Tensor::from([0f32, 0f32]);
    let smalls = raw_ops::scale_and_translate(ctx, images, size, scale, translation)?;

    let img = raw_ops::squeeze(ctx, smalls)?;
    let cast_to_uint8 = raw_ops::Cast::new().DstT(DataType::UInt8);
    let img = cast_to_uint8.call(ctx, img)?;
    let buf = raw_ops::encode_png(ctx, img)?;
    raw_ops::write_file(ctx, Tensor::from(String::from("small.png")), buf)?;

    Ok(())
}
