use tensorflow::eager::raw_ops;
use tensorflow::{DataType, Result, Tensor};

fn main() -> Result<()> {
    let mut v = vec![0u8; 600 * 800 * 3];
    for i in 0..v.len() / 3 {
        v[3 * i + 0] = (i % 256) as u8;
        v[3 * i + 1] = 255 - (i % 256) as u8;
    }
    let img = Tensor::new(&[600, 800, 3]).with_values(&v)?;
    let buf = raw_ops::encode_png(img.clone())?;
    raw_ops::write_file(Tensor::from(String::from("input.png")), buf)?;

    let images = raw_ops::expand_dims(img, Tensor::from(&[0]))?;
    let size = Tensor::from([224, 224]);
    let scale = raw_ops::div(
        raw_ops::cast_with_args(
            size.clone(),
            &raw_ops::Cast {
                DstT: Some(DataType::Float),
                ..Default::default()
            },
        )?,
        Tensor::from([600.0f32, 800.0f32]),
    )?;
    let translation = Tensor::from([0f32, 0f32]);
    let smalls = raw_ops::scale_and_translate(images, size, scale, translation)?;

    let img = raw_ops::squeeze(smalls)?;
    let args = raw_ops::Cast {
        DstT: Some(DataType::UInt8),
        ..Default::default()
    };
    let img = raw_ops::cast_with_args(img, &args)?;
    let buf = raw_ops::encode_png(img)?;
    raw_ops::write_file(Tensor::from(String::from("small.png")), buf)?;

    Ok(())
}
