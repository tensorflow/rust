use tensorflow::eager as tf;
use tensorflow::{DataType, Tensor};

fn main() {
    let mut v = vec![0u8; 600 * 800 * 3];
    for i in 0..v.len() / 3 {
        v[3 * i + 0] = (i % 256) as u8;
        v[3 * i + 1] = 255 - (i % 256) as u8;
    }
    let img = Tensor::new(&[600, 800, 3]).with_values(&v).unwrap();
    let buf = tf::encode_png(img.clone()).unwrap();
    tf::write_file(Tensor::from(String::from("input.png")), buf).unwrap();

    let images = tf::expand_dims(img, Tensor::from(&[0])).unwrap();
    let size = Tensor::from([224, 224]);
    let scale = Tensor::from([224.0f32 / 600.0f32, 224.0f32 / 800.0f32]);
    let translation = Tensor::from([0f32, 0f32]);
    let smalls = tf::scale_and_translate(images, size, scale, translation).unwrap();

    let img = tf::squeeze(smalls).unwrap();
    let args = tf::Cast {
        DstT: Some(DataType::UInt8),
        ..Default::default()
    };
    let img = tf::cast_with_args(img, &args).unwrap();
    let buf = tf::encode_png(img).unwrap();
    tf::write_file(Tensor::from(String::from("small.png")), buf).unwrap();
}
