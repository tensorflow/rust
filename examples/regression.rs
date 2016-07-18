// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
extern crate random;
extern crate tensorflow;

use random::Source;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process::exit;
use tensorflow::Code;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::Step;
use tensorflow::Tensor;

fn main() {
  // Putting the main code in another function serves two purposes:
  // 1. We can use the try! macro.
  // 2. We can call exit safely, which does not run any destructors.
  exit(match run() {
    Ok(_) => 0,
    Err(e) => {
      println!("{}", e);
      1
    }
  })
}

fn run() -> Result<(), Box<Error>> {
  let filename = "regression-model/model.pb"; // y = w * x + b
  if !Path::new(filename).exists() {
    return Err(Box::new(Status::new_set(Code::NotFound, &format!(
      "Run 'python regression.py' to generate {} and try again.", filename)).unwrap()));
  }

  // Generate some test data.
  let w = 0.1;
  let b = 0.3;
  let num_points = 100;
  let steps = 201;
  let mut rand = random::default();
  let mut x_tensor = Tensor::new(&[num_points as u64]);
  let mut y_tensor = Tensor::new(&[num_points as u64]);
  {
    let mut x = x_tensor.data_mut();
    let mut y = y_tensor.data_mut();
    for i in 0..num_points {
      x[i] = (2.0 * rand.read::<f64>() - 1.0) as f32;
      y[i] = w * x[i] + b;
    }
  }

  // Load the computation graph defined by regression.py.
  let mut session = try!(Session::new(&SessionOptions::new()));
  let mut proto = Vec::new();
  try!(try!(File::open(filename)).read_to_end(&mut proto));
  try!(session.extend_graph(&proto));

  // Load the test data into the session.
  let mut init_step = Step::new();
  try!(init_step.add_input("x", &x_tensor));
  try!(init_step.add_input("y", &y_tensor));
  try!(init_step.add_target("init"));
  try!(session.run(&mut init_step));

  // Train the model.
  let mut train_step = Step::new();
  try!(train_step.add_input("x", &x_tensor));
  try!(train_step.add_input("y", &y_tensor));
  try!(train_step.add_target("train"));
  for _ in 0..steps {
    try!(session.run(&mut train_step));
  }

  // Grab the data out of the session.
  let mut output_step = Step::new();
  let w_ix = try!(output_step.request_output("w"));
  let b_ix = try!(output_step.request_output("b"));
  try!(session.run(&mut output_step));

  // Check our results.
  let w_hat: f32 = try!(output_step.take_output(w_ix)).data()[0];
  let b_hat: f32 = try!(output_step.take_output(b_ix)).data()[0];
  println!("Checking w: expected {}, got {}. {}", w, w_hat,
           if (w - w_hat).abs() < 1e-3 {"Success!"} else {"FAIL"});
  println!("Checking b: expected {}, got {}. {}", b, b_hat,
           if (b - b_hat).abs() < 1e-3 {"Success!"} else {"FAIL"});
  Ok(())
}
