// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
extern crate tensorflow;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process::exit;
use tensorflow::Code;
use tensorflow::DeprecatedSession;
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
  let filename = "examples/addition-model/model.pb"; // z = x + y
  if !Path::new(filename).exists() {
    return Err(Box::new(Status::new_set(Code::NotFound, &format!(
      "Run 'python addition.py' to generate {} and try again.", filename)).unwrap()));
  }

  // Create input variables for our addition
  let mut x = Tensor::new(&[1]);
  x[0] = 2i32;
  let mut y = Tensor::new(&[1]);
  y[0] = 40i32;

  // Load the computation graph defined by regression.py.
  let mut session = try!(DeprecatedSession::new(&SessionOptions::new()));
  let mut proto = Vec::new();
  try!(try!(File::open(filename)).read_to_end(&mut proto));
  try!(session.extend_graph(&proto));


  // Run the Step
  let mut step = Step::new();
  step.add_input("x", &x).unwrap();
  step.add_input("y", &y).unwrap();
  let z = try!(step.request_output("z"));
  try!(session.run(&mut step));

  // Check our results.
  let z_res: i32 = try!(step.take_output(z)).data()[0];
  println!("{:?}", z_res);

  Ok(())
}
