// -*-  indent-tabs-mode:nil; tab-width:2;  -*-
extern crate random;
extern crate tensorflow;

use std::error::Error;
use std::result::Result;
use std::process::exit;
use tensorflow::Code;
use tensorflow::expr::Compiler;
use tensorflow::expr::Placeholder;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
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

struct Checker {
  success: bool,
  epsilon: f32,
}

impl Checker {
  fn new(epsilon: f32) -> Self {
    Checker {
      success: true,
      epsilon: epsilon,
    }
  }
  
  fn check(&mut self, name: &str, expected: f32, actual: f32) {
    let success = (expected - actual).abs() < self.epsilon;
    println!("Checking {}: expected {}, got {}. {}",
             name, expected, actual, if success {"Success!"} else {"FAIL"});
    self.success &= success;
  }

  fn result(&self) -> Result<(), Box<Error>> {
    if self.success {
      Ok(())
    } else {
      Err(Box::new(try!(Status::new_set(Code::Internal, "At least one check failed"))))
    }
  }
}

fn run() -> Result<(), Box<Error>> {
  // Build the graph
  let mut g = Graph::new();
  let y_node = {
    let mut compiler = Compiler::new(&mut g);
    let x_expr = <Placeholder<f32>>::new_expr(&vec![2], "x");
    try!(compiler.compile(x_expr * 2.0f32 + 1.0f32))
  };
  let x_node = try!(g.operation_by_name_required("x"));
  // This is another valid way to get x_node and y_node:
  // let (x_node, y_node) = {
  //   let mut compiler = Compiler::new(&mut g);
  //   let x_expr = <Placeholder<f32>>::new_expr(&vec![2], "x");
  //   let x_node = try!(compiler.compile(x_expr.clone()));
  //   let y_node = try!(compiler.compile(x_expr * 2.0f32 + 1.0f32));
  //   (x_node, y_node)
  // };
  let options = SessionOptions::new();
  let mut session = try!(Session::new(&options, &g));

  // Evaluate the graph.
  let mut x = <Tensor<f32>>::new(&[2]);
  x[0] = 2.0;
  x[1] = 3.0;
  let mut step = StepWithGraph::new();
  step.add_input(&x_node, 0, &x);
  let output_token = step.request_output(&y_node, 0);
  session.run(&mut step).unwrap();

  // Check our results.
  let output_tensor = try!(step.take_output::<f32>(output_token));
  let data = output_tensor.data();
  let mut checker = Checker::new(1e-3);
  checker.check("data[0]", 5.0, data[0]);
  checker.check("data[1]", 7.0, data[1]);
  checker.result()
}
