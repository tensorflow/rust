extern crate bindgen;

use std::env;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::result::Result;

/// Note that this modifies the contents of the TensorFlow source tree!
fn build_tensorflow(tensorflow_dir: &Path) -> Result<PathBuf, String> {
  // This is correct.  There is a tensorflow directory inside the TensorFlow source tree.
  let patch_dir = tensorflow_dir.join("tensorflow/libtensorflow");
  if let Err(_) = fs::create_dir_all(&patch_dir) {
    return Err(format!("Unable to create {}", patch_dir.to_str().unwrap()));
  }
  let build_file_path = patch_dir.join("BUILD");
  let is_file = match fs::metadata(&build_file_path) {
    Ok(m) => m.is_file(),
    Err(_) => false,
  };
  if !is_file {
    let mut build_file = match File::create(&build_file_path) {
      Ok(f) => f,
      Err(_) => return Err(format!("Unable to open file {}", build_file_path.to_str().unwrap())),
    };
    write!(build_file, "{}", r###"
      cc_binary(
          name = "libtensorflow.so",
          linkshared = 1,
          deps = [
              "//tensorflow/core:tensorflow",
          ],
      )
    "###.trim()).unwrap();
  }
  let lib_file_path = tensorflow_dir.join("bazel-bin/tensorflow/libtensorflow/libtensorflow.so");
  let configure_output = Command::new("./configure")
    .current_dir(tensorflow_dir)
    .output()
    .unwrap();
  assert!(configure_output.status.success(), "configure failed with output: {}\nand error: {}", String::from_utf8(configure_output.stdout).unwrap(), String::from_utf8(configure_output.stderr).unwrap());
  let output = Command::new("bazel")
    .arg("build")
    .arg(":libtensorflow.so")
    .current_dir(patch_dir)
    .output()
    .unwrap();
  assert!(output.status.success(), "Blaze failed with output: {}\nand error: {}", String::from_utf8(output.stdout).unwrap(), String::from_utf8(output.stderr).unwrap());
  Ok(lib_file_path)
}

fn clone_tensorflow(url: &str, tensorflow_dir: &Path) -> Result<(), String> {
  // Wanted to use libgit2, but the Rust bindings don't support --recurse-submodules,
  // and doing anything like 'git submodule update --init' is a PITA.
  match fs::metadata(tensorflow_dir) {
    Ok(_) => {},
    Err(_) => {
      let output = Command::new("git")
        .arg("clone")
        .arg("--recurse-submodules")
        .arg(url)
        .current_dir(tensorflow_dir.parent().unwrap())
        .output()
        .unwrap();
      if !output.status.success() {
        return Err(format!("Git failed with output: {}\nand error: {}", String::from_utf8(output.stdout).unwrap(), String::from_utf8(output.stderr).unwrap()));
      }
    }
  }

  Ok(())
}

fn log_env_var(log: &mut Write, var: &str) -> Result<(), io::Error> {
  match env::var(var) {
    Ok(s) => writeln!(log, "{}={}", var, s),
    Err(env::VarError::NotPresent) => writeln!(log, "{} is not present", var),
    Err(env::VarError::NotUnicode(_)) => writeln!(log, "{} is not unicode", var),
  }
}

fn main() {
  let out_dir_str = env::var("OUT_DIR").unwrap();
  let out_dir = Path::new(&out_dir_str);
  let lib_dir = out_dir.join("lib/");
  let dest_path = lib_dir.join("ffi.rs");
  let tensorflow_dir = out_dir.join("tensorflow");
  let log_path = out_dir.join("build.log");

  let mut log = match File::create(&log_path) {
    Ok(f) => f,
    Err(_) => panic!(format!("Unable to open file {}", log_path.to_str().unwrap())),
  };

  log_env_var(&mut log, "TENSORFLOW_GIT").unwrap();
  let lib_file_path = match env::var("TENSORFLOW_GIT") {
    Ok(url) => {
      writeln!(log, "Cloning TensorFlow...").unwrap();
      clone_tensorflow(&url, &tensorflow_dir).unwrap();
      writeln!(log, "Building TensorFlow...").unwrap();
      build_tensorflow(&tensorflow_dir).unwrap()
    },
    Err(_) => panic!("The TENSORFLOW_GIT environment variable is currently required."),
  };

  let _ = fs::create_dir(&lib_dir);

  let mut bindings = bindgen::builder();
  bindings.forbid_unknown_types();

  let tf_header = tensorflow_dir.join("tensorflow/core/public/tensor_c_api.h");
  let tf_lib_dir = lib_file_path.parent().unwrap();

  bindings.link("tensorflow");
  bindings.match_pat("tensor_c_api.h");
  bindings.header(tf_header.to_str().unwrap());

  let bindings = bindings.generate();
  let bindings = bindings.unwrap();
  bindings.write_to_file(&dest_path).unwrap();

  println!("cargo:include={}", dest_path.to_str().unwrap());
  println!("cargo:rustc-link-search=native={}", tf_lib_dir.to_str().unwrap());
}
