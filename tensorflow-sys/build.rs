extern crate pkg_config;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

const LIBRARY: &'static str = "tensorflow";
const REPOSITORY: &'static str = "https://github.com/tensorflow/tensorflow.git";
const TARGET: &'static str = "libtensorflow.so";
const VERSION: &'static str = "0.10.0rc0";

macro_rules! get(($name:expr) => (ok!(env::var($name))));
macro_rules! ok(($expression:expr) => ($expression.unwrap()));

fn main() {
    if pkg_config::find_library(LIBRARY).is_ok() {
        return;
    }

    let output = PathBuf::from(&get!("OUT_DIR"));
    if !output.join(TARGET).exists() {
        let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join("target/source");
        if !Path::new(&source.join(".git")).exists() {
            run("git", |command| command.arg("clone")
                                        .arg(format!("--branch=v{}", VERSION))
                                        .arg("--recursive")
                                        .arg(REPOSITORY)
                                        .arg(&source));
        }
        run("./configure", |command| command.current_dir(&source));
        run("bazel", |command| command.current_dir(&source)
                                      .arg("build")
                                      .arg(format!("--jobs={}", get!("NUM_JOBS")))
                                      .arg("--compilation_mode=opt")
                                      .arg(format!("{}:{}", LIBRARY, TARGET)));
        ok!(fs::copy(source.join("bazel-bin").join(LIBRARY).join(TARGET), output.join(TARGET)));
    }

    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", output.display());
}

fn run<F>(name: &str, mut configure: F) where F: FnMut(&mut Command) -> &mut Command {
    let mut command = Command::new(name);
    if !ok!(configure(&mut command).status()).success() {
        panic!("failed to execute {:?}", command);
    }
}
