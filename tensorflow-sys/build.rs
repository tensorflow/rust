extern crate pkg_config;
extern crate semver;

use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process;
use std::process::Command;
use std::{env, fs};
use semver::Version;

const LIBRARY: &'static str = "tensorflow_c";
const REPOSITORY: &'static str = "https://github.com/tensorflow/tensorflow.git";
const TARGET: &'static str = "tensorflow:libtensorflow_c.so";
const TAG: &'static str = "v0.12.0";
const MIN_BAZEL: &'static str = "0.3.2";

macro_rules! get(($name:expr) => (ok!(env::var($name))));
macro_rules! ok(($expression:expr) => ($expression.unwrap()));
macro_rules! log {
    ($fmt:expr) => (println!(concat!("libtensorflow-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("libtensorflow-sys/build.rs:{}: ", $fmt), line!(), $($arg)*));
}
macro_rules! log_var(($var:ident) => (log!(concat!(stringify!($var), " = {:?}"), $var)));

fn main() {
    if pkg_config::find_library(LIBRARY).is_ok() {
        log!("Returning early because {} was already found", LIBRARY);
        return;
    }

    let output = PathBuf::from(&get!("OUT_DIR"));
    log_var!(output);
    let source = PathBuf::from(&get!("CARGO_MANIFEST_DIR")).join(format!("target/source-{}", TAG));
    log_var!(source);
    let lib_dir = output.join(format!("lib-{}", TAG));
    log_var!(lib_dir);
    if lib_dir.exists() {
        log!("Directory {:?} already exists", lib_dir);
    } else {
        log!("Creating directory {:?}", lib_dir);
        fs::create_dir(lib_dir.clone()).unwrap();
    }
    let library_path = lib_dir.join(format!("lib{}.so", LIBRARY));
    log_var!(library_path);
    if library_path.exists() {
        log!("{:?} already exists, not building", library_path);
    } else {
        if let Err(e) = check_bazel() {
            println!("cargo:error=Bazel must be installed at version {} or greater. (Error: {})", MIN_BAZEL, e);
            process::exit(1);
        }
        let target_path = &TARGET.replace(":", "/");
        log_var!(target_path);
        if !Path::new(&source.join(".git")).exists() {
            run("git", |command| command.arg("clone")
                                        .arg(format!("--branch={}", TAG))
                                        .arg("--recursive")
                                        .arg(REPOSITORY)
                                        .arg(&source));
            run("git", |command| command.current_dir(&source)
                                        .arg("apply")
                                        .arg("../../0001-Fix-zlib-dependency.patch"));
        }
        // Only configure if not previously configured.  Configuring runs a
        // `bazel clean`, which we don't want, because we want to be able to
        // continue from a cancelled build.
        let configure_hint_file_pb = source.join(".rust-configured");
        let configure_hint_file = Path::new(&configure_hint_file_pb);
        if !configure_hint_file.exists() {
            run("bash", |command| command.current_dir(&source).arg("-c").arg("yes ''|./configure"));
            File::create(configure_hint_file).unwrap();
        }
        run("bazel", |command| command.current_dir(&source)
                                      .arg("build")
                                      .arg(format!("--jobs={}", get!("NUM_JOBS")))
                                      .arg("--compilation_mode=opt")
                                      .arg(TARGET));
        let target_bazel_bin = source.join("bazel-bin").join(target_path);
        log!("Copying {:?} to {:?}", target_bazel_bin, library_path);
        fs::copy(target_bazel_bin, library_path).unwrap();
    }

    println!("cargo:rustc-link-lib=dylib={}", LIBRARY);
    println!("cargo:rustc-link-search={}", lib_dir.display());
}

fn run<F>(name: &str, mut configure: F) where F: FnMut(&mut Command) -> &mut Command {
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    log!("Executing {:?}", configured);
    if !ok!(configured.status()).success() {
        panic!("failed to execute {:?}", configured);
    }
    log!("Command {:?} finished successfully", configured);
}

// Building TF 0.11.0rc1 with Bazel 0.3.0 gives this error when running `configure`:
//   expected ConfigurationTransition or NoneType for 'cfg' while calling label_list but got string instead:     data.
//       ERROR: com.google.devtools.build.lib.packages.BuildFileContainsErrorsException: error loading package '': Extension file 'tensorflow/tensorflow.bzl' has errors.
// And the simple solution is to require Bazel 0.3.1 or higher.
fn check_bazel() -> Result<(), Box<Error>> {
    let mut command = Command::new("bazel");
    command.arg("version");
    log!("Executing {:?}", command);
    let out = try!(command.output());
    log!("Command {:?} finished successfully", command);
    let stdout = try!(String::from_utf8(out.stdout));
    let mut found_version = false;
    for line in stdout.lines() {
        if line.starts_with("Build label:") {
            found_version = true;
            let mut version_str = line.split(":").nth(1).unwrap()
                .split(" ").nth(1).unwrap().trim();
            if version_str.ends_with('-') {
                // hyphen is 1 byte long, so it's safe
                version_str = &version_str[..version_str.len() - 1];
            }
            let version = try!(Version::parse(version_str));
            let want = try!(Version::parse(MIN_BAZEL));
            if version < want {
                return Err(format!("Installed version {} is less than required version {}", version_str, MIN_BAZEL).into());
            }
        }
    }
    if !found_version {
        return Err("Did not find version number in `bazel version` output.".into());
    }
    Ok(())
}
