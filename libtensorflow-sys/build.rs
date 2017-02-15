extern crate bindgen;

use std::env;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::result::Result;

fn log_env_var(log: &mut Write, var: &str) -> Result<(), io::Error> {
    match env::var(var) {
        Ok(s) => writeln!(log, "{}={}", var, s),
        Err(env::VarError::NotPresent) => writeln!(log, "{} is not present", var),
        Err(env::VarError::NotUnicode(_)) => writeln!(log, "{} is not unicode", var),
    }
}

fn find_header(name: &str) -> Option<PathBuf> {
    let cpath_str = match env::var("CPATH") {
        Ok(s) => s,
        Err(_) => "".to_string(),
    } + ":/usr/include:/usr/local/include";
    for s in cpath_str.split(":") {
        if s != "" {
            let full = Path::new(s).join(name);
            let exists = match fs::metadata(&full) {
                Ok(m) => m.is_file(),
                Err(_) => false,
            };
            if exists {
                return Some(full);
            }
        }
    }
    None
}

fn main() {
    let out_dir_str = env::var("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir_str);
    let dest_path = out_dir.join("ffi.rs");
    let log_path = out_dir.join("build.log");

    let mut log = match File::create(&log_path) {
        Ok(f) => f,
        Err(_) => panic!(format!("Unable to open file {}", log_path.to_str().unwrap())),
    };

    let mut bindings = bindgen::builder();
    bindings.forbid_unknown_types();

    log_env_var(&mut log, "CPATH").unwrap();
    let tf_header = match find_header("tensor_c_api.h") {
        Some(p) => p,
        None => panic!("Unable to find tensor_c_api.h"),
    };

    bindings.link("tensorflow");
    bindings.match_pat("tensor_c_api.h");
    bindings.header(tf_header.to_str().unwrap());

    let bindings = bindings.generate();
    let bindings = bindings.unwrap();
    bindings.write_to_file(&dest_path).unwrap();

    println!("cargo:include={}", dest_path.to_str().unwrap());
}
