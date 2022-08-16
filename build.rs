use std::env;

fn main() {
    println!("cargo:rustc-link-search={}/SuiteSparse-5.11.0/CXSparse/Lib/", env::var("HOME").unwrap());
    println!("cargo:rustc-link-lib=cxsparse");
}