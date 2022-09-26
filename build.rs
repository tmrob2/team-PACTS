fn main() {
    println!("cargo:rustc-link-search=CXSparse/Lib/");
    println!("cargo:rustc-link-lib=cxsparse");
}