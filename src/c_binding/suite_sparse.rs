#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

pub type size_t = ::std::os::raw::c_ulong;
pub type cs_di = cs_di_sparse;
pub type cs_din = cs_di_numeric;
pub type cs_dis = cs_di_symbolic;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cs_di_symbolic {
    pub pinv: *mut ::std::os::raw::c_int,
    pub q: *mut ::std::os::raw::c_int,
    pub parent: *mut ::std::os::raw::c_int,
    pub cp: *mut ::std::os::raw::c_int,
    pub leftmost: *mut ::std::os::raw::c_int,
    pub m2: ::std::os::raw::c_int,
    pub lnz: f64,
    pub unz: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cs_di_numeric {
    pub L: *mut cs_di,
    pub U: *mut cs_di,
    pub pinv: *mut ::std::os::raw::c_int,
    pub B: *mut f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cs_di_sparse {
    pub nzmax: ::std::os::raw::c_int,
    pub m: ::std::os::raw::c_int,
    pub n: ::std::os::raw::c_int,
    pub p: *mut ::std::os::raw::c_int,
    pub i: *mut ::std::os::raw::c_int,
    pub x: *mut f64,
    pub nz: ::std::os::raw::c_int,
}

extern "C" {
    pub fn cs_di_spalloc(
        m: ::std::os::raw::c_int,
        n: ::std::os::raw::c_int,
        nzmax: ::std::os::raw::c_int,
        values: ::std::os::raw::c_int,
        t: ::std::os::raw::c_int,
    ) -> *mut cs_di;
}
extern "C" {
    pub fn cs_di_realloc(
        p: *mut ::std::os::raw::c_void,
        n: ::std::os::raw::c_int,
        size: size_t,
        ok: *mut ::std::os::raw::c_int,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn cs_di_spfree(A: *mut cs_di) -> *mut cs_di;
}
extern "C" {
    pub fn cs_di_sprealloc(A: *mut cs_di, nzmax: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_malloc(n: ::std::os::raw::c_int, size: size_t) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn cs_di_entry(
        T: *mut cs_di,
        i: ::std::os::raw::c_int,
        j: ::std::os::raw::c_int,
        x: f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_lusol(
        order: ::std::os::raw::c_int,
        A: *const cs_di,
        b: *mut f64,
        tol: f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_gaxpy(A: *const cs_di, x: *const f64, y: *mut f64) -> ::std::os::raw::c_int;
}

extern "C" {
    pub fn cs_di_multiply(A: *const cs_di, B: *const cs_di) -> *mut cs_di;
}
extern "C" {
    pub fn cs_di_qrsol(
        order: ::std::os::raw::c_int,
        A: *const cs_di,
        b: *mut f64,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_transpose(A: *const cs_di, values: ::std::os::raw::c_int) -> *mut cs_di;
}
extern "C" {
    pub fn cs_di_compress(T: *const cs_di) -> *mut cs_di;
}
extern "C" {
    pub fn cs_di_norm(A: *const cs_di) -> f64;
}
extern "C" {
    pub fn cs_di_print(A: *const cs_di, brief: ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_ipvec(
        p: *const ::std::os::raw::c_int,
        b: *const f64,
        x: *mut f64,
        n: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_lsolve(L: *const cs_di, x: *mut f64) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_ltsolve(L: *const cs_di, x: *mut f64) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_lu(A: *const cs_di, S: *const cs_dis, tol: f64) -> *mut cs_din;
}
extern "C" {
    pub fn cs_di_sqr(
        order: ::std::os::raw::c_int,
        A: *const cs_di,
        qr: ::std::os::raw::c_int,
    ) -> *mut cs_dis;
}
extern "C" {
    pub fn cs_di_usolve(U: *const cs_di, x: *mut f64) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn cs_di_add(A: *const cs_di, B: *const cs_di, alpha: f64, beta: f64) -> *mut cs_di;
}