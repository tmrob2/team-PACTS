#![allow(non_snake_case)]

pub mod scpm;
pub mod agent;
pub mod dfa;
pub mod parallel;
pub mod algorithm;
pub mod c_binding;

use std::hash::Hash;
use pyo3::prelude::*;
use hashbrown::HashMap;
use scpm::model::{build_model, SCPM, MOProductMDP};
use agent::agent::{Agent, Team};
use dfa::dfa::{DFA, Mission};
use parallel::{threaded::process_mdps, grid};
use c_binding::suite_sparse::*;
extern crate blis_src;
extern crate cblas_sys;
use cblas_sys::{cblas_dcopy, cblas_dgemv, cblas_dscal, cblas_ddot};
use algorithm::dp::value_iteration;


const UNSTABLE_POLICY: i32 = 5;
// -----------------------------------------------------------
//  Matrix Types and Construction
// -----------------------------------------------------------


/// Construct a sparse matrix data structure for use with CXSparse
/// a sparse blas library for sparse matrix algebra. 
pub struct SparseMatrix {
    pub m: *mut cs_di,
    pub nr: usize,
    pub nc: usize,
    pub nnz: usize
}

/// Construct a dense matrix data structure for use with BLAS
pub struct DenseMatrix{
    // matrix values in 1d format
    pub m: Vec<f64>,
    // number of rows in the matrix
    pub rows: usize,
    // number of columns in the matrix
    pub cols: usize
}

impl DenseMatrix {
    pub fn new(n: usize, m: usize) -> Self {
        DenseMatrix {
            m: Vec::new(),
            rows: n,
            cols: m
        }
    }
}

#[derive(Debug)]
// We should be able to run Rayon with this structure
pub struct COO {
    pub nzmax: i32,
    pub nr: i32,
    pub nc: i32,
    pub i: Vec<i32>,
    pub j: Vec<i32>,
    pub x: Vec<f64>,
    pub nz: i32,
}

pub fn construct_blas_matrix(
    n: usize,
    m: usize,
    transitions: &HashMap<(i32, i32), Vec<(i32, f64)>>,
    states: &[i32],
    actions: &[i32]
) -> DenseMatrix {
    let mut P: Vec<f64> = vec![0.; n * m];
    let mut matrix = DenseMatrix::new(n, m);
    for action in actions.iter() {
        for state in states.iter() {
            match transitions.get(&(*state, *action)) {
                None => { }
                Some(v) => {
                    for (sprime, p) in v.iter() {
                        // col major format
                        P[*sprime as usize * m + *state as usize] = *p;
                    }
                }
            }
        }
    }
    matrix.m = P;
    matrix
}

// -----------------------------------------------------------
//  CSSparse BLAS functions
// -----------------------------------------------------------

pub fn create_sparse_matrix(m: i32, n: i32, rows: &[i32], cols: &[i32], x: &[f64])
                            -> *mut cs_di {
    unsafe {
        let T: *mut cs_di = cs_di_spalloc(m, n, x.len() as i32, 1, 1);
        for (k, elem) in x.iter().enumerate() {
            cs_di_entry(T, rows[k], cols[k], *elem);
        }
        return T
    }
}

/// Converts a Sparse struct representing a matrix into a C struct for CSS Sparse matrix
/// the C struct doesn't really exist, it is a mutable pointer reference to the Sparse struct
pub fn sparse_to_cs(sparse: &COO) -> *mut cs_di {
    let T = create_sparse_matrix(
        sparse.nr,
        sparse.nc,
        &sparse.i[..],
        &sparse.j[..],
        &sparse.x[..]
    );
    convert_to_compressed(T)
}

pub fn convert_to_compressed(T: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_compress(T)
    }
}

pub fn print_matrix(A: *mut cs_di) {
    unsafe {
        cs_di_print(A, 0);
    }
}

pub fn transpose(A: *mut cs_di, nnz: i32) -> *mut cs_di {
    unsafe {
        cs_di_transpose(A, nnz)
    }
}

pub fn sp_mm_multiply_f64(A: *mut cs_di, B: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_multiply(A, B)
    }
}

pub fn sp_mv_multiply_f64(A: *mut cs_di, x: &[f64], y: &mut [f64]) -> i32 {
    unsafe {
        cs_di_gaxpy(A, x.as_ptr(), y.as_mut_ptr())
    }
}

pub fn sp_add(A: *mut cs_di, B: *mut cs_di, alpha: f64, beta: f64) -> *mut cs_di {
    // equation of the form alpha * A + beta * B
    unsafe {
        cs_di_add(A, B, alpha, beta)
    }
}

pub fn spfree(A: *mut cs_di) {
    unsafe {
        cs_di_spfree(A);
    }
}

pub fn spalloc(m: i32, n: i32, nzmax: i32, values: i32, t: i32) -> *mut cs_di {
    unsafe {
        cs_di_spalloc(m, n, nzmax, values, t)
    }
}

#[allow(non_snake_case)]
pub fn add_vecs(x: &[f64], y: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_sys::cblas_daxpy(ns, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn copy(x: &[f64], y: &mut [f64], ns: i32) {
    unsafe {
        cblas_dcopy(ns, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn dscal(x: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_dscal(ns, alpha, x.as_mut_ptr(), 1);
    }
}

fn blas_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    unsafe {
        cblas_ddot(v1.len() as i32, v1.as_ptr(), 1, v2.as_ptr(), 1)
    }
}

fn blas_matrix_vector_mulf64(matrix: &[f64], v: &[f64], m: i32, n: i32, result: &mut [f64]) {
    unsafe {
        cblas_dgemv(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            1.0,
            matrix.as_ptr(),
            m,
            v.as_ptr(),
            1,
            1.0,
            result.as_mut_ptr(),
            1
        )
    }
}

//-----------------------------------
//  Value Iteration Helpers
//
fn max_eps(x: &[f64]) -> f64 {
    *x.iter().max_by(|a, b| a.partial_cmp(&b).expect("No NaNs allowed")).unwrap()
}

#[allow(non_snake_case)]
fn update_qmat(q: &mut [f64], v: &[f64], row: usize, nr: usize) -> Result<(), String>{
    for (ii, val) in v.iter().enumerate() {
        q[ii * nr + row] = *val;
    }
    Ok(())
}

fn update_policy(eps: &[f64], thresh: &f64, pi: &mut [f64], pi_new: &[f64],
    ns: usize, policy_stable: &mut bool) {
    for ii in 0..ns {
        if eps[ii] > *thresh {
            // println!("ix: {}, eps {}, action: {:?}", ii, eps[ii], pi_new[ii]);
            // update the action in pi with pnew
            pi[ii] = pi_new[ii];
            *policy_stable = false
        }
    }
}

fn max_values(x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize) {
    for ii in 0..ns {
        let (imax, max) = q[ii*na..(ii + 1)*na]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)|
                a.partial_cmp(b).expect("no NaNs allowed!"))
            .unwrap();
        pi[ii] = imax as f64;
        x[ii] = *max;
    }
}

//-----------------------------------=
pub fn reverse_key_value_pairs<T, U>(map: &HashMap<T, U>) -> HashMap<U, T> 
where T: Clone + Hash, U: Clone + Hash + Eq {
    map.into_iter().fold(HashMap::new(), |mut acc, (a, b)| {
        acc.insert(b.clone(), a.clone());
        acc
    })
}

#[pyfunction]
#[pyo3(name="vi_test")]
fn value_iteration_test(model: &MOProductMDP) {
    //let prods = model.construct_products();
    //process_mdps(prods);
    value_iteration(model, &[0.5, 0.5], &0.001);
}

/// A Python module implemented in Rust.
#[pymodule]
fn ce(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Agent>()?;
    m.add_class::<DFA>()?;
    m.add_class::<Mission>()?;
    m.add_class::<Team>()?;
    m.add_class::<SCPM>()?;
    m.add_function(wrap_pyfunction!(build_model, m)?)?;
    m.add_function(wrap_pyfunction!(value_iteration_test, m)?)?;
    Ok(())
}