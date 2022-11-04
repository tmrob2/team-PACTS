#![allow(non_snake_case)]
#![allow(dead_code)]

pub mod scpm;
pub mod agent;
pub mod dfa;
pub mod parallel;
pub mod algorithm;
pub mod c_binding;
pub mod lp;
pub mod envs;
pub mod executor;
pub mod gpu_scpm;
pub mod sparse;

use std::hash::Hash;
use std::mem;
use std::thread::current;
use std::time::Instant;
use agent::agent::Env;
use executor::executor::{GenericSolutionFunctions, DefineSolution, 
    Solution, Execution, DefineSerialisableExecutor};
use pyo3::prelude::*;
//use pyo3::exceptions::PyValueError;
use hashbrown::HashMap;
use rand_chacha::rand_core::block;
use scpm::model::SCPM; // , MOProductMDP};
use algorithm::synth::scheduler_synthesis;
//test_scpm, warehouse_scheduler_synthesis
use envs::warehouse::{Warehouse, warehouse_scheduler_synthesis, 
    Executor, SerialisableExecutor};
use envs::{warehouse, message};
use envs::message::*;
use gpu_scpm::gpu_model::{self, GPUSCPM};
//use agent::agent::{MDP};
use dfa::dfa::{DFA, Mission, json_deserialize_from_string};
//use parallel::{threaded::process_mdps};
use c_binding::suite_sparse::*;
extern crate blis_src;
extern crate cblas_sys;
use cblas_sys::{cblas_dcopy, cblas_dgemv, cblas_dscal, cblas_ddot, cblas_sdot};
//use algorithm::dp::value_iteration;
use float_eq::float_eq;
use std::fs;
use sparse::definition::{CxxMatrixf32, CSparsef32};
use sparse::compress::compress;

use crate::gpu_scpm::gpu_dp_utils::random_policy;


const UNSTABLE_POLICY: i32 = 5;
// -----------------------------------------------------------
//  C Library FFI
// -----------------------------------------------------------
extern "C" {
    fn gather_policy(pi: *const i32, output: *mut i32, nc: i32, nr: i32, prod_block_size: i32);
}

pub fn gather_policy_ffi(pi: &[i32], output: &mut [i32], nc: i32, nr: i32, prod_block_size: i32) {
    unsafe {
        gather_policy(pi.as_ptr(), output.as_mut_ptr(), nc, nr, prod_block_size);
    }
}

extern "C" {
    fn test_csr_create(m: i32, n: i32, nnz: i32, i: *const i32, j: *const i32, x: *const f32);
}

pub fn test_csr_create_ffi(m: i32, n: i32, nnz: i32, i: &[i32], j: &[i32], x: &[f32]) {
    unsafe {
        test_csr_create(m, n, nnz,i.as_ptr(), j.as_ptr(), x.as_ptr());
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn initial_policy_value(
        pm: i32,
        pn: i32,
        pnz: i32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        x: *mut f32,
        y: *mut f32,
        w: *const f32,
        rmv: *mut f32,
        eps: f32
    );
}

// TODO this needs the library linking 
pub fn initial_policy_value_ffi(
    P: &CxxMatrixf32,
    R: &CxxMatrixf32,
    x: &mut [f32],
    y: &mut [f32],
    w: &[f32],
    rmv: &mut [f32],
    eps: f32
) {
    unsafe {
        initial_policy_value(
            P.m,
            P.n,
            P.nz,
            P.i.as_ptr(),
            P.p.as_ptr(),
            P.x.as_ptr(),
            R.m, 
            R.n, 
            R.nz,
            R.i.as_ptr(),
            R.p.as_ptr(),
            R.x.as_ptr(),
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            w.as_ptr(),
            rmv.as_mut_ptr(),
            eps
        );
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn policy_optimisation(
        init_value: *const f32,
        init_pi: *const i32,
        pm: i32, 
        pn: i32,
        pnz: i32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        x: *mut f32,
        y: *mut f32,
        rmv: *const f32,
        w: *const f32,
        eps: f32,
        block_size: i32,
        nact: i32
    );
}

pub fn policy_optimisation_ffi(
    init_value: &[f32],
    policy: &mut [i32],
    P: &CxxMatrixf32,
    R: &CxxMatrixf32,
    w: &[f32],
    x: &mut [f32],
    y: &mut [f32],
    rmv: &[f32],
    eps: f32,
    block_size: i32,
    nact: i32
) {
    unsafe {
        policy_optimisation(
            init_value.as_ptr(), 
            policy.as_mut_ptr(),
            P.m, 
            P.n, 
            P.nz, 
            P.i.as_ptr(), 
            P.p.as_ptr(), 
            P.x.as_ptr(), 
            R.m, 
            R.n, 
            R.nz, 
            R.i.as_ptr(), 
            R.p.as_ptr(), 
            R.x.as_ptr(), 
            x.as_mut_ptr(), 
            y.as_mut_ptr(), 
            rmv.as_ptr(),
            w.as_ptr(),
            eps,
            block_size,
            nact
        )
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn multi_objective_values(
        R: *const f32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        pm: i32,
        pn: i32,
        pnz: i32,
        eps: f32,
        x: *mut f32
    );
}

pub fn multi_objective_values_ffi(
    R: &[f32],
    P: &CxxMatrixf32,
    eps: f32,
    x: &mut [f32]
) {
    unsafe {
        multi_objective_values(
            R.as_ptr(), 
            P.i.as_ptr(), 
            P.p.as_ptr(), 
            P.x.as_ptr(), 
            P.m, 
            P.n, 
            P.nz, 
            eps, 
            x.as_mut_ptr()
        );
    }
}

#[link(name="cudatest", kind="static")]
extern "C" {
    fn test_csr_spmv(
        csr_row: *const i32, 
        csr_col: *const i32, 
        csr_vals: *const f32,
        x: *const f32,
        y: *mut f32,
        nnz: i32, 
        sizeof_row: i32, 
        m: i32,
        n: i32
    );
}

pub fn test_csr_spmv_ffi(
    row: &[i32],
    col: &[i32],
    vals: &[f32],
    x: &[f32],
    y: &mut [f32],
    nnz: i32,
    m: i32,
    n: i32,
) {
    unsafe {
        test_csr_spmv(
            row.as_ptr(), 
            col.as_ptr(), 
            vals.as_ptr(), 
            x.as_ptr(), 
            y.as_mut_ptr(),
            nnz, 
            row.len() as i32,
            m, 
            n
        )
    }
}

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
pub struct DenseMatrixf32{
    // matrix values in 1d format
    pub m: Vec<f32>,
    // number of rows in the matrix
    pub rows: usize,
    // number of columns in the matrix
    pub cols: usize
}

impl DenseMatrixf32 {
    pub fn new(n: usize, m: usize) -> Self {
        DenseMatrixf32 {
            m: Vec::new(),
            rows: n,
            cols: m
        }
    }
}

#[derive(Debug)]
pub struct Triple {
    pub nzmax: i32,
    pub nr: i32,
    pub nc: i32,
    pub i: Vec<i32>,
    pub j: Vec<i32>,
    pub x: Vec<f64>,
    pub nz: i32,
}

pub enum SpType {
    Triple, 
    CSR
}

impl Triple {
    pub fn new() -> Self {
        Triple {
            nzmax: 0,
            nr: 0,
            nc: 0,
            i: Vec::new(),
            j: Vec::new(),
            x: Vec::new(),
            nz: 0
        }
    }
}

#[pyclass]
pub struct GPUProblemMetaData {
    #[pyo3(get)]
    pub max_size: usize,
    #[pyo3(get)]
    pub transition_prod_block_size: usize,
    #[pyo3(get)]
    pub reward_obj_prod_block_size: usize,
    pub init_state_idx: Vec<usize>
}

#[derive(Debug)]
pub struct SparseMatrixComponents {
    pub i: Vec<i32>, // row indices per column
    pub p: Vec<i32>, // column ranges
    pub x: Vec<f64>  // values per column row indices
}

pub fn deconstruct(A: *mut cs_di, nnz: usize, cols: usize) -> SparseMatrixComponents {
    let x: Vec<f64>;
    let p: Vec<i32>;
    let i: Vec<i32>;
    unsafe {
        x = Vec::from_raw_parts((*A).x as *mut f64, nnz, nnz);
        i = Vec::from_raw_parts((*A).i as *mut i32, nnz, nnz);
        p = Vec::from_raw_parts((*A).p as *mut i32, cols + 1, cols + 1);
    }
    SparseMatrixComponents {i, p, x}
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
pub fn sparse_to_cs(sparse: &Triple) -> *mut cs_di {
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

fn blas_dot_productf32(v1: &[f32], v2: &[f32]) -> f32 {
    unsafe {
        cblas_sdot(v1.len() as i32, v1.as_ptr(), 1, v2.as_ptr(), 1)
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

// -----------------------------------------------------------
//  Value iteration helpers
// -----------------------------------------------------------

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
            // update the action in pi with pnew
            pi[ii] = pi_new[ii];
            *policy_stable = false
        }
    }
}

fn max_values(x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize) {
    for ii in 0..ns {
        //println!("q: {:?}", &q[ii*na..(ii + 1)*na]);
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

//-------------------------------------
pub fn reverse_key_value_pairs<T, U>(map: &HashMap<T, U>) -> HashMap<U, T> 
where T: Clone + Hash, U: Clone + Hash + Eq {
    map.into_iter().fold(HashMap::new(), |mut acc, (a, b)| {
        acc.insert(b.clone(), a.clone());
        acc
    })
}

#[derive(Hash, Eq, PartialEq)]
pub struct Mantissa((u64, i16, i8));

impl Mantissa {
    pub fn new(val: f64) -> Mantissa {
        Mantissa(integer_decode(val))
    }
}

pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52 ) & 0x7ff ) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff ) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

/// This method will adjust any values close to zero as zeroes, correcting LP rounding errors
pub fn val_or_zero_one(val: &f64) -> f64 {
    if float_eq!(*val, 0., abs <= 0.25 * f64::EPSILON) {
        0.
    } else if float_eq!(*val, 1., abs <= 0.25 * f64::EPSILON) {
        1.
    } else {
        *val
    }
}

pub fn double_vec(v: Vec<i32>) -> Vec<i32> {
    [&v[..], &v[..]].concat()
}
//--------------------------------------
// Python mdp env wrapper 
//--------------------------------------
fn get_actions(fpath: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let solver_script_call: String = fs::read_to_string(fpath)?.parse()?;
    let result: Vec<f64> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
        let lpsolver = PyModule::from_code(py, &solver_script_call, "", "")?;
        //let gym = PyModule::import(py, "gym")?;
        let action_space_size = lpsolver.getattr("action_space.n")?.call0()?.extract()?;
        Ok(action_space_size)
    }).unwrap();
    Ok(result)
}
//--------------------------------------
// Python lp wrappers, for linear programming scripts
//--------------------------------------
fn solver(hullset: Vec<Vec<f64>>, t: Vec<f64>, nobjs: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let solver_script_call = include_str!("lp/pylp.py");
    let result: Vec<f64> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
        let lpsolver = PyModule::from_code(py, solver_script_call, "", "")?;
        let solver_result = lpsolver.getattr("hyperplane_solver")?.call1(
            (hullset, t, nobjs,)
        )?.extract()?;
        Ok(solver_result)
    }).unwrap();
    Ok(result)
}

fn solverf32(hullset: Vec<Vec<f32>>, t: Vec<f32>, nobjs: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let solver_script_call = include_str!("lp/pylp.py");
    let result: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let lpsolver = PyModule::from_code(py, solver_script_call, "", "")?;
        let solver_result = lpsolver.getattr("hyperplane_solver")?.call1(
            (hullset, t, nobjs,)
        )?.extract()?;
        Ok(solver_result)
    }).unwrap();
    Ok(result)
}

fn random_sched(
    alloc: Vec<(i32, i32, i32, Vec<f64>)>, 
    t: Vec<f64>, 
    l: usize, 
    m: usize, 
    n: usize
) -> Option<Vec<f64>> {
    let script_call = include_str!("lp/pylp.py");
    let result: Result<Vec<f64>, PyErr> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
        let lpsolver = PyModule::from_code(py, script_call, "", "")?;
        let solver_result = lpsolver.getattr("randomised_scheduler")?.call1((
            alloc,
            t,
            l, 
            m, 
            n
        ))?.extract()?;
        Ok(solver_result)
    });
    match result {
        Ok(r) => { return Some(r) }
        Err(e) => { println!("Err: {:?}", e); return None }
    }
}

fn new_target(
    hullset: Vec<Vec<f64>>, 
    weights: Vec<Vec<f64>>, 
    target: Vec<f64>,
    l: usize,
    //m: usize,
    n: usize,
    //iteration: usize,
    //cstep: f64,
    //pstep: f64
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let new_target_script = include_str!("lp/pylp.py");
    let result: Vec<f64> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
        let lpnewtarget = PyModule::from_code(py, new_target_script, "", "")?;
        let lpnewtarget_result = lpnewtarget.getattr("eucl_new_target")?.call1((
            hullset,
            weights,
            target,
            l,
            //m,
            n,
            //iteration,
            //cstep,
            //pstep
        ))?.extract()?;
        Ok(lpnewtarget_result)
    }).unwrap();
    Ok(result)
}

fn new_targetf32(
    hullset: Vec<Vec<f32>>, 
    weights: Vec<Vec<f32>>, 
    target: Vec<f32>,
    l: usize,
    //m: usize,
    n: usize,
    //iteration: usize,
    //cstep: f32,
    //pstep: f32
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let new_target_script = include_str!("lp/pylp.py");
    let result: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let lpnewtarget = PyModule::from_code(py, new_target_script, "", "")?;
        let lpnewtarget_result = lpnewtarget.getattr("eucl_new_target")?.call1((
            hullset,
            weights,
            target,
            l,
            //m,
            n,
            //iteration,
            //cstep,
            //pstep
        ))?.extract()?;
        Ok(lpnewtarget_result)
    }).unwrap();
    Ok(result)
}

//--------------------------------------
// Some testing functions for python testing of Rust API
//--------------------------------------

fn generic_scheduler_synthesis<E, S>(
    model: &mut SCPM,
    env: &mut E, 
    w: Vec<f64>, 
    eps: f64, 
    target: Vec<f64>,
    executor: &mut SerialisableExecutor
) -> Result<Vec<f64>, String>
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
    E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S>,
    SerialisableExecutor: DefineSerialisableExecutor<S> + Execution<S> {
    println!("Constructing products");
    let mut solution: Solution<S> = Solution::new_(model.tasks.size);
    let prods = model.construct_products(env);

    // todo input the "outputs" into the scheduler synthesis algorithm to capture data
    let (
        pis, 
        alloc, 
        t_new, 
        l,
        prods
    ) = scheduler_synthesis(model, &w[..], &eps, &target[..], prods);
    println!("alloc: \n{:.3?}", alloc);
    // convert output schedulers to 
    // we need to construct the randomised scheduler here, then the output from the randomised
    // scheduler, which will already be from a python script, will be the output of this function
    let weights = random_sched(
        alloc, t_new.to_vec(), l, model.tasks.size, model.num_agents
    );
    // Assign the ownership of the product models into the outputs at this point
    for prod in prods.into_iter() {
            solution.set_prod_state_maps(prod.state_map, prod.agent_id, prod.task_id);
    }
    match weights {
        Some(w) => {
            solution.set_schedulers(pis);
            solution.set_weights(w, l);
            solution.add_to_agent_task_queues();

            executor.add_alloc_to_execution(&mut solution, model.num_agents);
            for t in (0..model.tasks.size).rev() {
                match model.tasks.release_last_dfa() {
                    Some(task_) => { executor.insert_dfa(task_, t as i32); }
                    None => { }    
                };
            }

            return Ok(t_new) 
        }
        None => { 
            return Err(format!(
                "Randomised scheduler weights could not be found for target vector: {:?}", 
                t_new
            ))
        }
    }
}

fn generic_scheduler_synthesis_without_execution<E, S>(
    model: &mut SCPM,
    env: &mut E, 
    w: Vec<f64>, 
    eps: f64, 
    target: Vec<f64>,
) -> Result<Vec<f64>, String>
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
    E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S> {
    println!("Constructing products");
    let mut solution: Solution<S> = Solution::new_(model.tasks.size);
    let prods = model.construct_products(env);
    let t1 = Instant::now();
    // todo input the "outputs" into the scheduler synthesis algorithm to capture data
    let (
        pis, 
        alloc, 
        t_new, 
        l,
        prods
    ) = scheduler_synthesis(model, &w[..], &eps, &target[..], prods);
    println!("alloc: \n{:.3?}", alloc);
    // convert output schedulers to 
    // we need to construct the randomised scheduler here, then the output from the randomised
    // scheduler, which will already be from a python script, will be the output of this function
    let weights = random_sched(
        alloc, t_new.to_vec(), l, model.tasks.size, model.num_agents
    );
    println!("Time: {:?}", t1.elapsed().as_secs_f32());
    // Assign the ownership of the product models into the outputs at this point
    for prod in prods.into_iter() {
            solution.set_prod_state_maps(prod.state_map, prod.agent_id, prod.task_id);
    }
    match weights {
        Some(w) => {
            solution.set_schedulers(pis);
            solution.set_weights(w, l);
            solution.add_to_agent_task_queues();
            return Ok(t_new) 
        }
        None => { 
            return Err(format!(
                "Randomised scheduler weights could not be found for target vector: {:?}", 
                t_new
            ))
        }
    }
}

// This is a function for constructing the data structures which will go into
// the GPU for computation
fn construct_gpu_problem<S, E>
(
    model: &mut gpu_model::GPUSCPM,
    env: &mut E,
) -> (CxxMatrixf32, Vec<i32>, CxxMatrixf32, GPUProblemMetaData)
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
    E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S>  {
    let mut prods = model.construct_products(env);

    // determine the size of the sparse matrix
    // for each of the product model transition matrices which is the largest
    let mut max_size: i32 = 0;
    let mut nz = 0;
    let mut rmax_nz: i32 = 0;
    let mut matrix_ranges: Vec<i32> = Vec::new();
    let mut initial_policy: Vec<i32> = Vec::new();
    let nobjs: usize = model.num_agents + model.tasks.size;
    let mut init_state_idx: Vec<usize> = Vec::new();

    let mut current_states_size = 0;
    
    for prod in prods.iter() {
        for action in env.get_action_space().iter() {
            let mat = prod.transition_mat.get(action).unwrap();
            nz += mat.nz;
            max_size = std::cmp::max(max_size, mat.n);
            matrix_ranges.push(mat.m);
            let r = prod.rewards_mat.get(&action).unwrap();
            rmax_nz += (r.rows * r.cols) as i32;
        }   
        let mut pi: Vec<i32> = random_policy(prod).iter().map(|x| *x as i32).collect();
        initial_policy.append(&mut pi);
        init_state_idx.push(
            *prod.get_state_map().get(&prod.initial_state).unwrap() + 
            current_states_size
        );
        current_states_size += prod.states.len();
    }

    println!("maximum product state space: {}", max_size);
    println!("number of non zero entries: {}", nz);

    // what is the ordering of the product matrices, because this is also important
    // what is the size of the big matrix going to be?
    // instantiate a new COO matrix
    let prod_block: i32 = model.state_spaces
        .iter()
        .map(|(_, v)| *v)
        .sum();
    let matrix_size: i32 = (prod_block * env.get_action_space().len() as i32).pow(2);
    println!("size of GPU matrix: {} x {} = {}, sparsity %: {:.2}", 
        prod_block * env.get_action_space().len() as i32, 
        prod_block * env.get_action_space().len() as i32, 
        matrix_size,
        (nz as f32 / matrix_size as f32) * 100.
    );

    let objs_prod_block = (nobjs * prods.len()) as i32;
    println!("matrix stride: {}", prod_block);
    let mut mat = CxxMatrixf32::make(
        nz, 
        prod_block * env.get_action_space().len() as i32, 
        prod_block * env.get_action_space().len() as i32, 
        0
    );
    let mut r = CxxMatrixf32::make(
        rmax_nz,
        prod_block * env.get_action_space().len() as i32,
        objs_prod_block * env.get_action_space().len() as i32,
        0
    );
    let mut mat_coo: Vec<(i32, i32, f32)> = Vec::new();
    for action in env.get_action_space().iter() {
        for (z, prod) in prods.iter_mut().enumerate() {
            let prod_triple = prod.transition_mat.remove(action).unwrap();
            // sum the state spaces up to z
            let block_size: i32 = model.state_spaces
                .iter()
                .filter(|(k, _)| (**k as usize) < z)
                .map(|(_, v)| *v)
                .sum();
            
                println!("agent {}, task: {}, base_idx: {}", 
                prod.agent_id, prod.task_id, block_size + action * prod_block);

            // we need to consume this matrix at this point so as not to double up on memory
            // insert the triple into the large coo mat
            for k in 0..prod_triple.nz {
                mat_coo.push((
                    prod_triple.i[k as usize] + block_size + action * prod_block,
                    prod_triple.p[k as usize] + block_size + action * prod_block,
                    prod_triple.x[k as usize]
                ));
            }
            // free rprod to be consumed by the larger matrix
            let rprod = prod.rewards_mat.remove(action).unwrap();
            
            for row_ in 0..rprod.rows as usize {
                for c in 0..nobjs {
                    let rval = rprod.m[c * rprod.rows + row_];
                    r.triple_entry(
                        row_ as i32 + block_size + action * prod_block, 
                        c as i32 + (z as i32) * nobjs as i32 + action * objs_prod_block, 
                        rval
                    )
                }
            }
        }
    }
    // The triple entries need to be sorted before they can be entered into
    mat_coo.sort_by(
        |(a1, a2, _), (b1, b2, _)| 
        (a1, a2).cmp(&(b1, b2))
    );

    for (i, j, val) in mat_coo.drain(..) {
        mat.triple_entry(i, j, val);
    }

    // Store the problem metadata
    let meta = GPUProblemMetaData {
        max_size: max_size as usize,
        transition_prod_block_size: prod_block as usize,
        reward_obj_prod_block_size: objs_prod_block as usize,
        init_state_idx
    };

    (mat, initial_policy, r, meta)
}

/// A Python module implemented in Rust.
#[pymodule]
fn ce(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    //m.add_class::<MDP>()?;
    m.add_class::<DFA>()?;
    m.add_class::<Mission>()?;
    //m.add_class::<PySolution>()?;
    //m.add_class::<Team>()?;
    m.add_class::<SCPM>()?;
    m.add_class::<GPUSCPM>()?;
    m.add_class::<Warehouse>()?;
    m.add_class::<MessageSender>()?;
    m.add_class::<Executor>()?;
    m.add_class::<SerialisableExecutor>()?;
    m.add_class::<CxxMatrixf32>()?;
    m.add_class::<GPUProblemMetaData>()?;
    //m.add_function(wrap_pyfunction!(build_model, m)?)?;
    //m.add_function(wrap_pyfunction!(value_iteration_test, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_prod, m)?)?;
    m.add_function(wrap_pyfunction!(warehouse::test_prod, m)?)?;
    m.add_function(wrap_pyfunction!(warehouse_scheduler_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(message::msg_scheduler_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(json_deserialize_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_gpu_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_compress, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_initial_policy, m)?)?;
    m.add_function(wrap_pyfunction!(message::gpu_test_csr_mv, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_cpu_init_pi, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_argmax_csr, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_cpu_converged_init_pi, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_output_trans_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_output_rewards_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_policy_optimisation,m)?)?; 
    m.add_function(wrap_pyfunction!(message::test_nobj_argmax_csr, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_gpu_value_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(message::test_gpu_synth, m)?)?;
    Ok(())
}