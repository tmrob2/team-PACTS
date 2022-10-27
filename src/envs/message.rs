use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use hashbrown::HashMap;
use crate::agent::agent::Env;
use crate::sparse::argmax::argmaxM;
use crate::{generic_scheduler_synthesis_without_execution, test_csr_create_ffi, GPUProblemMetaData, deconstruct};
use crate::sparse::definition::CxxMatrixf32;
use crate::sparse::{compress, argmax};
use crate::test_initial_policy_value_ffi;
use crate::gpu_scpm::gpu_model;
use crate::construct_gpu_problem;
use crate::scpm::model::{SCPM, build_model};
use crate::algorithm::dp::value_iteration;
use crate::test_csr_spmv_ffi;
use crate::SparseMatrix;
use std::time::Instant;

type State = i32;

#[pyclass]
// A message sender is a single agent
pub struct MessageSender {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub action_space: Vec<i32>
}

#[pymethods]
impl MessageSender {
    #[new]
    fn new() -> Self {
        MessageSender {
            states: (0..5).collect(),
            initial_state: 0,
            action_space: (0..2).collect()
        }
    }
}

impl Env<State> for MessageSender {
    fn step_(&self, s: State, action: u8) -> Result<Vec<(State, f64, String)>, String> {
        let transition: Result<Vec<(State, f64, String)>, String> = match s {
            0 => {
                // return the transition for state 0
                match action {
                    0 => {Ok(vec![(0, 0.01, "".to_string()), (1, 0.99, "i".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            1 => {
                // return the transition for state 1
                match action {
                    0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            2 => { 
                // return the transition for state 2
                match action {
                    0 => {Ok(vec![(3, 0.99, "s".to_string()), (4, 0.01, "e".to_string())])}
                    1 => {Ok(vec![(4, 1.0, "e".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            3 => {
                match action {
                    0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            4 => {
                match action {
                    0 => {Ok(vec![(0, 1.0, "".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            _ => {
                // Not implemented error
                Err("Not-implemented".to_string())
            }

        };
        transition
    }

    fn get_init_state(&self, _agent: usize) -> State {
        0
    }

    fn set_task(&mut self, _task_id: usize) {
    }

    fn get_action_space(&self) -> Vec<i32> {
        self.action_space.to_vec()
    }
}

#[pyfunction]
pub fn test_cpu_init_pi(
    model: &SCPM,
    env: &mut MessageSender,
    test_policy: Vec<f64>,
) -> Vec<f64> 
where MessageSender: Env<State> {
    let pmdp = build_model(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    let size = pmdp.states.len();
    //let nobjs: usize = model.num_agents + model.tasks.size;
    let x = vec![0f64; size]; // value vector for agent-task pair
    let mut P: HashMap<i32, SparseMatrix> = HashMap::new();
    for action in pmdp.actions.iter() {
        let TP = pmdp.transition_mat.get(action).unwrap();
        let A = crate::sparse_to_cs(TP);
        P.insert(*action, SparseMatrix {
            m: A,
            nr: TP.nr as usize,
            nc: TP.nc as usize,
            nnz: TP.nz as usize
        });
    }
    // Determine the value of the initial vector
    let argmaxP = crate::algorithm::dp::construct_argmax_spmatrix(
        &pmdp, &test_policy[..], &P, size
    );
    let argmaxR = crate::algorithm::dp::construct_argmax_Rvector(
        &pmdp, &test_policy[..]);
    let mut vmv = vec![0f64; argmaxP.nr];
    crate::sp_mv_multiply_f64(argmaxP.m, &x[..], &mut vmv[..]);
    crate::add_vecs(&argmaxR[..], &mut vmv[..], argmaxP.nr as i32, 1.0);
    //crate::algorithm::dp::value_for_init_policy(&argmaxR[..], &mut x[..], &eps, &argmaxP);
    vmv
}


#[pyfunction]
pub fn test_cpu_converged_init_pi(
    model: &SCPM,
    env: &mut MessageSender,
    test_policy: Vec<f64>,
    eps: f64,
    agent: i32,
    task: i32
) -> Vec<f64> 
where MessageSender: Env<State> {
    let pmdp = build_model(
        (env.get_init_state(agent as usize), 0), 
        env, 
        &model.tasks.get_task(task as usize), 
        agent, 
        task, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    let size = pmdp.states.len();
    //let nobjs: usize = model.num_agents + model.tasks.size;
    let mut x = vec![0f64; size]; // value vector for agent-task pair
    let mut P: HashMap<i32, SparseMatrix> = HashMap::new();
    for action in pmdp.actions.iter() {
        let TP = pmdp.transition_mat.get(action).unwrap();
        let A = crate::sparse_to_cs(TP);
        P.insert(*action, SparseMatrix {
            m: A,
            nr: TP.nr as usize,
            nc: TP.nc as usize,
            nnz: TP.nz as usize
        });
    }
    // Determine the value of the initial vector
    let argmaxP = crate::algorithm::dp::construct_argmax_spmatrix(
        &pmdp, &test_policy[..], &P, size
    );
    let argmaxR = crate::algorithm::dp::construct_argmax_Rvector(
        &pmdp, &test_policy[..]);
    crate::algorithm::dp::value_for_init_policy(&argmaxR, &mut x, &eps, &argmaxP);
    x
}

#[pyfunction]
pub fn test_output_trans_matrix(
    model: &SCPM,
    env: &mut MessageSender,
    test_policy: Vec<f64>,
) -> (Vec<i32>, Vec<i32>, Vec<f64>, usize, usize, usize)
where MessageSender: Env<State> {
    let pmdp = build_model(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    let size = pmdp.states.len();
    //let nobjs: usize = model.num_agents + model.tasks.size;
    let mut x = vec![0f64; size]; // value vector for agent-task pair
    let mut P: HashMap<i32, SparseMatrix> = HashMap::new();
    for action in pmdp.actions.iter() {
        let TP = pmdp.transition_mat.get(action).unwrap();
        let A = crate::sparse_to_cs(TP);
        P.insert(*action, SparseMatrix {
            m: A,
            nr: TP.nr as usize,
            nc: TP.nc as usize,
            nnz: TP.nz as usize
        });
    }
    // Determine the value of the initial vector
    let argmaxP = crate::algorithm::dp::construct_argmax_spmatrix(
        &pmdp, &test_policy[..], &P, size
    );
    let output = deconstruct(argmaxP.m, argmaxP.nnz, argmaxP.nc);
    (output.i, output.p, output.x, argmaxP.nnz, argmaxP.nc, argmaxP.nr)
}

#[pyfunction]
pub fn test_output_rewards_matrix(
    model: &SCPM,
    env: &mut MessageSender,
    test_policy: Vec<f64>,
    nobjs: i32
) -> (Vec<f64>, usize, usize)
where MessageSender: Env<State> {
    let pmdp = build_model(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    //let nobjs: usize = model.num_agents + model.tasks.size;
    let argmaxR = crate::algorithm::dp::construct_argmax_Rmatrix(
        &pmdp, &test_policy[..], nobjs as usize);
    (argmaxR.m, argmaxR.cols, argmaxR.rows)
}

#[pyfunction]
pub fn test_prod(
    model: &SCPM,
    env: &mut MessageSender,
    w: Vec<f64>,
    eps: f64
) -> Vec<f64>
where MessageSender: Env<State> {
    let t1 = Instant::now();

    let pmdp = build_model(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    println!("ProdMDP |S|: {}", pmdp.states.len());
    println!("ProdMDP |P|: {}", pmdp.transition_mat.get(&0).unwrap().nr);
    println!("Starting value iteration");
    let (pi, r) = value_iteration(
        &pmdp, 
        &w[..], 
        &eps, 
        model.num_agents, 
        model.tasks.size
    );
    println!("r: {:?}", r);
    println!("{:?}", t1.elapsed());
    let mut pmap: HashMap<(i32, i32), HashMap<(State, i32), usize>> = HashMap::new();
    pmap.insert((0, 0), pmdp.state_map);
    //outputs.prod_state_map = pmap;
    pi
}

#[pyfunction]
#[pyo3(name="msg_experiment")]
pub fn msg_scheduler_synthesis(
    model: &mut SCPM,
    env: &mut MessageSender,
    w: Vec<f64>,
    target: Vec<f64>,
    eps: f64
) -> PyResult<Vec<f64>> {
    let result = 
        generic_scheduler_synthesis_without_execution(model, env, w, eps, target);
    match result {
        Ok(r) => { Ok(r) }
        Err(e) => Err(PyValueError::new_err(e))
    }
}

#[pyfunction]
pub fn test_gpu_matrix(
    model: &mut gpu_model::GPUSCPM,
    env: &mut MessageSender
) -> (CxxMatrixf32, Vec<i32>, CxxMatrixf32, GPUProblemMetaData) {
    let (mat, pi, r, data) = 
        construct_gpu_problem(model, env);

    // generate a random policy

    (mat, pi, r, data)
}

#[pyfunction]
pub fn test_initial_policy(
    pi: Vec<i32>,
    P: &CxxMatrixf32,
    R: &CxxMatrixf32,
    row_block_size: i32,
    trans_col_block_size: i32,
    reward_col_block_size: i32,
    w: Vec<f32>,
    mut rmv: Vec<f32>
) {
    // output mutable value vector
    let mut value: Vec<f32> = vec![0.; row_block_size as usize];
    
    //println!("P mat nz: {}", P.nz);
    //println!("R mat nz: {}", R.nz);
    println!("Policy: {:?}", pi);

    // with the initial policy create the argmax P matrix and R matrix
    //let row_idx_from_policy: 
    
    let argmaxP = argmax::argmaxM(
        P, 
        &pi, 
        row_block_size, 
        trans_col_block_size
    );
    

    let argmaxR = argmax::argmaxM(
        R, 
        &pi, 
        row_block_size, 
        reward_col_block_size
    );

    let mut x = vec![0.; row_block_size as usize];
    let mut y = vec![0.; row_block_size as usize];
    println!("rx: \n{:?}", argmaxR.x);
    println!("rnz: {}", argmaxR.nz);
    println!("rcols: {}", argmaxR.n);
    println!("rrows: {}", argmaxR.m);
    println!("|rmv| {}", rmv.len());
    test_initial_policy_value_ffi(
        &mut value,
        &argmaxP,
        &argmaxR,
        &mut x,
        &mut y,
        &w,
        &mut rmv
    );

    println!("y:\n{:?}", y);
    println!("rmv:\n{:?}", rmv);

}

#[pyfunction]
pub fn test_compress(
    m: CxxMatrixf32
) -> CxxMatrixf32 {
    // create a new COO matrix from the input data
    // compress Coo -> Csr and then return the data to python
    let m = compress::compress(m);

    println!("TEST => load CSR data to GPU");
    test_csr_create_ffi(m.m, m.n, m.nz, &m.i, &m.p, &m.x);

    m
}

#[pyfunction]
pub fn gpu_test_csr_mv() {
    // Input a matrix in COO format; convert it to CSR and then 
    // test matrix vector multiplication to see if the result is 
    // as expected
    //let row: Vec<i32> = vec![0, 0, 0, 1, 2, 2, 2, 3, 3];
    //let col: Vec<i32> = vec![0, 2, 3, 1, 0, 2, 3, 1, 3]; 
    //let val: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
    let row: Vec<i32> = vec![0, 1, 2, 3];
    let col: Vec<i32> = vec![0, 0, 0, 1];
    let val: Vec<f32> = vec![-1., -1., -1., 1.];
    println!("TEST: COO Creation => Performing COO entries.");

    /*let mut coo = CxxMatrixf32::make(
        val.len() as i32, 4, 4, 0
    );*/
    let mut coo = CxxMatrixf32::make(4, 14, 2, 0);

    for k in 0..val.len() {
        coo.triple_entry(row[k], col[k], val[k]);
    }

    println!("TEST: CSR Creation => Consuming COO matrix into CSR");
    // consume the COO matrix into a CSR matrix
    let csr = compress::compress(coo);
    
    //let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let x: Vec<f32> = vec![1.0, 0.];
    //let mut y: Vec<f32> = vec![0., 0., 0., 0.];
    let mut y: Vec<f32> = vec![0.; 14];

    println!("TEST: cuSparse => Download data and perform matrix multiplication");

    test_csr_spmv_ffi(
        &csr.i, 
        &csr.p, 
        &csr.x, 
        &x, 
        &mut y,
        csr.nz, 
        csr.m, 
        csr.n
    );

    //println!("TEST: result => {:?}\n\tExpected: {:?}", y, vec![19.0, 8.0, 51.0, 52.0]);
    println!("TEST: result => {:?}\n\tExpected: {:?}", y, vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    //assert_eq!(y, vec![19.0, 8.0, 51.0, 52.0]);
    assert_eq!(y, vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    println!("TEST: result => SUCEEDED");
}

#[pyfunction]
pub fn test_argmax_csr(
    mat: CxxMatrixf32,
    pi: Vec<i32>,
    row_block: i32,
    col_block: i32
) -> CxxMatrixf32 {
    argmax::argmaxM(&mat, &pi, row_block, col_block)
}

