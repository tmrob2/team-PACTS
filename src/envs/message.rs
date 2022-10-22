use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use hashbrown::HashMap;
use crate::agent::agent::Env;
use crate::generic_scheduler_synthesis_without_execution;
use crate::scpm::model::{SCPM, build_model};
use crate::algorithm::dp::value_iteration;
use std::time::Instant;

type State = i32;

#[pyclass]
// A message sender is a single agent
pub struct MessageSender {
    pub states: Vec<i32>,
    pub initial_state: i32,
}

#[pymethods]
impl MessageSender {
    #[new]
    fn new() -> Self {
        MessageSender {
            states: (0..5).collect(),
            initial_state: 0
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
