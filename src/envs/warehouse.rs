use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use crate::agent::agent::{MDP, Env};
use crate::scpm::model::{SCPM, build_model, MOProductMDP};
use crate::algorithm::synth::{process_scpm, scheduler_synthesis};
use crate::dfa::dfa::{DFA, Expression};
use crate::algorithm::dp::value_iteration;
use crate::{random_sched};
use serde::{Serialize, Deserialize};
use serde_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;
//use rand::seq::SliceRandom;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

//-------
// Types 
//-------
type Point = (i8, i8);
type State = (Point, u8, Option<Point>);

//--------
// Helpers
//--------
#[pyclass]
#[derive(Clone)]
pub struct MDPOutputs {
    pub state_map: HashMap<State, i32>,
    pub rev_state_map: HashMap<i32, State>,
    pub prod_state_map: HashMap<(i32, i32), HashMap<(i32, i32), usize>>
}

#[pymethods]
impl MDPOutputs {
    #[new]
    pub fn new() -> MDPOutputs {
        MDPOutputs {
            state_map: HashMap::new(),
            rev_state_map: HashMap::new(),
            prod_state_map: HashMap::new(),
        }
    }

    pub fn get_index(&self, state: State, q: i32, agent: i32, task: i32) -> usize {
        let mdp_idx: i32 = *self.state_map.get(&state).unwrap();
        let map = &self.prod_state_map.get(&(agent, task)).unwrap();
        *map.get(&(mdp_idx, q)).unwrap()
    }
}

impl MDPOutputs {
    pub fn store_maps(&mut self, state_map: HashMap<State, i32>, rev_state_map: HashMap<i32, State>) {
        self.state_map = state_map;
        self.rev_state_map = rev_state_map;
    }
}

//-----------------
// Python Interface
//-----------------
#[pyclass]
struct Warehouse {
    current_task: Option<usize>,
    size: usize,
    nagents: usize,
    feedpoints: Vec<Point>,
    racks: HashSet<Point>,
    agent_initial_locs: Vec<Point>
}

#[pymethods]
impl Warehouse {
    #[new]
    fn new(
        size: usize, 
        nagents: usize, 
        feedpoints: Vec<Point>, 
        initial_locations: Vec<Point>
    ) -> Self {
        // call the rack function
        let new_env = Warehouse {
            current_task: None,
            size,
            nagents,
            feedpoints,
            racks: HashSet::new(),
            agent_initial_locs: initial_locations
        };
        new_env.place_racks(size);
        new_env
    }
    
    fn get_racks(&self) -> HashSet<Point> {
        self.racks
    }

    fn step(&self, s: i32, action: u8) -> PyResult<Vec<(i32, f64, String)>> {
        let mut v: Vec<(i32, f64, String)> = Vec::new();

        Ok(v)
    }
}

impl Warehouse {
    fn place_racks(&self, size: usize) 
        -> HashSet<Point> {
        let mut racks: HashSet<Point> = HashSet::new();
        let mut count: usize = 0;
        assert!(size > 5);
        assert!(size > 4);
        for c in 2..size - 2 {
            if count < 2 {
                for r in 1..size - 2 { 
                    racks.insert((c as i8, r as i8));
                }
                count += 1;
            } else {
                count = 0;
            }
        }
        racks
    }

    fn set_state_space<'a> (
        mdp: &mut MDP<State, Data<'a>>
    )
    where MDP<State, Data<'a>>: Env<State> {
        let mut stack: Vec<i32> = Vec::new();
        // set a counter for the number of states involved in the system
        let mut visited: HashSet<i32> = HashSet::new();
        stack.push(mdp.init_state);
        //println!("{:?}", mdp.state_map);
        
        while !stack.is_empty() {
            let new_state: i32 = stack.pop().unwrap();
    
            for action in 0..mdp.actions.len() {
                let v = mdp.step(
                    new_state, action as u8
                ).unwrap();
                for (sprime, _, _) in v.iter() {
                    //println!("s: {}, a: {} -> s': {}", new_state, action, sprime);
    
                    if !visited.contains(sprime) {
                        visited.insert(*sprime);
                        stack.push(*sprime);
                    }
                }
            }
        }
    }
}
