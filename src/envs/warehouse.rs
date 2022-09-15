use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use crate::agent::agent::Env;
use crate::scpm::model::{SCPM, build_model};
//use crate::algorithm::synth::{process_scpm, scheduler_synthesis};
//use crate::dfa::dfa::{DFA, Expression};
use crate::algorithm::dp::value_iteration;
use crate::{generic_scheduler_synthesis, OutputData};
//use crate::{random_sched};
//use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::time::Instant;
//use rand::seq::SliceRandom;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use array_macro::array;

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
    pub prod_state_map: HashMap<(i32, i32), HashMap<(State, i32), usize>>,
    pub schedulers: HashMap<(i32, i32), Vec<f64>>
}

#[pymethods]
impl MDPOutputs {
    pub fn get_index(&self, state: State, q: i32, agent: i32, task: i32) -> usize {
        self.get_index_(state, q, agent, task)
    }

    pub fn get_action(&self, state: usize, agent: i32, task: i32) -> i32 {
        self.get_action_(state, agent, task)
    }
}

impl OutputData<State> for MDPOutputs {
    fn get_index_(&self, state: State, q: i32, agent: i32, task: i32) -> usize {
        let map = 
            &self.prod_state_map.get(&(agent, task)).unwrap();
        *map.get(&(state, q)).unwrap()
    }

    fn get_action_(&self, state: usize, agent: i32, task: i32) -> i32 {
        self.schedulers.get(&(agent, task)).unwrap()[state] as i32
    }

    fn new_() -> Self {
        MDPOutputs { 
            prod_state_map: HashMap::new(),
            schedulers: HashMap::new()
        }
    }
}

//-----------------
// Python Interface
//-----------------
#[pyclass]
pub struct Warehouse {
    pub current_task: Option<usize>,
    pub size: usize,
    #[pyo3(get)]
    pub nagents: usize,
    #[pyo3(get)]
    pub feedpoints: Vec<Point>,
    #[pyo3(get)]
    pub racks: HashSet<Point>,
    #[pyo3(get)]
    pub agent_initial_locs: Vec<Point>,
    pub action_to_dir: HashMap<u8, [i8; 2]>,
    #[pyo3(get)]
    pub task_racks: Vec<Point>,
    #[pyo3(get)]
    pub task_feeds: Vec<Point>,
    #[pyo3(get)]
    pub words: [String; 12],
    pub seed: u64
}

#[pymethods]
impl Warehouse {
    #[new]
    fn new(
        size: usize, 
        nagents: usize, 
        feedpoints: Vec<Point>, 
        initial_locations: Vec<Point>,
        actions_to_dir: Vec<[i8; 2]>,
        seed: u64
    ) -> Self {
        // call the rack function
        let mut action_map: HashMap<u8, [i8; 2]> = HashMap::new();
        for (i, act) in actions_to_dir.into_iter().enumerate() {
            action_map.insert(i as u8, act);
        }
        let w0: [String; 3] = ["R".to_string(), "F".to_string(), "NFR".to_string()];
        let w1: [String; 4] = ["P".to_string(), "D".to_string(), "C".to_string(), "NC".to_string()];
        let mut words: [String; 12] = array!["".to_string(); 12];
        let mut count: usize = 0;
        for wa in w0.iter() {
            for wb in w1.iter() {
                words[count] = format!("{}_{}", wa, wb);
                count += 1;
            }
        }
        //let actions: [u8; 7] = array![i => i as u8; 7];
        let mut new_env = Warehouse {
            current_task: None,
            size,
            nagents,
            feedpoints,
            racks: HashSet::new(),
            agent_initial_locs: initial_locations,
            action_to_dir: action_map,
            task_racks: Vec::new(),
            task_feeds: Vec::new(),
            words,
            seed
        };
        // TODO finish testing of different minimal warehouse layouts.
        new_env.racks = new_env.place_racks(size);
        // test: => new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 2));
        new_env
    }
    
    fn get_racks(&self) -> HashSet<Point> {
        self.racks.clone()
    }

    /// Get a random group of racks from the set of all racks
    fn set_random_task_rack(&mut self, ntasks: usize) {
        let mut rnd = ChaCha8Rng::seed_from_u64(self.seed);
        let racksv: Vec<&Point> = self.racks.iter().collect();
        println!("Racks: {:?}", self.racks);
        self.task_racks = racksv.choose_multiple(&mut rnd, ntasks).map(|x| **x).collect();
        println!("SET: => task racks: {:?}", self.task_racks);
    }

    fn set_random_task_feeds(&mut self, ntasks: usize) {
        let mut rnd = ChaCha8Rng::seed_from_u64(self.seed);
        self.task_feeds = self.feedpoints.choose_multiple(&mut rnd, ntasks).map(|x| *x).collect();
    }

    fn step(&self, state: State, action: u8) -> PyResult<Vec<(State, f64, String)>> {
        let v = match self.step_(state, action){
            Ok(result) => { result }
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Could not step the environment forward => {:?}", e
                )))
            }
        };
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
                for r in 2..size - 2 { 
                    racks.insert((c as i8, r as i8));
                }
                count += 1;
            } else {
                count = 0;
            }
        }
        racks
    }

    fn pos_addition(
        &self,
        pos: &Point, 
        dir: &[i8; 2],
        size_max: i8
    ) -> Point {
        let pos_: [i8; 2] = [pos.0, pos.1];
        let mut add_ = Self::new_coord_from(pos_.iter().zip(dir).map(|(a, b)| a + b));
        for a in add_.iter_mut() {
            if *a < 0 {
                *a = 0;
            } else if *a > size_max - 1 {
                *a = size_max - 1;
            }
        }
        (add_[0], add_[1])
    }

    fn new_coord_from<F: Iterator<Item=i8>>(src: F) -> [i8; 2] {
        let mut result = [0; 2];
        for (rref, val) in result.iter_mut().zip(src) {
            *rref = val;
        }
        result
    }

    fn process_word(&self, current_state: &State, new_state: &State) -> String {
        // The 'character' of the 'word' should be the rack position
        // i.e. did the agent move to the rack corresponding to the task
        // 
        // The second 'character' in the 'word' should be the status of whether the agent
        // picked something up or not
        let mut word: [&str; 2] = [""; 2];
        if current_state.1 == 0 && new_state.1 == 1 {
            // add pick up
            word[1] = "P";
        } else if current_state.1 == 1 && new_state.1 == 0 {
            // add drop
            word[1] = "D";
        } else if current_state.1 == 1 && new_state.1 == 1 {
            word[1] = "C"
        } else {
            word[1] = "NC"
        }
        if new_state.0 == self.task_racks[self.current_task.unwrap()] {
            // then the rack is in position
            word[0] = "R";
        } else if new_state.0 == self.task_feeds[self.current_task.unwrap()] {
            word[0] = "F";
        } else if current_state.2.is_some() && 
            current_state.2.unwrap() == new_state.0 && current_state.1 == 0 {
            word[0] = "R";
        } else {
            word[0] = "NFR";
        }
        format!("{}_{}", word[0], word[1])
    }
}

impl Env<State> for Warehouse {
    fn step_(&self, state: State, action: u8) -> Result<Vec<(State, f64, String)>, String> {
        let mut v: Vec<(State, f64, String)> = Vec::new();
        if vec![0, 1, 2, 3].contains(&action) {
            let direction: &[i8; 2] = self.action_to_dir.get(&action).unwrap();
            
            let agent_new_loc = self.pos_addition( 
                &state.0, 
                direction, 
                self.size as i8, 
            );
            
            if state.1 == 0 {
                let new_state = (agent_new_loc, 0, state.2);
                // TODO we need to convert the words to how they should be represented with 
                // respect to the DFA because this is a labelled MDP
                let w = self.process_word(&state, &new_state);
                v.push((new_state, 1.0, w));
            } else {
                // an agent is restricted to move with corridors
                // check that the agent is in a corridor, if not, then
                // it cannot proceed in a direction that is not a corridor
                if self.racks.contains(&state.0) {
                    if self.racks.contains(&agent_new_loc) {
                        let w = self.process_word(&state, &state);
                        v.push((state, 1.0, w));
                    } else {
                        // Define the failure scenario
                        let success_rack_pos = Some(agent_new_loc);
                        let success_state: State = (agent_new_loc, 1, success_rack_pos);
                        let success_word = self.process_word(&state, &success_state);
                        //println!("{:?}, {} -> {:?} => {}", 
                        //    state, action, success_state, success_word
                        //);
                        let fail_state: State = (agent_new_loc, 0, state.2);
                        let fail_word = self.process_word(&state, &fail_state);

                        v.push((success_state, 0.99, success_word));
                        v.push((fail_state, 0.01, fail_word));
                    }
                } else {
                    // Define the failure scenario
                    let success_rack_pos = Some(agent_new_loc);
                    let success_state: State = (agent_new_loc, 1, success_rack_pos);
                    let success_word = self.process_word(&state, &success_state);
                    //println!("{:?}, {} -> {:?} => {}", 
                    //    state, action, success_state, success_word
                    //);
                    let fail_state: State = (agent_new_loc, 0, state.2);
                    let fail_word = self.process_word(&state, &fail_state);
                    
                    v.push((success_state, 0.99, success_word));
                    v.push((fail_state, 0.01, fail_word));
                }
            };
        } else if action == 4 {
            if state.1 == 0 {
                // if the agent is in a rack position then it may carry a rack
                // OR is the agent is not in a rack position but is superimposed on 
                // a rack that is in a corridor then it may pick up the rack. 
                let cmp_state = state.2;
                if self.racks.contains(&state.0) {
                    let new_rack_pos = Some(state.0);
                    let new_state = (state.0, 1, new_rack_pos);
                    let word = self.process_word(&state, &new_state);
                    v.push((new_state, 1.0, word));
                } else if cmp_state.is_some() {
                    if cmp_state.unwrap() == state.0 {
                        let new_rack_pos = Some(state.0);
                        let new_state = (state.0, 1, new_rack_pos);
                        let word = self.process_word(&state, &new_state);
                        v.push((new_state, 1.0, word));
                    } else {
                        let new_state = (state.0, 0, state.2);
                        let word = self.process_word(&state, &new_state);
                        v.push((new_state, 1.0, word));
                    }
                } else {
                    let new_state = (state.0, 0, state.2);
                    let word = self.process_word(&state, &new_state);
                    v.push((new_state, 1.0, word));
                }
            } else {
                let word = self.process_word(&state, &state);
                v.push((state, 1.0, word));
            }
            //println!("{:?} -> {:?}", state, v);
        } else if action == 5 {
            if state.1 == 1 {
                // this agent is currently carrying something
                // therefore, drop the rack at the current agent position
                let new_state = (state.0, 0, Some(state.0));
                let word = self.process_word(&state, &new_state);
                v.push((new_state, 1.0, word));
            } else {
                let word = self.process_word(&state, &state);
                v.push((state, 1.0, word));
            }
        } else {
            return Err("action not registered".to_string())
        }
        Ok(v)
    }

    fn get_init_state(&self) -> State {
        // with the current task construct the initial state
        let agent_pos = self.agent_initial_locs[self.current_task.unwrap()];
        (agent_pos, 0, None)
    }

    fn set_task(&mut self, task_id: usize) {
        self.current_task = Some(task_id);
    }
}


#[pyfunction]
#[pyo3(name="construct_prod_test")]
pub fn test_prod(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f64>,
    eps: f64,
    mut outputs: MDPOutputs
) -> (Vec<f64>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    env.set_task(0);
    let pmdp = build_model(
        (env.get_init_state(), 0), 
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
    outputs.prod_state_map = pmap;
    (pi, outputs)
}


#[pyfunction]
#[pyo3(name="scheduler_synth")]
pub fn warehouse_scheduler_synthesis(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f64>,
    target: Vec<f64>,
    eps: f64
) -> (
    Vec<f64>, 
    HashMap<usize, HashMap<(i32, i32), Vec<f64>>>, 
    Vec<f64>,
    MDPOutputs
) {
    // To construct a warehouse scheduler synthesis which is a wrapper around
    // the generic scheduler synthesis
    let result = 
        generic_scheduler_synthesis(model, env, w, eps, target);
    
    // wrap the result into a PyValueError
    result.unwrap()
}
