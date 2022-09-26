use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use crate::agent::agent::Env;
use crate::dfa::dfa::DFA;
use crate::scpm::model::{SCPM, build_model};
//use crate::algorithm::synth::{process_scpm, scheduler_synthesis};
use crate::algorithm::dp::value_iteration;
use crate::generic_scheduler_synthesis;
//use crate::{random_sched};
use pyo3::exceptions::PyValueError;
use std::hash::Hash;
//use std::env::set_current_dir;
//use std::hash::Hash;
use std::time::Instant;
//use rand::seq::SliceRandom;
//use rand::prelude::*;
//use rand_chacha::ChaCha8Rng;
use array_macro::array;
use crate::executor::executor::{Execution, DefineSerialisableExecutor};
use std::collections::HashMap as DefaultHashMap;

//-------
// Types 
//-------
type Point = (i8, i8);
type State = (Point, u8, Option<Point>);

//--------
// Helpers
//--------

// TODO Make the Executor String based and generic so that it does not need to 
// be specified in the env file
#[pyclass]
pub struct Executor {
    state_maps: DefaultHashMap<(i32, i32), DefaultHashMap<(State, i32), usize>>,
    schedulers: DefaultHashMap<(i32, i32), Vec<f64>>,
    #[pyo3(get)]
    agent_task_allocations: DefaultHashMap<i32, Vec<i32>>,
    #[pyo3(get)]
    priority_agent_allocations: DefaultHashMap<i32, Vec<i32>>,
    tasks: DefaultHashMap<i32, DFA>,
    task_map: DefaultHashMap<i32, (i32, i32)>,
}

#[pymethods]
impl Executor {
    #[new]
    fn new(nagents: usize) -> Executor {
        let mut init_agent_alloc: DefaultHashMap<i32, Vec<i32>> = DefaultHashMap::new();
        let mut init_priority_agent_alloc: DefaultHashMap<i32, Vec<i32>> = DefaultHashMap::new();
        for agent in 0..nagents {
            init_agent_alloc.insert(agent as i32, Vec::new());
            init_priority_agent_alloc.insert(agent as i32, Vec::new());
        }

        Executor { 
            state_maps: DefaultHashMap::new(), 
            schedulers: DefaultHashMap::new(), 
            agent_task_allocations: init_agent_alloc, 
            priority_agent_allocations: init_priority_agent_alloc,
            tasks: DefaultHashMap::new(), 
            task_map: DefaultHashMap::new(), 
        }
    }

    fn get_next_task(&mut self, agent: i32) -> (Option<i32>, i32) {
        // Check if there are tasks in 
        match self.priority_agent_allocations.get_mut(&agent) {
            Some(tasks_remaining) => { 
                if tasks_remaining.len() > 0 {
                    println!("agent {} priority tasks: {:?}", agent, tasks_remaining);
                    (tasks_remaining.pop(), 1)
                } else {
                    match self.agent_task_allocations.get_mut(&agent) {
                        Some(tasks_remaining) => { 
                            (tasks_remaining.pop(), 0)
                        }
                        None => {
                            (None, 0)
                        }
                    }
                }
            }
            None => {
                (None, 0)
            }
        }        
    }

    fn get_task_mapping(&self, task: i32) -> (i32, i32) {
        *self.task_map.get(&task).unwrap()
    }

    fn check_done(&self, task: i32) -> u8 {
        self.tasks.get(&task).unwrap().check_done()
    }

    fn dfa_current_state(&self, task: i32) -> i32 {
        self.tasks.get(&task).unwrap().current_state
    }

    fn get_action(&self, agent: i32, task: i32, state: State, q: i32) -> i32 {
        //println!("state: {:?}, q: {}", state, q);
        let sidx = *self.state_maps.get(&(agent, task)).unwrap()
            .get(&(state, q))
            .expect(&format!("agent: {}, failed at {:?}, task -> {}", 
                agent, (state, q), task));
        let action = self.schedulers.get(&(agent, task)).unwrap()[sidx] as i32;
        action
    }

    fn dfa_next_state(&mut self, task: i32, q: i32, word: String) -> i32 {
        self.tasks.get_mut(&task).unwrap().next(q, word)
    }

    fn merge_exec(&mut self, other: &mut Executor, num_agents: usize, batch_size: usize) {
        for agent in 0..num_agents {
            for task in 0..batch_size {
                self.state_maps.insert(
                    (agent as i32, other.task_map.get(&(task as i32)).unwrap().0), 
                    other.state_maps.get_mut(&(agent as i32, task as i32)).unwrap().to_owned()
                );
                match other.schedulers.get_mut(&(agent as i32, task as i32)) {
                    Some(sch) => { 
                        self.schedulers.insert(
                            (agent as i32, other.task_map.get(&(task as i32)).unwrap().0),
                            sch.to_owned() 
                        );
                    }
                    None => {  }
                }
                
                self.tasks.insert(
                    other.task_map.get(&(task as i32)).unwrap().0,
                    other.tasks.get(&(task as i32)).unwrap().to_owned()
                );
            }
            // Regular task allocations
            match self.agent_task_allocations.get_mut(&(agent as i32)) {
                Some(tvec) => { 
                    match other.agent_task_allocations.get(&(agent as i32)) {
                        Some(new_alloc) => {
                            for t in new_alloc.iter() {
                                if other.task_map.get(&(*t as i32)).unwrap().1 == 0 {
                                    tvec.push(other.task_map.get(&(*t as i32)).unwrap().0); 
                                }
                            }
                        }
                        None => { }
                    };
                }
                None => { 
                    let  mut tvnew = Vec::new();
                    match other.agent_task_allocations.get(&(agent as i32)) {
                        Some(new_alloc) => { 
                            for t in new_alloc.iter() {
                                if other.task_map.get(&(*t as i32)).unwrap().1 == 0 {
                                    tvnew.push(other.task_map.get(&(*t as i32)).unwrap().0) 
                                }
                            }
                            self.agent_task_allocations.insert(
                                agent as i32, 
                                tvnew
                            );
                        }
                        None => { }
                    };
                }
            }
            // Priority task allocations
            match self.priority_agent_allocations.get_mut(&(agent as i32)) {
                Some(tvec) => { 
                    match other.agent_task_allocations.get(&(agent as i32)) {
                        Some(new_alloc) => {
                            for t in new_alloc.iter() {
                                if other.task_map.get(&(*t as i32)).unwrap().1 == 1 {
                                    tvec.push(other.task_map.get(&(*t as i32)).unwrap().0); 
                                }
                            }
                        }
                        None => { }
                    };
                }
                None => { 
                    let  mut tvnew = Vec::new();
                    match other.agent_task_allocations.get(&(agent as i32)) {
                        Some(new_alloc) => { 
                            for t in new_alloc.iter() {
                                if other.task_map.get(&(*t as i32)).unwrap().1 == 1 {
                                    tvnew.push(other.task_map.get(&(*t as i32)).unwrap().0) 
                                }
                            }
                            self.agent_task_allocations.insert(
                                agent as i32, 
                                tvnew
                            );
                        }
                        None => { }
                    };
                }
            }
        }
        self.clean_up(other);
    }

    fn clean_up(&self, other: &mut Executor) {
        drop(other);
    }

    fn print_current_task_alloc(&self, num_agents: usize) {
        for agent in 0..num_agents {
            if !self.agent_task_allocations.get(&(agent as i32)).unwrap().is_empty() {
                println!("Agent: {} -> {:?}", 
                    agent, self.agent_task_allocations.get(&(agent as i32)));
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct TaskPriority{
    task: i32, 
    priority: i32
}

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct SerialisableExecutor {
    state_maps: Vec<((i32, i32), Vec<((State, i32), usize)>)>,
    schedulers: Vec<((i32, i32), Vec<f64>)>,
    agent_task_allocations: DefaultHashMap<i32, Vec<i32>>,
    task: DefaultHashMap<i32, DFA>,
    task_map: DefaultHashMap<i32, (i32, i32)>,
}

impl DefineSerialisableExecutor<State> for SerialisableExecutor {
    fn new_(nagents: usize) -> Self {

        let mut agent_task_alloc: HashMap<i32, Vec<i32>> = HashMap::new();
        for agent in 0..nagents {
            agent_task_alloc.insert(agent as i32, Vec::new());
        }
        SerialisableExecutor { 
            state_maps: Vec::new(), 
            schedulers: Vec::new(), 
            agent_task_allocations: DefaultHashMap::new(),
            task: DefaultHashMap::new(), 
            task_map: DefaultHashMap::new(),
        }
    }

    fn set_state_map(&mut self, agent: i32, task: i32, map: Vec<((State, i32), usize)>) {
        self.state_maps.push(((agent, task), map));
    }

    fn convert_to_hashmap(&mut self, nagents: usize, ntasks: usize) -> 
        DefaultHashMap<(i32, i32), DefaultHashMap<(State, i32), usize>> {
        let mut statemap: DefaultHashMap<(i32, i32), DefaultHashMap<(State, i32), usize>> = DefaultHashMap::new();
        // start by constructing the keys of the hashmap
        for i in 0..nagents {
            for j in 0..ntasks {
                statemap.insert((i as i32, j as i32), DefaultHashMap::new());
            }
        }

        for ((i, j), v) in self.state_maps.drain(..) {
            match statemap.get_mut(&(i, j)) {
                Some(smap) => {
                    for (k, val) in v.into_iter() {
                        smap.insert(k, val);
                    }
                 }
                _ => { }
            }
        }

        statemap
    }

    fn insert_dfa(&mut self, dfa: DFA, task: i32) {
        self.task.insert(task, dfa);
    }

    fn set_scheduler(&mut self, agent: i32, task: i32, scheduler: Vec<f64>) {
        self.schedulers.push(((agent, task), scheduler));
    }

    fn set_agent_task_allocation(&mut self, agent: i32, task: i32) {
        match self.agent_task_allocations.get_mut(&agent) {
            Some(t) => {
                t.push(task);
            }
            None => { 
                self.agent_task_allocations.insert(agent, vec![task]); 
            }
        }
    }
}

#[pymethods]
impl SerialisableExecutor {
    #[new]
    fn new(nagents: Option<usize>, input_json: Option<String>) -> Self {
        match input_json {
            Some(s) => {
                serde_json::from_str(&s).unwrap()
            }
            None => { 
                SerialisableExecutor::new_(nagents.unwrap())
            }
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    fn insert_task_map(&mut self, task_map: DefaultHashMap<i32, (i32, i32)>) {
        self.task_map = task_map; //Vec::from_iter(task_map.into_iter());
    }

    fn convert_to_executor(&mut self, nagents: usize, ntasks: usize) -> Executor {
        let smap = 
            self.convert_to_hashmap(nagents, ntasks);
         
        Executor {
            state_maps: smap,
            schedulers: self.schedulers.drain(..).collect(), // Vector to hashmap creation
            agent_task_allocations: self.agent_task_allocations.drain().collect(),
            priority_agent_allocations: DefaultHashMap::new(),
            tasks: self.task.drain().collect(),
            task_map: self.task_map.drain().collect(),
        }
    }
}

#[pyclass]
pub struct ExecutorString {
    s: String
}

impl Execution<State> for SerialisableExecutor { }

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
    pub original_racks: HashSet<Point>,
    #[pyo3(get)]
    pub agent_initial_locs: Vec<Point>,
    pub action_to_dir: HashMap<u8, [i8; 2]>,
    #[pyo3(get)]
    pub task_racks_start: HashMap<i32, Point>,
    #[pyo3(get)]
    pub task_racks_end: HashMap<i32, Point>,
    #[pyo3(get)]
    pub task_feeds: HashMap<i32, Point>,
    #[pyo3(get)]
    pub words: [String; 25],
    pub seed: u64,
    pub psuccess: f64
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
        seed: u64,
        psuccess: f64
    ) -> Self {
        // call the rack function
        let mut action_map: HashMap<u8, [i8; 2]> = HashMap::new();
        for (i, act) in actions_to_dir.into_iter().enumerate() {
            action_map.insert(i as u8, act);
        }
        let w0: [String; 5] = ["RS".to_string(), "RE".to_string(), "F".to_string(), "FR".to_string(), "NFR".to_string()];
        let w1: [String; 5] = ["P".to_string(), "D".to_string(), "CR".to_string(), "CNR".to_string(), "NC".to_string()];
        let mut words: [String; 25] = array!["".to_string(); 25];
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
            original_racks: HashSet::new(),
            agent_initial_locs: initial_locations,
            action_to_dir: action_map,
            task_racks_start: HashMap::new(),
            task_racks_end: HashMap::new(),
            task_feeds: HashMap::new(),
            words,
            seed,
            psuccess
        };
        // TODO finish testing of different minimal warehouse layouts.
        new_env.racks = new_env.place_racks(size);
        // test: => new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 2));
        new_env
    }

    fn set_psuccess(&mut self, pnew: f64) {
        self.psuccess = pnew;
    }
    
    fn get_racks(&self) -> HashSet<Point> {
        self.racks.clone()
    }

    fn update_rack(&mut self, new_rack_pos: Point, old_rack_pos: Point) {
        println!("Recieved a rack update: new: {:?}, old: {:?}", new_rack_pos, old_rack_pos);
        self.racks.remove(&old_rack_pos);
        self.racks.insert(new_rack_pos);
    }

    fn add_task_rack_start(&mut self, task: i32, rack: Point) {
        self.task_racks_start.insert(task, rack);
    }

    fn remove_task_rack_start(&mut self, task: i32) {
        self.task_racks_start.remove(&task);
    }

    fn add_task_rack_end(&mut self, task: i32, rack: Point) {
        self.task_racks_end.insert(task, rack);
    }

    fn remove_task_rack_end(&mut self, task: i32) {
        self.task_racks_start.remove(&task);
    }

    
    fn add_task_feed(&mut self, task: i32, feed: Point) {
        self.task_feeds.insert(task, feed);
    }
    
    fn remove_task_feed(&mut self, task: i32) {
        self.task_feeds.remove(&task);
    }

    fn clear_env(&mut self) {
        self.task_racks_start = HashMap::new();
        self.task_racks_end = HashMap::new();
        self.task_feeds = HashMap::new();
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

    fn set_task_(&mut self, task: usize) {
        self.set_task(task);
    }

    fn get_current_task(&self) -> usize {
        self.current_task.unwrap()
    }

    fn mutate_task_racks_start(&mut self, new_hash: HashMap<i32, Point>) {
        self.task_racks_start.extend(&new_hash);
    }

    fn mutate_task_feeds(&mut self, new_hash: HashMap<i32, Point>) {
        self.task_feeds.extend(&new_hash);
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
            let current_task = self.get_current_task();
            if self.racks.contains(&new_state.0) {
                // this is a problem...but only if the rack is not the task rack
                if *self.task_racks_start.get(&(current_task as i32)).unwrap() == new_state.0 {
                    word[1] = "CNR";
                } else if self.task_racks_end.get(&(current_task as i32)).is_some() && 
                    *self.task_racks_end.get(&(current_task as i32)).unwrap() == new_state.0 {
                    // this is fine
                    word[1] = "CNR";
                } else {
                    word[1] = "CR";
                }
            } else {
                word[1] = "CNR";
            }
        } else {
            word[1] = "NC";
        }

    
        if new_state.0 == *self.task_racks_start.get(&(self.current_task.unwrap() as i32)).unwrap() {
            // then the rack is in position
            word[0] = "RS";
        } else if self.task_racks_end.get(&(self.current_task.unwrap() as i32)).is_some() && 
            new_state.0 == *self.task_racks_end.get(&(self.current_task.unwrap() as i32)).unwrap()
        {
            word[0] = "RE";
        } else if new_state.0 == *self.task_feeds.get(&(self.current_task.unwrap() as i32))
            .expect(&format!("Could not find task: {}", self.current_task.unwrap())) {
            word[0] = "F";
        } else if current_state.2.is_some() && 
            current_state.2.unwrap() == new_state.0 && current_state.1 == 0 {
            word[0] = "RS";
        } else {
            word[0] = "NFR";
        }
        format!("{}_{}", word[0], word[1])
    }
}

impl Env<State> for Warehouse {
    fn step_(&self, state: State, action: u8) -> Result<Vec<(State, f64, String)>, String> {
        let psuccess :f64 = self.psuccess;
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

                        v.push((success_state, psuccess, success_word));
                        v.push((fail_state, 1. - psuccess, fail_word));
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
                    
                    v.push((success_state, psuccess, success_word));
                    v.push((fail_state, 1. - psuccess, fail_word));
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
                let new_state = (state.0, 0, None);
                let word = self.process_word(&state, &new_state);
                v.push((new_state, 1.0, word));
            } else {
                let word = self.process_word(&state, &state);
                v.push((state, 1.0, word));
            }
        } else if action == 6 {
            // do nothing
            let word = self.process_word(&state, &state);
            v.push((state, 1.0, word));
        } else {
            return Err("action not registered".to_string())
        }
        Ok(v)
    }

    fn get_init_state(&self, agent: usize) -> State {
        // with the current task construct the initial state
        let agent_pos = self.agent_initial_locs[agent as usize];
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
    //mut outputs: MDPOutputs
) -> Vec<f64> //(Vec<f64>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    env.set_task(0);
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
    //(pi, outputs)
}


#[pyfunction]
#[pyo3(name="scheduler_synth")]
pub fn warehouse_scheduler_synthesis(
    model: &mut SCPM,
    env: &mut Warehouse,
    w: Vec<f64>,
    target: Vec<f64>,
    eps: f64,
    executor: &mut SerialisableExecutor
) -> PyResult<Vec<f64>> {
    
    // To construct a warehouse scheduler synthesis which is a wrapper around
    // the generic scheduler synthesis
    let result = 
        generic_scheduler_synthesis(model, env, w, eps, target, executor);
    
    // wrap the result into a PyValueError
    match result {
        Ok(r) => { Ok(r) }
        Err(e) => Err(PyValueError::new_err(e))
    }
}
