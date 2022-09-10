use crate::agent::agent::{MDP, Env};
use crate::scpm::model::{SCPM, build_model, MOProductMDP};
use crate::algorithm::synth::{process_scpm, scheduler_synthesis};
use crate::dfa::dfa::{DFA, Expression};
use crate::algorithm::dp::value_iteration;
use crate::{random_sched};
use hashbrown::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use serde_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;
//use rand::seq::SliceRandom;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

type Point = (i8, i8);
type State = (Point, u8, Option<Point>);

// ---------------------------------------------------------------
// Grid Helper Functions
// ---------------------------------------------------------------

struct WordData<'a> {
    current: State,
    next: State,
    warehouse_data: Data<'a>,
    current_task_id: usize
}


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

fn new_coord_from<F: Iterator<Item=i8>>(src: F) -> [i8; 2] {
    let mut result = [0; 2];
    for (rref, val) in result.iter_mut().zip(src) {
        *rref = val;
    }
    result
}

fn pos_addition(
    pos: &Point, 
    dir: &[i8; 2],
    size_min: i8,
    size_max: i8
) -> Point {
    let pos_: [i8; 2] = [pos.0, pos.1];
    let mut add_ = new_coord_from(pos_.iter().zip(dir).map(|(a, b)| a + b));
    for a in add_.iter_mut() {
        if *a < 0 {
            *a = 0;
        } else if *a > size_max - 1 {
            *a = size_max - 1;
        }
    }
    (add_[0], add_[1])
}

#[derive(Serialize, Deserialize)]
struct Word {
    a: Point,
    c: u8,
    r: Option<Point>
}

impl Word {
    fn new(a: Point, c: u8, r: Option<Point>) -> Self {
        Self {a, c, r}
    }
}


// ---------------------------------------------------------------
// Env MDP implementation
// ---------------------------------------------------------------
#[derive(Debug)]
struct Data<'a> {
    xsize: usize,
    ysize: usize,
    racks: &'a HashSet<Point>,
    action_to_dir: &'a HashMap<u8, [i8; 2]>,
    feedpoints: &'a [Point],
    rackpoints: &'a [Point]
}

impl<'a> Data<'a> {
    fn new(
        xsize: usize,
        ysize: usize,
        racks: &'a HashSet<Point>, 
        action_to_dir: &'a HashMap<u8, [i8; 2]>,
        feedpoints: &'a [Point],
        rackpoints: &'a [Point]
    ) -> Self {
        Data {
            xsize,
            ysize,
            racks,
            action_to_dir,
            feedpoints,
            rackpoints
        }
    }
}

impl<'a> Env<State> for MDP<State, Data<'a>> {
    fn step(&mut self, s: i32, action: u8) 
        -> Result<Vec<(i32, f64, String)>, String> {
        //
        //let mut returnv: Vec<(i32, f64, String)> = Vec::new();
        // convert the state from i32 into its real representation
        //println!("attempting to retrieve: {}", s);
        let state: State = *self.reverse_state_map.get(&s).expect(&format!("couldn't find: {}", s));
        //
        let mut v: Vec<(i32, f64, String)> = Vec::new();
        if vec![0, 1, 2, 3].contains(&action) {
            let direction: &[i8; 2] = self.extraData.action_to_dir.get(&action).unwrap();
            
            let agent_new_loc = pos_addition( 
                &state.0, 
                direction, 
                0, 
                self.extraData.xsize as i8, 
            );
            
            if state.1 == 0 {
                let new_state = (agent_new_loc, 0, state.2);
                let word = Word::new(new_state.0, new_state.1, new_state.2);
                let serialised_word = serde_json::to_string(&word).unwrap();
                let i32newstate = self.get_or_mut_state(&new_state);
                v.push((i32newstate, 1.0, serialised_word));
            } else {
                // an agent is restricted to move with corridors
                // check that the agent is in a corridor, if not, then
                // it cannot proceed in a direction that is not a corridor
                if self.extraData.racks.contains(&state.0) {
                    if self.extraData.racks.contains(&agent_new_loc) {
                    let word = Word::new(state.0, state.1, state.2);
                    v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
                    } else {
                        // Define the failure scenario
                        let success_rack_pos = Some(agent_new_loc);
                        let success_state: State = (agent_new_loc, 1, success_rack_pos);
                        let i32success_state = self.get_or_mut_state(&success_state);
                        // The rack moves with the agent
                        let success_word = Word::new(agent_new_loc, 1, success_rack_pos); 
                        let fail_state: State = (agent_new_loc, 0, state.2);
                        let fail_word = Word::new(agent_new_loc, 0, state.2);
                        let i32fail_state = self.get_or_mut_state(&fail_state);
                        v.push((i32success_state, 0.99, serde_json::to_string(&success_word).unwrap()));
                        v.push((i32fail_state, 0.01, serde_json::to_string(&fail_word).unwrap()));
                    }
                } else {
                    // Define the failure scenario
                    let success_rack_pos = Some(agent_new_loc);
                    let success_state: State = (agent_new_loc, 1, success_rack_pos);
                    let i32success_state = self.get_or_mut_state(&success_state);
                    // The rack moves with the agent
                    let success_word = Word::new(agent_new_loc, 1, success_rack_pos); 
                    let fail_state: State = (agent_new_loc, 0, state.2);
                    let fail_word = Word::new(agent_new_loc, 0, state.2);
                    let i32fail_state = self.get_or_mut_state(&fail_state);
                    v.push((i32success_state, 0.99, serde_json::to_string(&success_word).unwrap()));
                    v.push((i32fail_state, 0.01, serde_json::to_string(&fail_word).unwrap()));
                }
            };
        } else if action == 4 {
            if state.1 == 0 {
                // if the agent is in a rack position then it may carry a rack
                // OR is the agent is not in a rack position but is superimposed on 
                // a rack that is in a corridor then it may pick up the rack. 
                let cmp_state = state.2;
                if self.extraData.racks.contains(&state.0) {
                    let new_rack_pos = Some(state.0);
                    let word = Word::new(state.0, 1, new_rack_pos);
                    let new_state = (state.0, 1, new_rack_pos);
                    let i32newstate = self.get_or_mut_state(&new_state);
                    v.push((i32newstate, 1.0, serde_json::to_string(&word).unwrap()));
                } else if cmp_state.is_some() {
                    if cmp_state.unwrap() == state.0 {
                        let new_rack_pos = Some(state.0);
                        let word = Word::new(state.0, 1, new_rack_pos);
                        let new_state = (state.0, 1, new_rack_pos);
                        let i32newstate = self.get_or_mut_state(&new_state);
                        v.push((i32newstate, 1.0, serde_json::to_string(&word).unwrap()));
                    } else {
                        let word = Word::new(state.0, 0, state.2);
                        let new_state = (state.0, 0, state.2);
                        let i32newstate = self.get_or_mut_state(&new_state);
                        v.push((i32newstate, 1.0, serde_json::to_string(&word).unwrap()));
                    }
                } else {
                    let word = Word::new(state.0, 0, state.2);
                    let new_state = (state.0, 0, state.2);
                    let i32newstate = self.get_or_mut_state(&new_state);
                    v.push((i32newstate, 1.0, serde_json::to_string(&word).unwrap()));
                }
            } else {
                let word = Word::new(state.0, 1, state.2);
                v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
            }
        } else if action == 5 {
            if state.1 == 1 {
                // this agent is currently carrying something
                // therefore, drop the rack at the current agent position
                let new_rack_pos = Some(state.0);
                let word = Word::new(state.0, 0, new_rack_pos);
                v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
            } else {
                let word = Word::new(state.0, 0, state.2);
                v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
            }
        } else {
            return Err("action not registered".to_string())
        }
        Ok(v)
    }
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

#[pyfunction]
pub fn place_racks(xsize: usize, ysize: usize) 
-> HashSet<Point> {
    let mut racks: HashSet<Point> = HashSet::new();
    let mut count: usize = 0;
    assert!(xsize > 5);
    assert!(ysize > 4);
    for c in 2..xsize - 2 {
        if count < 2 {
            for r in 1..ysize - 2 { 
                racks.insert((c as i8, r as i8));
            }
            count += 1;
        } else {
            count = 0;
        }
    }
    racks
}


#[pyfunction]
#[pyo3(name="scheduler_synthesis")]
pub fn warehouse_scheduler_synthesis(
    model: &SCPM, 
    w: Vec<f64>, 
    eps: f64, 
    target: Vec<f64>,
    xsize: usize,
    ysize: usize,
    initial_states: Vec<Point>,  // refers to the agent grid state
    feedpoints: Vec<Point>,
    racks: HashSet<Point>,
    rack_tasks: Vec<Point>,
    mut outputs: MDPOutputs
) -> PyResult<(
        Vec<f64>, 
        usize, 
        HashMap<usize, HashMap<(i32, i32), Vec<f64>>>,
        MDPOutputs
    )> {
    // construct the product MDPs
    let t1 = Instant::now();
    let mut rnd = ChaCha8Rng::seed_from_u64(1234);
    println!("Constructing MDP of agent environment");
    let init_state: State = ((0, 0), 0, None);
    let mut action_to_dir: HashMap<u8, [i8; 2]> = HashMap::new();
    let dirs = vec![(0, [1, 0]),(1, [0, 1]),(2, [-1, 0]),(3, [0, -1])];
    for (k, v) in dirs.iter() {
        action_to_dir.insert(*k, *v);
    }
    // TODO move this out of the 
    let data: Data = Data::new(xsize, ysize, &racks, &action_to_dir, &feedpoints[..], &rack_tasks[..]);
    let actions: Vec<u8> = (0..6).collect();
    let mut mdp: MDP<State, Data> = MDP::new(data, init_state, actions);
    set_state_space(&mut mdp);
    println!("built in: {:?}", t1.elapsed().as_secs_f32());
    println!("MDP state space |S|: {}", mdp.states.len());

    let mut prod_init_state: Vec<(i32, i32)> = Vec::new();
    for i in 0..initial_states.len() {
        let state_ = (initial_states[i], 0, None);
        prod_init_state.push((*mdp.state_map.get(&state_).unwrap(), 0));
    }
    let prods = model.construct_products(&mut mdp, &prod_init_state[..]);
    let (pis, alloc, t_new, l) = scheduler_synthesis(model, &w[..], &eps, &target[..], prods);
    //println!("{:?}", pis);
    println!("alloc: \n{:.3?}", alloc);
    // convert output schedulers to 
    // we need to construct the randomised scheduler here, then the output from the randomised
    // scheduler, which will already be from a python script, will be the output of this function
    let weights = random_sched(alloc, t_new.to_vec(), l, model.tasks.size, model.num_agents);
    outputs.state_map = mdp.state_map;
    outputs.rev_state_map = mdp.reverse_state_map;
    match weights {
        Some(w) => { return Ok((w, l, pis, outputs)) }
        None => { 
            return Err(PyValueError::new_err(
                format!(
                    "Randomised scheduler weights could not be found for 
                    target vector: {:?}", 
                    t_new)
                )
            )
        }
    }
}

#[pyfunction]
pub fn test_scpm(
    model: &SCPM,
    w: Vec<f64>,
    eps: f64,
    xsize: usize,
    ysize: usize,
    initial_states: Vec<Point> // vector of initial agent positions
) {
    // construct the product MDPs
    let t1 = Instant::now();
    let mut rnd = ChaCha8Rng::seed_from_u64(1234);
    println!("Constructing MDP of agent environment");
    let init_state: State = ((0, 0), 0, None);
    let racks = place_racks(xsize, ysize);
    // TODO move random rack task generation outside of lib and as an input 
    // from the python interface
    let mut action_to_dir: HashMap<u8, [i8; 2]> = HashMap::new();
    let dirs = vec![(0, [1, 0]),(1, [0, 1]),(2, [-1, 0]),(3, [0, -1])];
    for (k, v) in dirs.iter() {
        action_to_dir.insert(*k, *v);
    }
    let rack_points: Vec<Point> = { 
        let racksv: Vec<Point> = racks.iter().map(|x| *x).collect();
        racksv.choose_multiple(&mut rnd, model.tasks.size).map(|x| *x).collect()
    };
    let feed_points_ = vec![(xsize as i8 - 1, ysize as i8 / 2)]; // [(0, ysize as i8 / 2)]; //, [(xsize as i8 - 1, ysize as i8 / 2)];
    // for each task choose a random feedpoint
    let mut feeds = Vec::new();
    for _ in 0..model.tasks.size {
        feeds.push(*feed_points_.choose(&mut rand::thread_rng()).unwrap());
    }
    let data: Data = Data::new(xsize, ysize, &racks, &action_to_dir, &feeds[..], &rack_points[..]);
    let actions: Vec<u8> = (0..6).collect();
    let mut mdp: MDP<State, Data> = MDP::new(data, init_state, actions);
    set_state_space(&mut mdp);
    println!("built in: {:?}", t1.elapsed().as_secs_f32());
    println!("MDP state space |S|: {}", mdp.states.len());
    let mut prod_init_state: Vec<(i32, i32)> = Vec::new();
    for i in 0..initial_states.len() {
        let state_ = (initial_states[i], 0, None);
        prod_init_state.push((*mdp.state_map.get(&state_).unwrap(), 0));
    }
    let prods = model.construct_products(&mut mdp, &prod_init_state[..]);
    println!("Processing MDPS");
    let t1 = Instant::now();
    let (r, _prods, _pis, alloc) = process_scpm(model, &w[..], &eps, prods);
    println!("r: {:?}", r);
    println!("alloc: {:?}", alloc);
    println!("Time to multiprocess MDPs: {:?}", t1.elapsed().as_secs_f32());
}

#[pyfunction]
#[pyo3(name="construct_prod_test")]
pub fn test_prod(
    model: &SCPM, 
    xsize: usize,
    ysize: usize,
    w: Vec<f64>,
    eps: f64,
    init_agent_pos: Point,
    rack_points: Vec<Point>,
    task_feeds: Vec<Point>,
    feedpoints: Vec<Point>,
    racks: HashSet<Point>,
    mut outputs: MDPOutputs
)
-> (Vec<f64>, MDPOutputs) {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    let init_state: State = (init_agent_pos, 0, None);
    //let racks = place_racks(xsize, ysize);
    let mut action_to_dir: HashMap<u8, [i8; 2]> = HashMap::new();
    let dirs = vec![(0, [1, 0]),(1, [0, 1]),(2, [-1, 0]),(3,[0, -1])];
    for (k, v) in dirs.iter() {
        action_to_dir.insert(*k, *v);
    } // [(0, ysize as i8 / 2)]; //, [(xsize as i8 - 1, ysize as i8 / 2)];
    // for each task choose a random feedpoint
    let data: Data = Data::new(xsize, ysize, &racks, &action_to_dir, &task_feeds[..], &rack_points[..]);
    println!("{:?}", data);
    println!("Rack Task: {:?}, Feed Points: {:?}", rack_points, feedpoints);
    let actions: Vec<u8> = (0..6).collect();
    let mut mdp: MDP<State, Data> = MDP::new(data, init_state, actions);
    set_state_space(&mut mdp);
    println!("built in: {:?}", t1.elapsed().as_secs_f32());
    println!("MDP state space |S|: {}", mdp.states.len());
    //model.construct_products(&mut mdp);
    let pmdp = build_model(
        (mdp.init_state, 0), 
        &mut mdp,
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
    outputs.state_map = mdp.state_map;
    outputs.rev_state_map = mdp.reverse_state_map;
    let mut pmap: HashMap<(i32, i32), HashMap<(i32, i32), usize>> = HashMap::new();
    decode_policy(&pi, &outputs, &pmdp);
    pmap.insert((0, 0), pmdp.state_map);
    outputs.prod_state_map = pmap;
    (pi, outputs)
}

pub fn decode_policy(policy: &[f64], outputs: &MDPOutputs, pmdp: &MOProductMDP) {
    for (i, act) in policy.iter().enumerate().filter(|(s, act)| **act == 5.0) {
        // the index of the state is i so we first need to decode it into a product state
        let (s, q) = pmdp.get_reverse_state_map().get(&i).unwrap();
        // now convert the s index to an mdp state
        let warehouse_state = outputs.rev_state_map.get(s).unwrap();
        if warehouse_state.0 == (9, 5) && *q == 4 {
            println!("s: {:?}, q: {}, prod: idx: {}", warehouse_state, q, i);
        }
    }
}

impl<'a> Expression<WordData<'a>> for DFA {
    fn conversion(&self, q: i32, data: &WordData) -> String {
        match q {
            0 => {
                if data.current.0 == data.warehouse_data.rackpoints[]
            }
            1 => {

            }
            2 => {

            }
            3 => {

            }
            4 => {}
            5 => {}
            6 => {}
            7 => {}
            _ => {}
        }
    }
}