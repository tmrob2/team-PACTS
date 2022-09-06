use crate::agent::agent::{MDP, Env};
use crate::scpm::model::{SCPM, build_model};
use crate::algorithm::synth::{process_scpm, scheduler_synthesis};
use crate::dfa::dfa::{DFA, ProcessAlphabet};
use crate::algorithm::dp::value_iteration;
use crate::random_sched;
use hashbrown::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use serde_json;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::time::Instant;
//use rand::seq::SliceRandom;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

type State = ((i8, i8), u8, Option<(u8, u8)>);

// ---------------------------------------------------------------
// Grid Helper Functions
// ---------------------------------------------------------------

fn pos_addition(
    pos: &(i8, i8), 
    dir: &(i8, i8),
    xmin: usize, 
    xmax: usize, 
    ymin: usize, 
    ymax: usize
) -> (i8, i8) {
    
    let x = if (pos.0 + dir.0) > xmin as i8 && (pos.0 + dir.0) < xmax as i8 - 1 {
        pos.0 + dir.0
    } else if (pos.0 + dir.0) < xmin as i8 {
        0 as i8
    } else {
        (xmax - 1) as i8
    };

    let y = if (pos.1 + dir.1) > ymin as i8 && (pos.1 + dir.1) < ymax as i8 - 1 {
        pos.1 + dir.1
    } else if (pos.1 + dir.1) < ymin as i8 {
        0 as i8
    } else {
        (ymax - 1) as i8
    };

    (x, y)
}

#[derive(Serialize, Deserialize)]
struct Word {
    a: (i8, i8),
    c: u8,
    r: Option<(u8, u8)>
}

impl Word {
    fn new(a: (i8, i8), c: u8, r: Option<(u8, u8)>) -> Self {
        Self {a, c, r}
    }
}


// ---------------------------------------------------------------
// Env MDP implementation
// ---------------------------------------------------------------

struct Data<'a> {
    xsize: usize,
    ysize: usize,
    racks: &'a HashSet<(i8, i8)>,
    action_to_dir: &'a HashMap<u8, (i8, i8)>,
    feedpoints: &'a [(i8, i8)],
    rackpoints: &'a [(i8, i8)]
}

impl<'a> Data<'a> {
    fn new(
        xsize: usize,
        ysize: usize,
        racks: &'a HashSet<(i8, i8)>, 
        action_to_dir: &'a HashMap<u8, (i8, i8)>,
        feedpoints: &'a [(i8, i8)],
        rackpoints: &'a [(i8, i8)]
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
            let direction: &(i8, i8) = self.extraData.action_to_dir.get(&action).unwrap();
            
            let agent_new_loc = pos_addition( 
                &state.0, 
                direction, 
                0, 
                self.extraData.xsize, 
                0, 
                self.extraData.ysize
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
                        let success_rack_pos = Some((agent_new_loc.0 as u8, agent_new_loc.1 as u8));
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
                    let success_rack_pos = Some((agent_new_loc.0 as u8, agent_new_loc.1 as u8));
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
                let cmp_state: Option<(i8, i8)> = match state.2 {
                    Some(val) => { Some((val.0 as i8, val.1 as i8)) }
                    None => { None }
                };
                if self.extraData.racks.contains(&state.0) {
                    let new_rack_pos = Some((state.0.0 as u8, state.0.1 as u8));
                    let word = Word::new(state.0, 1, new_rack_pos);
                    let new_state = (state.0, 1, new_rack_pos);
                    let i32newstate = self.get_or_mut_state(&new_state);
                    v.push((i32newstate, 1.0, serde_json::to_string(&word).unwrap()));
                } else if cmp_state.is_some() {
                    if cmp_state.unwrap() == state.0 {
                        let new_rack_pos = Some((state.0.0 as u8, state.0.1 as u8));
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
                let new_rack_pos = Some((state.0.0 as u8, state.0.1 as u8));
                let word = Word::new(state.0, 0, new_rack_pos);
                v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
            } else {
                let word = Word::new(state.0, 0, state.2);
                v.push((s, 1.0, serde_json::to_string(&word).unwrap()));
            }
        } else {
            return Err("action not registered".to_string())
        }
        //
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

fn place_racks(xsize: usize, ysize: usize) 
-> HashSet<(i8, i8)> {
    let mut racks: HashSet<(i8, i8)> = HashSet::new();
    let mut count: usize = 0;
    assert!(xsize > 5);
    assert!(ysize > 4);
    for c in 2..xsize - 2 {
        for r in 1..ysize - 2 {
            if count < 2 {
                count += 1;
                racks.insert((r as i8, c as i8));
            } else {
                count = 0;
            }
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
    initial_states: Vec<(i8, i8)>,  // refers to the agent grid state
    feedpoints: Vec<(i8, i8)>,
    rack_tasks: Vec<(i8, i8)>
) -> PyResult<(Vec<f64>, usize, HashMap<usize, HashMap<(i32, i32), Vec<f64>>>)> {
    // construct the product MDPs
    let t1 = Instant::now();
    let mut rnd = ChaCha8Rng::seed_from_u64(1234);
    println!("Constructing MDP of agent environment");
    let init_state: State = ((0, 0), 0, None);
    let racks = place_racks(xsize, ysize);
    let mut action_to_dir: HashMap<u8, (i8, i8)> = HashMap::new();
    let dirs = vec![(0,(1, 0)),(1, (0, 1)),(2,(-1, 0)),(3,(0, -1))];
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
    match weights {
        Some(w) => { return Ok((w, l, pis)) }
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
    initial_states: Vec<(i8, i8)> // vector of initial agent positions
) {
    // construct the product MDPs
    let t1 = Instant::now();
    let mut rnd = ChaCha8Rng::seed_from_u64(1234);
    println!("Constructing MDP of agent environment");
    let init_state: State = ((0, 0), 0, None);
    let racks = place_racks(xsize, ysize);
    // TODO move random rack task generation outside of lib and as an input 
    // from the python interface
    let mut action_to_dir: HashMap<u8, (i8, i8)> = HashMap::new();
    let dirs = vec![(0,(1, 0)),(1, (0, 1)),(2,(-1, 0)),(3,(0, -1))];
    for (k, v) in dirs.iter() {
        action_to_dir.insert(*k, *v);
    }
    let rack_points: Vec<(i8, i8)> = { 
        let racksv: Vec<(i8, i8)> = racks.iter().map(|x| *x).collect();
        racksv.choose_multiple(&mut rnd, model.tasks.size).map(|x| *x).collect()
    };
    let feed_points_ = [(xsize as i8 - 1, ysize as i8 / 2)]; // [(0, ysize as i8 / 2)]; //, [(xsize as i8 - 1, ysize as i8 / 2)];
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
    eps: f64
) {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    let init_state: State = ((0, 0), 0, None);
    let racks = place_racks(xsize, ysize);
    let mut action_to_dir: HashMap<u8, (i8, i8)> = HashMap::new();
    let dirs = vec![(0,(1, 0)),(1, (0, 1)),(2,(-1, 0)),(3,(0, -1))];
    let mut rnd = ChaCha8Rng::seed_from_u64(1234);
    for (k, v) in dirs.iter() {
        action_to_dir.insert(*k, *v);
    }
    let rack_points: Vec<(i8, i8)> = { 
        let racksv: Vec<(i8, i8)> = racks.iter().map(|x| *x).collect();
        racksv.choose_multiple(&mut rnd, model.tasks.size).map(|x| *x).collect()
    };
    let feed_points_ = [(xsize as i8 - 1, ysize as i8 / 2)]; // [(0, ysize as i8 / 2)]; //, [(xsize as i8 - 1, ysize as i8 / 2)];
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
    //model.construct_products(&mut mdp);
    let pmdp = build_model(
        (0, 0), 
        //agent, 
        //&mdp_rewards,
        &mut mdp,
        //&mdp_available_actions,
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

    //let (r, _prods, pis, alloc) = process_scpm(model, &w[..], &eps, prods);
    println!("r {:?}", r);
    //println!("pis {:?}", pi);
    //println!("alloc: {:?}", alloc);

    // then we will use the allocation to compute the randomised scheduler
}

impl<'a> ProcessAlphabet<Data<'a>> for DFA {
    fn word_router(&self, w: &str, q: i32, data: &Data, task: usize) -> String {
        // The word that comes in will need to be deserialised and then processed in the word map
        let word: Word = serde_json::from_str(w).unwrap();

        let result: String = match q {
            0 => {
                if word.a == data.rackpoints[task] {
                    if word.c == 0 {
                        "a".to_string()
                    } else {
                        "fail".to_string()
                    }
                } else {
                    "na".to_string()
                }
            }
            1 => {
                if word.c == 1 {
                    "r".to_string()
                } else {
                    "nr".to_string()
                }
            }
            2 => {
                if word.c == 1 {
                    let rack_pos_ = word.r.unwrap();
                    let rack_pos = (rack_pos_.0 as i8, rack_pos_.1 as i8);
                    // check that both the agent and the rack are back at
                    // the designated feedpoint
                    if data.feedpoints[task as usize] == rack_pos 
                        && data.feedpoints[task as usize] == word.a {
                        "cf".to_string()
                    } else {
                        "ncf".to_string()
                    }
                } else {
                    "fail".to_string()
                }
            }
            3 => {
                if word.c == 1 {
                    let rack_pos_ = word.r.unwrap();
                    let rack_pos = (rack_pos_.0 as i8, rack_pos_.1 as i8);
                    // check that both the agent and the rack are back at
                    // the designated feedpoint
                    if data.rackpoints[task as usize] == rack_pos 
                        && data.rackpoints[task as usize] == word.a {
                        "ca".to_string()
                    } else {
                        "nca".to_string()
                    }
                } else {
                    "fail".to_string()
                }
            }
            4 => {
                if word.c == 0 {
                    //let rack_pos_ = word.r.unwrap();
                    //let rack_pos = (rack_pos_.0 as i8, rack_pos_.1 as i8);
                    // check that both the agent and the rack are back at
                    // the designated feedpoint
                    "er".to_string()
                } else {
                    "ca".to_string()
                }
            }
            5 => {"true".to_string()}
            6 => {"true".to_string()}
            7 => {"true".to_string()}
            _ => {panic!("Q state not found")}
        };

        result

    }
}