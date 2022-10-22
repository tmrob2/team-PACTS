#![allow(non_snake_case)]

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use crate::agent::agent::{Env};
use crate::dfa::dfa::{DFA, Mission};
use std::collections::VecDeque;
use crate::*;


pub struct MOProductMDP<S> {
    pub initial_state: (S, i32),
    pub states: Vec<(S, i32)>,
    pub actions: Vec<i32>,
    //pub rewards: HashMap<((i32, i32), i32), Vec<f64>>,
    //pub transitions: HashMap<((i32, i32), i32), Vec<((i32, i32), f64)>>,
    pub transition_mat: HashMap<i32, Triple>,
    pub rewards_mat: HashMap<i32, DenseMatrix>,
    pub agent_id: i32,
    pub task_id: i32,
    action_map: HashMap<(S, i32), Vec<i32>>,
    pub state_map: HashMap<(S, i32), usize>,
    reverse_state_map: HashMap<usize, (S, i32)>
}

impl<S> MOProductMDP<S> where S: Copy + Eq + Hash {
    pub fn new(initial_state: (S, i32), actions: &[i32], agent_id: i32, task_id: i32) -> Self {
        let mut transitions: HashMap<i32, Triple> = HashMap::new();
        let mut rewards: HashMap<i32, DenseMatrix> = HashMap::new();
        for action in actions.iter() {
            transitions.insert(*action, Triple::new());
            rewards.insert(*action, DenseMatrix::new(0,0));
        }
        
        MOProductMDP {
            initial_state,
            states: Vec::new(),
            actions: actions.to_vec(),
            rewards_mat: rewards,
            transition_mat: transitions,
            agent_id,
            task_id,
            action_map: HashMap::new(),
            state_map: HashMap::new(),
            reverse_state_map: HashMap::new()
        }
    }

    fn insert_state(&mut self, state: (S, i32)) {
        let state_idx = self.states.len();
        self.states.push(state);
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_avail_act(&mut self, state: &(S, i32), action: i32) {
        match self.action_map.get_mut(state) {
            Some(acts) => {
                if !acts.contains(&action) {
                    acts.push(action);
                }
            }
            None => {
                self.action_map.insert(*state, vec![action]);
            }
        }
    }

    pub fn get_available_actions(&self, state: &(S, i32)) -> &[i32] {
        &self.action_map.get(state).unwrap()[..]
    }

    pub fn get_state_map(&self) -> &HashMap<(S, i32), usize> {
        &self.state_map
    }

    pub fn get_reverse_state_map(&self) -> &HashMap<usize, (S, i32)> {
        &self.reverse_state_map
    }

    fn insert_state_mapping(&mut self, state: (S, i32), state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }

    fn insert_action(&mut self, action: i32) {
        if !self.actions.contains(&action) {
            self.actions.push(action);
        }
    }

    fn insert_transition(&mut self, state: usize, action: i32, sprime: usize, p: f64) {
        let P: &mut Triple = self.transition_mat.get_mut(&action).unwrap();
        // we know that this doesn't exist because we have guarded against it
        P.i.push(state as i32);
        P.j.push(sprime as i32);
        P.x.push(p);
    }

    fn insert_reward(
        &mut self, 
        sidx: usize, 
        action: i32, 
        rewards: Vec<f64>,
        nobjs: usize,
        size: usize
    ) {
        let R: &mut DenseMatrix = self.rewards_mat.get_mut(&action).unwrap();
        for c in 0..nobjs {
            R.m[c * size + sidx] = rewards[c];
        }
    }
}

fn process_mo_reward(
    rewards_map: &mut HashMap<(i32, usize), Vec<f64>>,
    q: i32,
    sidx: usize,
    task: &DFA,
    action: i32,
    nobjs: usize,
    agent_idx: usize,
    task_idx: usize
) {
    let mut rewards = vec![0.; nobjs];
    //let agent_reward = match mdp_rewards.get(&(s, action)) {
    //    Some(r) => { *r }
    //    None => { panic!("Could not find reward for state: {:?}, action: {}", (s, q), action) }
    //};
    if task.accepting.contains(&q) 
        || task.done.contains(&q)
        || task.rejecting.contains(&q) {
        rewards[agent_idx] = 0.
    } else {
        rewards[agent_idx] = -1.;
    }
    if task.accepting.contains(&q) {
        rewards[task_idx] = 1.;
    }
    rewards_map.insert((action, sidx), rewards);
}

pub fn build_model<S, E>(
    initial_state: (S, i32),
    //agent: &Agent,
    //mdp_rewards: &HashMap<(i32, i32), f64>,
    //transitions: &HashMap<(i32, u8), Vec<(i32, f64, String)>>,
    mdp: &mut E, // where E is a generic environment
    task: &DFA,
    agent_id: i32,
    task_id: i32,
    nagents: usize,
    nobjs: usize,
    actions: &[i32]
) -> MOProductMDP<S> 
where S: Copy + std::fmt::Debug + Hash + Eq, E: Env<S> {
    let pmdp: MOProductMDP<S> = product_mdp_bfs(
        &initial_state,
        //agent, 
        //mdp_rewards,
        mdp,
        //mdp_available_actions,
        task, 
        agent_id, 
        task_id, 
        nagents, 
        nobjs,
        actions
    );
    pmdp
}

fn product_mdp_bfs<S, E>(
    initial_state: &(S, i32),
    //agent_fpath: &str,
    //mdp: &Agent,
    //mdp_rewards: &HashMap<(i32, i32), f64>,
    mdp: &E, // MDP could be a trait
    //mdp_available_actions: &HashMap<i32, Vec<i32>>,
    task: &DFA,
    agent_id: i32, 
    task_id: i32,
    nagents: usize,
    nobjs: usize,
    actions: &[i32]
) -> MOProductMDP<S>
where S: Copy + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut visited: HashSet<(S, i32)> = HashSet::new();
    let mut transitions: HashMap<(i32, usize, usize), f64> = HashMap::new();
    let mut rewards: HashMap<(i32, usize), Vec<f64>> = HashMap::new();
    let mut stack: VecDeque<(S, i32)> = VecDeque::new();
    
    // get the actions from the env
    
    let mut product_mdp: MOProductMDP<S> = MOProductMDP::new(
        *initial_state, &actions[..], agent_id, task_id
    );

    // input the initial state into the back of the stack
    stack.push_back(*initial_state);
    visited.insert(*initial_state);
    product_mdp.insert_state(*initial_state);


    while !stack.is_empty(){
        // pop the front of the stack
        let (s, q) = stack.pop_front().unwrap();
        let sidx = *product_mdp.state_map.get(&(s, q)).unwrap();
        // for the new state
        // 1. has the new state already been visited
        // 2. If the new stat has not been visited then insert 
        // its successors into the stack
        for action in 0..actions.len() {
            // insert the mdp rewards
            let task_idx: usize = nagents + task_id as usize;
            process_mo_reward(
                &mut rewards, q, sidx, task, action as i32, nobjs, agent_id as usize, task_idx
            );

            //let v = mdp.step_(s, action as u8).unwrap();
            match mdp.step_(s, action as u8) {
                Ok(v) => {
                    if !v.is_empty() {
                        product_mdp.insert_action(action as i32);
                        product_mdp.insert_avail_act(&(s, q), action as i32);
                        for (sprime, p, w) in v.iter() { 
        
                            let qprime: i32 = task.get_transition(q, w);
                            if !visited.contains(&(*sprime, qprime)) {
                                visited.insert((*sprime, qprime));
                                stack.push_back((*sprime, qprime));
                                product_mdp.insert_state((*sprime, qprime));
                            }
                            let sprime_idx = *product_mdp.state_map.get(&(*sprime, qprime)).unwrap();
                            transitions.insert((action as i32, sidx, sprime_idx), *p);
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }
    // once the BFS is finished then we can convert the HashMaps transitions and rewards into
    // there corresponding matrices
    for ((action, sidx, sprime_idx), p) in transitions.drain() {
        product_mdp.insert_transition(sidx, action, sprime_idx, p);
    }
   
    for action in product_mdp.actions.iter() {
        let size = product_mdp.states.len() as i32;
        let P: &mut Triple = product_mdp.transition_mat.get_mut(action).unwrap();
        P.nc = size;
        P.nr = size;
        P.nzmax = P.x.len() as i32;
        P.nz = P.x.len() as i32;
        let R: &mut DenseMatrix = product_mdp.rewards_mat.get_mut(action).unwrap();
        R.m = vec![-f32::MAX as f64; size as usize * nobjs];
        R.cols = nobjs;
        R.rows = size as usize;
    }
    // the next three lines of code must come after sizing of the matrices above because we
    // need to initialise the dense matrix to -inf before we can start inserting values into it
    for ((action, sidx), r) in rewards.drain() {
        product_mdp.insert_reward(sidx, action, r, nobjs, product_mdp.states.len());
    }
    product_mdp.reverse_state_map = reverse_key_value_pairs(&product_mdp.state_map);
    product_mdp
}


#[pyclass]
/// SCPM is an unusual object
/// 
/// For now we will just include the team of agents and a mission
/// because we just want to look out how the channel works
pub struct SCPM {
    pub num_agents: usize,
    //pub agents: Team,
    pub tasks: Mission, 
    // todo: I don't think that we actually require a mission any more, because we will not be storing any data here
    // all of the transition information should be stored in the mdp environment script which will be called via GIL 
    pub actions: Vec<i32>
}

#[pymethods]
impl SCPM{ 
    #[new]
    fn new(mission: Mission, num_agents: usize, actions: Vec<i32>) -> Self {
        //let num_tasks = mission.size;
        SCPM {
            num_agents,
            tasks: mission,
            actions
        }
    }

    // function to construct the product models for each agent and each task
    // todo construct products should take a file reference to the environment that we wish to construct
}

impl SCPM {
    pub fn construct_products<S, E>(
        &self, 
        mdp: &mut E,
    ) -> Vec<MOProductMDP<S>>
    where S: Copy + std::fmt::Debug + Hash + Eq, E: Env<S> {

        // TODO: because memory is very cheap now, we can implement
        // multithreading to create the product MDP models
        let mut output: Vec<MOProductMDP<S>> = Vec::new();
        //let initial_state: (i32, i32) = (0, 0);
        let nobjs = self.num_agents + self.tasks.size;
        let nagents = self.num_agents;
        //for (i, agent) in self.agents.agents.iter().enumerate() {
        for i in 0..self.num_agents {
            //let mdp_rewards = agent.convert_rewards_to_map();
            //let mdp_transitions = agent.convert_transitions_to_map();
            //let mdp_available_actions = agent.available_actions_to_map();
            for (j, task) in self.tasks.tasks.iter().enumerate() {
                mdp.set_task(j);
                output.push(build_model(
                    (mdp.get_init_state(i), 0), 
                    //agent, 
                    //&mdp_rewards,
                    mdp,
                    //&mdp_available_actions,
                    task, 
                    i as i32, 
                    j as i32, 
                    nagents, 
                    nobjs,
                    &self.actions
                ));
            }
        }
        output
    }
}