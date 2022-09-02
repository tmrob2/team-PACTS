#![allow(non_snake_case)]

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use crate::agent::agent::{Agent, Team};
use crate::dfa::dfa::{DFA, Mission};
use std::collections::VecDeque;
use crate::*;
use rand::seq::SliceRandom;


#[pyclass]
pub struct MOProductMDP {
    pub initial_state: (i32, i32),
    pub states: Vec<(i32, i32)>,
    pub actions: Vec<i32>,
    //pub rewards: HashMap<((i32, i32), i32), Vec<f64>>,
    //pub transitions: HashMap<((i32, i32), i32), Vec<((i32, i32), f64)>>,
    pub transition_mat: HashMap<i32, Triple>,
    pub rewards_mat: HashMap<i32, DenseMatrix>,
    pub agent_id: i32,
    pub task_id: i32,
    action_map: HashMap<(i32, i32), Vec<i32>>,
    state_map: HashMap<(i32, i32), usize>,
    reverse_state_map: HashMap<usize, (i32, i32)>
}

#[pymethods]
impl MOProductMDP {
    pub fn print_transitions(&self) {
        // todo convert matrix to cs_di and print
    }

    pub fn print_rewards(&self) {
        // todo convert matrix to cs_di and print
    }
}

impl MOProductMDP {
    pub fn new(initial_state: (i32, i32), actions: &[i32], agent_id: i32, task_id: i32) -> Self {
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

    fn insert_state(&mut self, state: (i32, i32)) {
        let state_idx = self.states.len();
        self.states.push(state);
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_avail_act(&mut self, state: &(i32, i32), action: i32) {
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

    pub fn get_available_actions(&self, state: &(i32, i32)) -> &[i32] {
        &self.action_map.get(state).unwrap()[..]
    }

    pub fn get_state_map(&self) -> &HashMap<(i32, i32), usize> {
        &self.state_map
    }

    pub fn get_reverse_state_map(&self) -> &HashMap<usize, (i32, i32)> {
        &self.reverse_state_map
    }

    fn insert_state_mapping(&mut self, state: (i32, i32), state_idx: usize) {
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
    mdp_rewards: &HashMap<(i32, i32), f64>,
    s: i32,
    q: i32,
    sidx: usize,
    task: &DFA,
    action: i32,
    nobjs: usize,
    agent_idx: usize,
    task_idx: usize
) {
    let mut rewards = vec![0.; nobjs];
    let agent_reward = match mdp_rewards.get(&(s, action)) {
        Some(r) => { *r }
        None => { panic!("Could not find reward for state: {:?}, action: {}", (s, q), action) }
    };
    if task.accepting.contains(&q) 
        || task.done.contains(&q)
        || task.rejecting.contains(&q) {
        rewards[agent_idx] = 0.
    } else {
        rewards[agent_idx] = agent_reward;
    }
    if task.accepting.contains(&q) {
        rewards[task_idx] = 1.;
    }
    rewards_map.insert((action, sidx), rewards);
}

pub fn build_model(
    initial_state: (i32, i32),
    agent: &Agent,
    mdp_rewards: &HashMap<(i32, i32), f64>,
    mdp_transitions: &HashMap<(i32, i32), Vec<(i32, f64, String)>>,
    mdp_available_actions: &HashMap<i32, Vec<i32>>,
    task: &DFA,
    agent_id: i32,
    task_id: i32,
    nagents: usize,
    nobjs: usize
) -> MOProductMDP {
    let mdp: MOProductMDP = product_mdp_bfs(
        &initial_state, 
        agent, 
        mdp_rewards,
        mdp_transitions,
        mdp_available_actions,
        task, 
        agent_id, 
        task_id, 
        nagents, 
        nobjs
    );
    mdp
}

fn product_mdp_bfs(
    initial_state: &(i32, i32),
    mdp: &Agent,
    mdp_rewards: &HashMap<(i32, i32), f64>, // this is a problem
    mdp_transitions: &HashMap<(i32, i32), Vec<(i32, f64, String)>>, // this is alos a problem
    mdp_available_actions: &HashMap<i32, Vec<i32>>, // this is also a problem: i.e. we wiull not know any of these things using an env as an mdp
    task: &DFA,
    agent_id: i32, 
    task_id: i32,
    nagents: usize,
    nobjs: usize
) -> MOProductMDP {
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut transitions: HashMap<(i32, usize, usize), f64> = HashMap::new();
    let mut rewards: HashMap<(i32, usize), Vec<f64>> = HashMap::new();
    let mut stack: VecDeque<(i32, i32)> = VecDeque::new();
    let mut product_mdp: MOProductMDP = MOProductMDP::new(
        *initial_state, &mdp.actions[..], agent_id, task_id
    );

    let dfa_transitions = task.construct_transition_hashmap();

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
        for action in mdp_available_actions.get(&s).unwrap() {
            product_mdp.insert_action(*action);
            product_mdp.insert_avail_act(&(s, q), *action);
            // insert the mdp rewards
            let task_idx: usize = nagents + task_id as usize;
            process_mo_reward(
                &mut rewards, mdp_rewards, s, q, sidx, task, *action, nobjs, agent_id as usize, task_idx
            );
            for (sprime, p, w) in mdp_transitions.get(&(s, *action)).unwrap().iter() {
                // add sprime to state map if it doesn't already exist
                let qprime: i32 = *dfa_transitions.get(&(q, w.to_string())).unwrap();
                if !visited.contains(&(*sprime, qprime)) {
                    visited.insert((*sprime, qprime));
                    stack.push_back((*sprime, qprime));
                    product_mdp.insert_state((*sprime, qprime));
                }
                let sprime_idx = *product_mdp.state_map.get(&(*sprime, qprime)).unwrap();
                transitions.insert((*action, sidx, sprime_idx), *p);
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
/// SCPM is an unsual object
/// 
/// For now we will just include the team of agents and a mission
/// because we just want to look out how the channel works
pub struct SCPM {
    pub init_state: GridState,
    pub agents: Team,
    pub tasks: Mission,
    pub grid: Grid,
}

#[pymethods]
impl SCPM { 
    #[new]
    fn new(agents: Team, mission: Mission) -> Self {
        let num_agent = agents.size;
        let num_tasks = mission.size;
        SCPM {
            init_state: GridState::new(0, 0),
            agents,
            tasks: mission,
            grid: Grid::new(num_agent, num_tasks),
        }
    }

    pub fn print_transitions(&self) {
        for state in self.grid.states.iter() {
            for action in self.grid.actions.iter() {
                match self.grid.transitions.get(&(*state, *action)) {
                    Some(t) => { println!("P[{:?}, {}] => {:.3?}", state, action, t); }
                    None => { }
                }
            }
        }
    }
}

impl SCPM {
    // function to construct the product models for each agent and each task
    pub fn construct_products(&self) -> Vec<MOProductMDP> {
        let mut output: Vec<MOProductMDP> = Vec::new();
        let initial_state: (i32, i32) = (0, 0);
        let nobjs = self.agents.size + self.tasks.size;
        let nagents = self.agents.size;
        for (i, agent) in self.agents.agents.iter().enumerate() {
            let mdp_rewards = agent.convert_rewards_to_map();
            let mdp_transitions = agent.convert_transitions_to_map();
            let mdp_available_actions = agent.available_actions_to_map();
            for (j, task) in self.tasks.tasks.iter().enumerate() {
                output.push(build_model(
                    initial_state, 
                    agent, 
                    &mdp_rewards,
                    &mdp_transitions,
                    &mdp_available_actions,
                    task, 
                    i as i32, 
                    j as i32, 
                    nagents, 
                    nobjs
                ));
            }
        }
        output
    }
}

/// A grid is actually an MDP but it is also an abstract parallel object
/// for gathering values from sub-MDPs
/// 
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GridState {
    pub agent: i32, 
    pub task: i32
}

impl GridState {
    pub fn new(agent: i32, task: i32) -> Self {
        GridState {
            agent,
            task
        }
    }
}

pub struct Grid {
    pub initial_state: GridState,
    pub states: Vec<GridState>,
    pub actions: [i32;2],
    available_actions: HashMap<GridState, Vec<i32>>,
    // there is a unique transition function in which the outcome
    // of taking some action is certain
    pub transitions: HashMap<(GridState, i32), GridState>,
    pub rewards: HashMap<(GridState, i32), Vec<f64>>,
    pub state_mapping: HashMap<GridState, usize>,
    pub reverse_state_mapping: HashMap<usize, GridState>,
}

impl Grid {
    pub fn new(n_agents: usize, n_tasks: usize) -> Self {
        // the initial state will always be the first agent, and first task
        let initial = GridState::new(0, 0);
        let mut states: Vec<GridState> = Vec::new();
        let mut transitions: HashMap<(GridState, i32), GridState> = HashMap::new();
        let mut rewards: HashMap<(GridState, i32), Vec<f64>> = HashMap::new();
        let mut state_map: HashMap<GridState, usize> = HashMap::new();
        let mut available_actions: HashMap<GridState, Vec<i32>> = HashMap::new();
        let actions = [0, 1];
        let mut state_count: usize = 0;
        for i in 0..n_agents {
            for j in 0..n_tasks {
                states.push(GridState::new(i as i32, j as i32));
                state_map.insert(GridState::new(i as i32, j as i32), state_count);
                state_count += 1;
            }
        }
        // Add a final terminal state which we can take to allow us to compute the rewards
        // of the system
        states.push(GridState::new(0, n_tasks as i32));
        state_map.insert(GridState::new(0, n_tasks as i32), state_count);

        let reverse_state_map = reverse_key_value_pairs(&state_map);

        for state in states.iter() {
            for a in actions.iter() {
                // action 0 means that the current i,j processes the task 0, j + 1
                // action 1 means that the next agent i + 1, j processes the task
                // if the j is < the last j |J| then this is possible otherwise 0, j + 1
                if *a == 0 {
                    if state.task < n_tasks as i32 {
                        // let the agents decide who does the next task
                        transitions.insert((*state, *a), GridState::new(0, state.task + 1));
                        rewards.insert((*state, *a), vec![0.; n_agents + n_tasks]);
                        match available_actions.get_mut(state) {
                            Some(v) => { 
                                if !v.contains(a) {
                                    v.push(*a); 
                                }
                            }
                            None => { available_actions.insert(*state, vec![*a]); }
                        }
                    }
                } else {
                    if state.agent < n_agents as i32 - 1 {
                        transitions.insert((*state, *a), GridState::new(state.agent + 1, state.task));
                        rewards.insert((*state, *a), vec![0.; n_agents + n_tasks]);
                        match available_actions.get_mut(state) {
                            Some(v) => { 
                                if !v.contains(a) {
                                    v.push(*a); 
                                }
                            }
                            None => { available_actions.insert(*state, vec![*a]); }
                        }
                    }
                }
            }
        }

        Grid {
            initial_state: initial,
            states,
            actions,
            available_actions,
            transitions,
            rewards,
            state_mapping: state_map,
            reverse_state_mapping: reverse_state_map
        }
    }

    /// Creates a Dense transition matrix per action for BLAS computation
    /// 
    /// The size of this matrix will always be relatively small and therefore
    /// BLAS is a better LinAlg method than Sparse BLAS computation. 
    /// 
    /// The shape of this matrix will be n + m - 1, n + m. This is because the 
    /// last state (corresponding to the last task and the last agent) will have
    /// no outgoing transitions. 
    /// 
    /// n: number of agents 
    /// m: number of tasks
    pub fn create_dense_transition_matrix(&self, num_tasks: usize) -> HashMap<i32, DenseMatrix> {
        let size = self.states.len();
        let mut result: HashMap<i32, DenseMatrix> = HashMap::new();
        //
        for action in self.actions.iter() {
            // initialise a dense matrix col major format. 
            let mut m: Vec<f64> = vec![0.; (size - 1) * size];
            for state in self.states.iter().filter(|g| g.task != num_tasks as i32) {
                match self.transitions.get(&(*state, *action)) {
                    Some(sprime) => { 
                        let sidx = self.state_mapping.get(&state).unwrap();
                        let sprimeidx = self.state_mapping.get(&sprime).unwrap();
                        m[sprimeidx * (size - 1) + sidx] = 1.;
                    }
                    None => { }
                }
            }
            let d = DenseMatrix {
                m,
                rows: size - 1,
                cols: size
            };
            result.insert(*action, d);
        }
        result
    }

    /// Constructs a Dense multi-objective rewards matrix for each action
    pub fn create_dense_rewards_matrix(
        &self, 
        nobjs: usize, 
        nagents: usize,
        ntasks: usize,
        rewards_fn: &HashMap<(GridState, i32), Vec<f64>>
    ) -> HashMap<i32, DenseMatrix> {
        let size = self.states.len();
        let mut result: HashMap<i32, DenseMatrix> = HashMap::new();
        //
        for action in self.actions.iter() {
            let mut m: Vec<f64> = vec![-f32::MAX as f64; (size - 1) * nobjs];
            for state in self.states.iter().filter(|g| g.task != ntasks as i32) {
                match rewards_fn.get(&(*state, *action)) {
                    Some(r) => {
                        // the action is activated so you should get zero reward
                        // for anything other then the agent task reward
                        let sidx = self.state_mapping.get(&state).unwrap();
                        for a in 0..nagents + ntasks {
                            m[a * (size - 1) + sidx] = 0.
                        }
                        let cagent_idx = state.agent as usize;
                        let ctask_idx = nagents + state.task as usize;
                        m[cagent_idx * (size - 1) + sidx] = r[cagent_idx];
                        m[ctask_idx * (size - 1) + sidx] = r[ctask_idx];
                    }
                    None => { }
                }
            } 
            let d = DenseMatrix {
                m,
                rows: size - 1,
                cols: nobjs
            };
            result.insert(*action, d);
        }
        return result;
    }

    pub fn insert_rewards(&mut self, agentidx: i32, taskidx: i32, nagents: usize, action: i32, v: &[f64]) {
        let tmp_state = GridState::new(agentidx, taskidx);
        match self.rewards.get_mut(&(tmp_state, action)) {
            Some(r) => { 
                r[agentidx as usize] = v[0];
                r[nagents + taskidx as usize] = v[1];
            }
            None => { panic!("Reward mapping not found for state: {:.3?}", tmp_state)}
        }
    }

    pub fn rand_proper_policy(&self, num_tasks: usize) -> Vec<f64> {
        let mut pi: Vec<f64> = vec![0.; self.states.len() - 1];
        for state in self.states.iter().filter(|s| s.task != num_tasks as i32) {
            let sidx = self.state_mapping.get(state).unwrap();
            // choose one of the available actions
            let avail_act = self.available_actions.get(state).unwrap();
            let act = avail_act.choose(&mut rand::thread_rng()).unwrap();
            pi[*sidx] = *act as f64;
        }
        pi
    }

    pub fn get_available_actions(&self) -> &HashMap<GridState, Vec<i32>> {
        &self.available_actions
    }
}
