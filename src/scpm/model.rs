use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use crate::agent::agent::{Agent, Team};
use crate::dfa::dfa::{DFA, Mission};
use std::collections::VecDeque;


#[pyclass]
pub struct MOProductMDP {
    pub initial_state: (i32, i32),
    pub states: Vec<(i32, i32)>,
    pub actions: Vec<i32>,
    pub rewards: HashMap<((i32, i32), i32), Vec<f64>>,
    pub transitions: HashMap<((i32, i32), i32), Vec<((i32, i32), f64)>>,
    pub agent_id: i32,
    pub task_id: i32,
    action_map: HashMap<(i32, i32), Vec<i32>>,
    state_map: HashMap<(i32, i32), usize>,
    reverse_state_map: HashMap<usize, (i32, i32)>
}

#[pymethods]
impl MOProductMDP {
    pub fn print_transitions(&self) {
        for state in self.states.iter() {
            for action in self.actions.iter() {
                match self.transitions.get(&(*state, *action)) {
                    Some(v) => {
                        println!("[{:?}, {}] => [{:?}]", state, action, v);
                    }
                    None => { }
                }
            }
        }
    }

    pub fn print_rewards(&self) {
        for state in self.states.iter() {
            for action in self.actions.iter() {
                match self.rewards.get(&(*state, *action)) {
                    Some(r) => { 
                        println!("[{:?}, {}] => {:?}", state, action, r);
                    }
                    None => { }
                }
            }
        }
    }
}

impl MOProductMDP {
    pub fn new(initial_state: (i32, i32), actions: &[i32], agent_id: i32, task_id: i32) -> Self {
        MOProductMDP {
            initial_state,
            states: Vec::new(),
            actions: actions.to_vec(),
            rewards: HashMap::new(),
            transitions: HashMap::new(),
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

    fn insert_state_mapping(&mut self, state: (i32, i32), state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }

    fn insert_action(&mut self, action: i32) {
        if !self.actions.contains(&action) {
            self.actions.push(action);
        }
    }

    fn insert_transition(&mut self, sprime: ((i32, i32), f64), state: (i32, i32), action: i32) {
        match self.transitions.get_mut(&(state, action)) {
            Some(t) => {
                t.push(sprime);
            }
            None => {
                self.transitions.insert((state, action), vec![sprime]);
            }
        }
    }

    fn insert_reward(&mut self, state: (i32, i32), mdp: &Agent, task: &DFA, action: i32) {
        let mut rewards = vec![0.; 2];
        let agent_reward = match mdp.rewards.get(&(state.0, action)) {
            Some(r) => { *r }
            None => { panic!("Could not find reward for state: {:?}, action: {}", state, action) }
        };
        if task.accepting.contains(&state.1) 
            || task.done.contains(&state.1)
            || task.rejecting.contains(&state.1) {
            rewards[0] = 0.
        } else {
            rewards[0] = agent_reward;
        }
        if task.accepting.contains(&state.1) {
            rewards[1] = 1.;
        }
        self.rewards.insert((state, action), rewards);
    }
}

#[pyfunction]
pub fn build_model(
    initial_state: (i32, i32),
    agent: &Agent,
    task: &DFA,
    agent_id: i32,
    task_id: i32
) -> MOProductMDP {
    let mdp: MOProductMDP = product_mdp_bfs(
        &initial_state, agent, task, agent_id, task_id
    );
    mdp
}

fn product_mdp_bfs(
    initial_state: &(i32, i32),
    mdp: &Agent,
    task: &DFA,
    agent_id: i32, 
    task_id: i32 
) -> MOProductMDP {
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut stack: VecDeque<(i32, i32)> = VecDeque::new();
    let mut product_mdp: MOProductMDP = MOProductMDP::new(
        *initial_state, &mdp.actions[..], agent_id, task_id
    );

    // input the initial state into the back of the stack
    stack.push_back(*initial_state);
    visited.insert(*initial_state);
    product_mdp.insert_state(*initial_state);

    while !stack.is_empty(){
        // pop the front of the stack
        let (s, q) = stack.pop_front().unwrap();
        // for the new state
        // 1. has the new state already been visited
        // 2. If the new stat has not been visited then insert 
        // its successors into the stack
        for action in mdp.available_actions.get(&s).unwrap() {
            product_mdp.insert_action(*action);
            product_mdp.insert_avail_act(&(s, q), *action);
            // insert the mdp rewards
            product_mdp.insert_reward((s, q), mdp, task, *action);
            for (sprime, p, w) in mdp.transitions.get(&(s, *action)).unwrap().iter() {
                let qprime: i32 = task.get_transitions(q, w.to_string());
                if !visited.contains(&(*sprime, qprime)) {
                    visited.insert((*sprime, qprime));
                    stack.push_back((*sprime, qprime));
                    product_mdp.insert_state((*sprime, qprime));
                }
                product_mdp.insert_transition(((*sprime, qprime), *p), (s, q), *action);
            }
        }
        
    }
    product_mdp
}


#[pyclass]
/// SCPM is an unsual object
/// 
/// For now we will just include the team of agents and a mission
/// because we just want to look out how the channel works
pub struct SCPM {
    pub agents: Team,
    pub tasks: Mission
}

#[pymethods]
impl SCPM { 
    #[new]
    fn new(agents: Team, mission: Mission) -> Self {
        SCPM {
            agents,
            tasks: mission
        }
    }
}

impl SCPM {
    // function to construct the product models for each agent and each task
    pub fn construct_products(&self) -> Vec<MOProductMDP> {
        let mut output: Vec<MOProductMDP> = Vec::new();
        let initial_state: (i32, i32) = (0, 0);
        for (i, agent) in self.agents.agents.iter().enumerate() {
            for (j, task) in self.tasks.tasks.iter().enumerate() {
                output.push(build_model(initial_state, agent, task, i as i32, j as i32));
            }
        }
        output
    }
}
