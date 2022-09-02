use pyo3::prelude::*;
use hashbrown::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;

// This is the code we already have for an MDP so we have to make this work
// but it should work fine because it already works in the current implementation
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct Agent {
    #[pyo3(get)]
    pub states: Vec<i32>,
    pub name: String, 
    #[pyo3(get)]
    pub init_state: i32,
    pub transitions: Vec<((i32, i32), Vec<(i32, f64, String)>)>,
    pub rewards: Vec<((i32, i32), f64)>,
    pub actions: Vec<i32>,
    pub available_actions: Vec<(i32, i32)>
}

#[pymethods]
impl Agent {
    #[new]
    fn new(init_state: i32, states: Vec<i32>, actions: Vec<i32>) -> PyResult<Self> {
        Ok(Agent { 
            states, 
            name: "agent".to_string(),
            init_state, 
            transitions: Vec::new(), 
            rewards: Vec::new(), 
            actions,
            available_actions: Vec::new()
        })
    }

    fn add_transition(&mut self, state: i32, action: i32, sprimes: Vec<(i32, f64, String)>) {
        if !self.available_actions.contains(&(state, action)) {
            self.available_actions.push((state, action));
        }

        self.transitions.push(((state, action), sprimes));
    }

    fn print_transitions(&self) {
        for transition in self.transitions.iter() {
            println!("{:?}", transition);
        }
    }

    fn add_reward(&mut self, state: i32, action: i32, reward: f64) {
        self.rewards.push(((state, action), reward));
    }

    fn clone(&self) -> Self {
        Agent { 
            states: self.states.to_vec(),
            name: self.name.to_string(),
            init_state: self.init_state,
            transitions: self.transitions.clone(),
            rewards: self.rewards.clone(),
            actions: self.actions.to_vec(),
            available_actions: self.available_actions.clone()
        }
    }

    fn json_serialize_agent(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[pyfunction]
#[pyo3(name="agent_from_str")]
pub fn json_deserialize_from_string(s: String) -> Agent {
    let agent: Agent = serde_json::from_str(&s).unwrap();
    agent
}

impl Agent {
    pub fn convert_transitions_to_map(&self) -> HashMap<(i32, i32), Vec<(i32, f64, String)>> {
        let mut transitions: HashMap<(i32, i32), Vec<(i32, f64, String)>> = HashMap::new();
        for ((s, q), v) in self.transitions.iter() {
            transitions.insert((*s, *q), v.to_vec());
        } 
        transitions
    }

    pub fn convert_rewards_to_map(&self) -> HashMap<(i32, i32), f64> {
        let mut rewards: HashMap<(i32, i32), f64> = HashMap::new();
        for ((s, q), r) in self.rewards.iter() {
            rewards.insert((*s, *q), *r);
        }
        rewards
    }

    pub fn available_actions_to_map(&self) -> HashMap<i32, Vec<i32>> {
        let mut available_action_map: HashMap<i32, Vec<i32>> = HashMap::new();
        for (s, a) in self.available_actions.iter() {
            match available_action_map.get_mut(s) {
                Some(x) => { x.push(*a); }
                None => { available_action_map.insert(*s, vec![*a]); }
            }
        }
        available_action_map
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Team {
    pub agents: Vec<Agent>,
    #[pyo3(get)]
    pub size: usize
}

#[pymethods]
impl Team {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Team { agents: Vec::new(), size: 0})
    }

    fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
        self.size += 1;
    }

    fn print_initial_states(&self){
        for agent in self.agents.iter() {
            println!("init state: {:?}", agent.init_state);
        }
    }

    pub fn print_transitions(&self, agent: usize) {
        for transition in self.agents[agent].transitions.iter() {
            println!("{:?}", transition)
        }
    }

    pub fn get_agent(&self, agent_idx: usize) -> Agent {
        self.agents[agent_idx].clone()
    } 
}