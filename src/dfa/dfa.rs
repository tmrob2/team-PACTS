use hashbrown::HashMap;
use pyo3::prelude::*;
use serde::{Serialize, Deserialize};
use serde_json;

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct DFA {
    #[pyo3(get)]
    pub states: Vec<i32>,
    pub name: String,
    #[pyo3(get)]
    pub initial_state: i32,
    #[pyo3(get)]
    pub accepting: Vec<i32>,
    #[pyo3(get)]
    pub rejecting: Vec<i32>,
    #[pyo3(get)]
    pub done: Vec<i32>,
    pub transitions: Vec<(i32, String, i32)>
}

#[pymethods]
impl DFA {
    #[new]
    fn new(states: Vec<i32>, initial_state: i32, accepting: Vec<i32>, rejecting: Vec<i32>, done: Vec<i32>) -> Self {
        DFA{
            states, 
            name: "task".to_string(),
            initial_state, 
            accepting, 
            rejecting, 
            done, 
            transitions: Vec::new()}
    }

    fn add_transition(&mut self, q: i32, w: String, qprime: i32) {
        if !self.transitions.contains(&(q, w.to_string(), qprime)) {
            self.transitions.push((q, w, qprime));
        }
    }

    fn clone(&self) -> Self {
        DFA {
            states: self.states.to_vec(), 
            name: self.name.to_string(),
            initial_state: self.initial_state, 
            accepting: self.accepting.to_vec(), 
            rejecting: self.rejecting.to_vec(), 
            done: self.done.to_vec(), 
            transitions: self.transitions.clone()
        }
    }

    fn print_transitions(&self, words: Vec<String>) {
        println!("{:?}", self.transitions);
    }

    fn json_serialize_dfa(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[pyfunction]
#[pyo3(name="task_from_str")]
pub fn json_deserialize_from_string(s: String) -> DFA {
    let task: DFA = serde_json::from_str(&s).unwrap();
    task
}

impl DFA {
    pub fn construct_transition_hashmap(&self) -> HashMap<(i32, String), i32> {
        let mut map: HashMap<(i32, String), i32> = HashMap::new();
        for (q, w, qprime) in self.transitions.iter() {
            map.insert((*q, w.to_string()), *qprime);
        }
        map
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Mission {
    pub tasks: Vec<DFA>,
    #[pyo3(get)]
    pub size: usize
}

#[pymethods]
impl Mission {
    #[new]
    fn new() -> Self {
        Mission {
            tasks: Vec::new(),
            size: 0
        }
    }

    fn add_task(&mut self, dfa: DFA) {
        self.tasks.push(dfa);
        self.size += 1;
    }

    fn print_task_transitions(&self) {
        for task in self.tasks.iter() {
            println!("|P|: {:?}", task.transitions.len());
        }
    }

    fn get_task(&self, task_idx: usize) -> DFA {
        self.tasks[task_idx].clone()
    }
}