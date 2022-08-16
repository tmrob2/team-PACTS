use hashbrown::HashMap;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct DFA {
    #[pyo3(get)]
    pub states: Vec<i32>,
    #[pyo3(get)]
    pub initial_state: i32,
    #[pyo3(get)]
    pub accepting: Vec<i32>,
    #[pyo3(get)]
    pub rejecting: Vec<i32>,
    #[pyo3(get)]
    pub done: Vec<i32>,
    pub transitions: HashMap<(i32, String), i32>
}

#[pymethods]
impl DFA {
    #[new]
    fn new(states: Vec<i32>, initial_state: i32, accepting: Vec<i32>, rejecting: Vec<i32>, done: Vec<i32>) -> Self {
        DFA{states, initial_state, accepting, rejecting, done, transitions: HashMap::new()}
    }

    fn add_transition(&mut self, q: i32, w: String, qprime: i32) {
        self.transitions.insert((q, w), qprime);
    }

    pub fn get_transitions(&self, q: i32, w: String) -> i32 {
        *self.transitions.get(&(q, w)).unwrap()
    }

    fn clone(&self) -> Self {
        DFA {
            states: self.states.to_vec(), 
            initial_state: self.initial_state, 
            accepting: self.accepting.to_vec(), 
            rejecting: self.rejecting.to_vec(), 
            done: self.done.to_vec(), 
            transitions: self.transitions.clone()
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Mission {
    pub tasks: Vec<DFA>,
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