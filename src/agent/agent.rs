use hashbrown::HashMap;

// TODO whateve
pub trait Env<S> {
    fn step_(&self, s: S, action: u8) -> Result<Vec<(S, f64, String)>, String>;

    fn get_init_state(&self, agent: usize) -> S;

    fn set_task(&mut self, task_id: usize);

    fn get_action_space(&self) -> Vec<i32>;
}

// This is the code we already have for an MDP so we have to make this work
// but it should work fine because it already works in the current implementation

pub struct MDP<S, D> {
    pub states: Vec<i32>,
    pub init_state: i32,
    pub actions: Vec<u8>,
    pub available_actions: Vec<(i32, u8)>, // keep this
    pub state_map: HashMap<S, i32>,
    pub reverse_state_map: HashMap<i32, S>,
    pub extraData: D,
    pub size: usize,
}

impl<S: std::hash::Hash + Eq + Copy, D> MDP<S, D> {
    pub fn new(data: D, init_state: S, actions: Vec<u8>) -> Self {
        // convert the init state to an i32
        let istate = 0;
        let mut state_map: HashMap<S, i32> = HashMap::new();
        let mut reverse_state_map: HashMap<i32, S> = HashMap::new();
        state_map.insert(init_state, 0);
        reverse_state_map.insert(0, init_state);
        MDP {
            states: Vec::new(),
            init_state: istate,
            actions,
            available_actions: Vec::new(),
            state_map,
            reverse_state_map,
            extraData: data,
            size: 1
        }
    }

    pub fn get_or_mut_state(&mut self, state: &S) -> i32 
    where S: std::fmt::Debug {
        match self.state_map.get_mut(state) {
            Some(idx) => { return *idx }
            None => { 
                //println!("state: {:?} not in state map adding at index: {}", state, self.size);
                self.state_map.insert(*state, self.size as i32);
                self.reverse_state_map.insert(self.size as i32, *state);
                //for (k, v) in self.reverse_state_map.iter() {
                //    println!("{}, {:?}", k, v);
                //}
                let rtn_idx: i32 = self.size as i32;
                self.size += 1;
                return rtn_idx
            }
        }
    }
}