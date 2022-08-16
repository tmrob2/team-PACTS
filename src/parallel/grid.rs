use hashbrown::HashMap;
use crate::{DenseMatrix, reverse_key_value_pairs};

/// A grid is actually an MDP but it is also an abstract parallel object
/// for gathering values from sub-MDPs
/// 
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GridState {
    pub agent: i32, 
    pub task: i32
}

impl GridState {
    fn new(agent: i32, task: i32) -> Self {
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
    // there is a unique transition function in which the outcome
    // of taking some action is certain
    pub transitions: HashMap<(GridState, i32), GridState>,
    pub rewards: HashMap<(GridState, i32), Vec<f64>>,
    pub state_mapping: HashMap<GridState, usize>,
    pub reverse_state_mapping: HashMap<usize, GridState>
}

impl Grid {
    pub fn new(n_agents: usize, n_tasks: usize) -> Self {
        // the initial state will always be the first agent, and first task
        let initial = GridState::new(0, 0);
        let mut states: Vec<GridState> = Vec::new();
        let mut transitions: HashMap<(GridState, i32), GridState> = HashMap::new();
        let mut rewards: HashMap<(GridState, i32), Vec<f64>> = HashMap::new();
        let mut state_map: HashMap<GridState, usize> = HashMap::new();
        let actions = [0, 1];
        let mut state_count: usize = 0;
        for i in 0..n_agents {
            for j in 0..n_tasks {
                states.push(GridState::new(i as i32, j as i32));
                state_map.insert(GridState::new(i as i32, j as i32), state_count);
                state_count += 1;
            }
        }

        let reverse_state_map = reverse_key_value_pairs(&state_map);

        for state in states.iter() {
            for a in actions.iter() {
                // action 0 means that the current i,j processes the task 0, j + 1
                // action 1 means that the next agent i + 1, j processes the task
                // if the j is < the last j |J| then this is possible otherwise 0, j + 1
                if *a == 0 {
                    if state.task < n_tasks as i32 - 1 {
                        // let the agents decide who does the next task
                        transitions.insert((*state, *a), GridState::new(0, state.task + 1));
                        rewards.insert((*state, *a), vec![0.; n_agents + n_tasks]);
                    }
                } else {
                    if state.agent < n_agents as i32 - 1 {
                        transitions.insert((*state, *a), GridState::new(state.agent + 1, state.task));
                        rewards.insert((*state, *a), vec![0.; n_agents + n_tasks]);
                    }
                }
            }
        }

        Grid {
            initial_state: initial,
            states,
            actions,
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
    pub fn create_dense_transition_matrix(&self) -> HashMap<i32, DenseMatrix> {
        let size = self.states.len();
        let mut result: HashMap<i32, DenseMatrix> = HashMap::new();
        //
        for action in self.actions.iter() {
            // initialise a dense matrix col major format. 
            let mut m: Vec<f64> = vec![0.; (size - 1) * size];
            for state in self.states.iter() {
                match self.transitions.get(&(*state, *action)) {
                    Some(sprime) => { 
                        let sidx = self.state_mapping.get(&state).unwrap();
                        let sprimeidx = self.state_mapping.get(&sprime).unwrap();
                        m[sprimeidx * size + sidx] = 1.;
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

    pub fn insert_rewards(&mut self, agentidx: i32, taskidx: i32, nagents: usize, action: i32, v: &[f64]) {
        let tmp_state = GridState::new(agentidx, taskidx);
        match self.rewards.get_mut(&(tmp_state, action)) {
            Some(r) => { 
                r[agentidx as usize] = v[0];
                r[nagents + taskidx as usize] = v[1];
            }
            None => { panic!("Reward mapping not found for state: {:?}", tmp_state)}
        }
    }


    /// Constructs a Dense multi-objective rewards matrix for each action
    pub fn create_dense_rewards_matrix(&self, nobjs: usize, nagents: usize) -> HashMap<i32, DenseMatrix> {
        let size = self.states.len();
        let mut result: HashMap<i32, DenseMatrix> = HashMap::new();
        //
        for action in self.actions.iter() {
            let mut m: Vec<f64> = vec![0.; (size - 1) * nobjs];
            for state in self.states.iter() {
                match self.rewards.get(&(*state, *action)) {
                    Some(r) => {
                        let sidx = self.state_mapping.get(&state).unwrap();
                        let cagent_idx = state.agent as usize;
                        let ctask_idx = nagents + state.task as usize;
                        m[cagent_idx * size + sidx] = r[cagent_idx];
                        m[ctask_idx * size + sidx] = r[ctask_idx];
                    }
                    None => { }
                }
            } 
            let d = DenseMatrix {
                m,
                rows: size,
                cols: nobjs
            };
            result.insert(*action, d);
        }
        result
    }
}