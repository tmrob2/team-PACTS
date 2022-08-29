#![allow(non_snake_case)]

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use crate::agent::agent::{Agent, Team};
use crate::dfa::dfa::{DFA, Mission};
use std::collections::VecDeque;
use crate::*;
use crate::algorithm::dp::value_for_init_policy_dense;
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
    s: i32,
    q: i32,
    sidx: usize,
    mdp: &Agent,
    task: &DFA,
    action: i32,
    nobjs: usize,
    agent_idx: usize,
    task_idx: usize
) {
    let mut rewards = vec![0.; nobjs];
    let agent_reward = match mdp.rewards.get(&(s, action)) {
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

#[pyfunction]
pub fn build_model(
    initial_state: (i32, i32),
    agent: &Agent,
    task: &DFA,
    agent_id: i32,
    task_id: i32,
    nagents: usize,
    nobjs: usize
) -> MOProductMDP {
    let mdp: MOProductMDP = product_mdp_bfs(
        &initial_state, agent, task, agent_id, task_id, nagents, nobjs
    );
    mdp
}

fn product_mdp_bfs(
    initial_state: &(i32, i32),
    mdp: &Agent,
    task: &DFA,
    agent_id: i32, 
    task_id: i32,
    nagents: usize,
    nobjs: usize,
) -> MOProductMDP {
    let mut visited: HashSet<(i32, i32)> = HashSet::new();
    let mut transitions: HashMap<(i32, usize, usize), f64> = HashMap::new();
    let mut rewards: HashMap<(i32, usize), Vec<f64>> = HashMap::new();
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
        let sidx = *product_mdp.state_map.get(&(s, q)).unwrap();
        // for the new state
        // 1. has the new state already been visited
        // 2. If the new stat has not been visited then insert 
        // its successors into the stack
        for action in mdp.available_actions.get(&s).unwrap() {
            product_mdp.insert_action(*action);
            product_mdp.insert_avail_act(&(s, q), *action);
            // insert the mdp rewards
            let task_idx: usize = nagents + task_id as usize;
            process_mo_reward(
                &mut rewards, s, q, sidx, mdp, task, *action, nobjs, agent_id as usize, task_idx
            );
            for (sprime, p, w) in mdp.transitions.get(&(s, *action)).unwrap().iter() {
                // add sprime to state map if it doesn't already exist
                let qprime: i32 = task.get_transitions(q, w.to_string());
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
            for (j, task) in self.tasks.tasks.iter().enumerate() {
                output.push(build_model(initial_state, agent, task, i as i32, j as i32, nagents, nobjs));
            }
        }
        output
    }

    pub fn print_transitions_matrices(
        &self, 
        blas_transition_matrices: &HashMap<i32, DenseMatrix>
    ) {
        for action in self.grid.actions.iter() {
            let m = blas_transition_matrices.get(action).unwrap();
            for r in 0..m.rows + 1 {
                for c in 0..m.cols + 1 {
                    if r == 0 {
                        if c == 0 {
                            print!("{0:width$}", "",width=5);
                        } else {
                            let g = self.grid.states[c - 1];
                            print!("[{},{}]", g.agent, g.task)
                        }
                    } else {
                        if c == 0 {
                            let g = self.grid.states[r - 1];
                            print!("[{},{}]", g.agent, g.task)
                        } else {
                            print!("{:width$}", m.m[(c-1) * m.rows + (r-1)], width=5)
                        }
                    }
                }
                println!();
            }
        }
    }

    pub fn print_rewards_matrices(
        &self, 
        blas_rewards_matrices: &HashMap<i32, DenseMatrix>
    ) {
        for action in self.grid.actions.iter() {
            let m = blas_rewards_matrices.get(action).unwrap();
            for r in 0..m.rows + 1 {
                for c in 0..m.cols + 1 {
                    if r == 0 {
                        if c == 0 {
                            print!("{0:width$}", "",width=5);
                        } else {
                            print!("[o[{}]]", c-1);
                        }
                    } else {
                        if c == 0 {
                            let g = self.grid.states[r - 1];
                            print!("[{},{}]", g.agent, g.task)
                        } else {
                            let pval = if m.m[(c-1) * m.rows + (r-1)] == -f32::MAX as f64 {
                                f64::NEG_INFINITY
                            } else {
                                m.m[(c-1) * m.rows + (r-1)]
                            };
                            print!("{:.2} ", pval);
                        }
                    }
                }
                println!();
            }
        }
    }

    pub fn insert_rewards(
        &self, 
        rewards: HashMap<(i32, i32), Vec<f64>>
    ) -> HashMap<(GridState, i32), Vec<f64>> {
        // construct a new scpm Grid
        let mut scpm_rewards = self.grid.rewards.clone();
        for state in self.grid.states.iter() {
            match scpm_rewards.get_mut(&(*state, 0)) {
                Some(r) => {
                    let prod_rewards = rewards.get(&(state.agent, state.task)).unwrap();
                    r[state.agent as usize] = prod_rewards[0];
                    r[self.agents.size + state.task as usize] = prod_rewards[1];
                }
                None => { }
            }
        }
        scpm_rewards
    }

    pub fn value_iteration(
        &self, 
        eps: &f64, 
        w: &[f64],
        blas_transition_matrices: &HashMap<i32, DenseMatrix>,
        blas_rewards_matrices: &HashMap<i32, DenseMatrix>
    ) -> (Vec<f64>, Vec<f64>) {
        //let mut pi: Vec<f64> = vec![0.; self.grid.states.len() - 1];
        let mut r: Vec<f64> = vec![0.; self.agents.size + self.tasks.size];
        //let mut epsilon = 1.0;
        let size = self.grid.states.len();
        let nobjs = self.agents.size + self.tasks.size;

        // construct an initial random policy and compute the value of it
        //self.construct_init_policy(&mut pi[..]);
        // initialise some values
        let mut x = vec![0f64; size];
        let mut xnew = vec![0f64; size];
        let mut xtemp = vec![0f64; size];
        let mut q = vec![0f64; (size - 1) * self.grid.actions.len()];

        // initialise multi-objective values
        let mut X: Vec<f64> = vec![0.; size * nobjs];
        let mut Xnew: Vec<f64> = vec![0.; size * nobjs];
        let mut Xtemp: Vec<f64> = vec![0.; size * nobjs];

        let mut pi = self.grid.rand_proper_policy(self.tasks.size);
        //println!("initial policy: {:?}", pi);

        // todo: need to compute the value of the initial policy
        let Pinit = self.argmaxP(&pi[..], blas_transition_matrices);
        let Rv = self.argmaxRv(&pi[..], blas_rewards_matrices);
        //println!("initial Rv_pi: {:.3?}", Rv.m);

        value_for_init_policy_dense(&Rv.m[..], &mut x[..], eps, &Pinit);
        let mut pi_new = vec![-1f64; size - 1];
        //println!("initial value vector: {:.3?}", x);

        let mut policy_stable = false;
        while !policy_stable {
            policy_stable = true;
            // for each action compute the value vector
            for action in self.grid.actions.iter() {
                let mut vmv = vec![0f64; size - 1];
                // this is just a memory lookup so it is O(1)
                let Pa = blas_transition_matrices.get(action).unwrap();
                // compute P.v
                blas_matrix_vector_mulf64(&Pa.m[..], &x[..], Pa.rows as i32, Pa.cols as i32, &mut vmv[..]);
                // perform the operation R.w
                let mut rmv = vec![0f64; size - 1];
                let Ra = blas_rewards_matrices.get(action).unwrap();
                //println!("w: {:?}\n R:{:.3?}", w, Ra.m);
                blas_matrix_vector_mulf64(&Ra.m[..], &w[..], Ra.rows as i32, Ra.cols as i32, &mut rmv[..]);
                //println!("scpm R: {:.3?}", rmv);
                add_vecs(&rmv[..], &mut vmv[..], (size - 1) as i32, 1.0);
                update_qmat(&mut q[..], &vmv[..], *action as usize, self.grid.actions.len()).unwrap();
            }
            max_values(&mut xnew[..], &q[..], &mut pi_new[..], size - 1 , self.grid.actions.len());
            copy(&xnew[..], &mut xtemp[..], size as i32);
            add_vecs(&x[..], &mut xnew[..], size as i32, -1.0);
            update_policy(&xnew, &eps, &mut pi[..], &pi_new[..], size - 1, &mut policy_stable);
            //println!("updated policy: {:?}", pi);
            //epsilon = max_eps(&xnew[..]);
            copy(&xtemp[..], &mut x[..], size as i32);
            //println!("scpm x: {:?}", x);
        }

        //println!("w: {:?}\nStable policy: {:?}", w, pi);
        
        // construct the matrices based on the policy
        let P = self.argmaxP(&pi[..], blas_transition_matrices);
        let R = self.argmaxR(&pi[..], blas_rewards_matrices);

        // get the objective values
        let mut epsilon = 1.;
        let nobj_len = size * nobjs;
        while epsilon > *eps {
            for k in 0..nobjs {
                let mut vobjvec = vec![0f64; P.rows];
                blas_matrix_vector_mulf64(
                    &P.m[..], &X[k*size..(k + 1)*size], P.rows as i32, P.cols as i32, &mut vobjvec[..]
                );
                add_vecs(&R.m[k*R.rows..(k+1)*R.rows], &mut vobjvec[..], R.rows as i32, 1.0);
                copy(&vobjvec[..], &mut Xnew[k*size..(k + 1)*size], P.rows as i32);
            }

            // determine the difference between X, Xnew
            copy(&Xnew[..], &mut Xtemp[..], nobj_len as i32);
            add_vecs(&Xnew[..], &mut X[..], nobj_len as i32, -1.0);
            epsilon = max_eps(&X[..]);
            copy(&Xtemp[..], &mut X[..], nobj_len as i32);
        }
        for k in 0..nobjs {
            r[k] = X[k * size]; // assumes that the initial state is 0
        }
        //println!("r: {:?}", r);
        (pi, r)
    }

    /// Given some policy what is the the corresponding transition matrix
    pub fn argmaxP(
        &self, 
        pi: &[f64],
        blas_transition_matrices: &HashMap<i32, DenseMatrix>
    ) -> DenseMatrix {
        // for the action in the given state get the vector
        let size = self.grid.states.len();
        let mut P: Vec<f64> = vec![0f64; (size - 1) * size];
        for r in 0..size - 1 {
            let action = pi[r] as i32;
            let row = blas_transition_matrices.get(&action).unwrap();
            for c in 0..size {
                P[c * (size -1) + r] = row.m[c * (size - 1) + r];
            }
        }
        DenseMatrix {
            m: P,
            cols: size,
            rows: size - 1
        }
    }

    pub fn argmaxR(
        &self, 
        pi: &[f64],
        blas_rewards_matrices: &HashMap<i32, DenseMatrix>
    ) -> DenseMatrix {
        let size = self.grid.states.len();
        let nobjs = self.agents.size + self.tasks.size;
        let mut R: Vec<f64> = vec![0f64; (size - 1) * nobjs];
        for r in 0..size - 1 {
            let action = pi[r] as i32;
            let row = blas_rewards_matrices.get(&action).unwrap();
            for c in 0..nobjs {
                R[c * (size-1) + r] = row.m[c * (size-1) + r];
            }
        }
        DenseMatrix {
            m: R,
            cols: size,
            rows: size - 1
        }
    }

    pub fn argmaxRv(
        &self, 
        pi: &[f64],
        blas_rewards_matrices: &HashMap<i32, DenseMatrix>
    ) -> DenseMatrix {
        let size = self.grid.states.len();
        let mut R: Vec<f64> = vec![0f64; size - 1];
        for r in 0..size - 1 {
            let action = pi[r] as i32;
            let row = blas_rewards_matrices.get(&action).unwrap();
            let state = self.grid.reverse_state_mapping.get(&r).unwrap();
            let agent_idx = state.agent;
            R[r] = row.m[agent_idx as usize * (size-1) + r];
        }
        DenseMatrix {
            m: R,
            cols: size,
            rows: size - 1
        }
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
