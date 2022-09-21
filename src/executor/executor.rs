use hashbrown::HashMap;
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use crate::dfa::dfa::DFA;

//----------------------------------------------------------------------------
// MOTAP Solution
//----------------------------------------------------------------------------
pub struct Solution<S> {
    pub prod_state_maps: HashMap<(i32, i32), HashMap<(S, i32), usize>>,
    pub schedulers: HashMap<usize, HashMap<(i32, i32), Vec<f64>>>,
    pub weights: Vec<f64>,
    pub ntasks: usize,
    pub npoints: usize,
    pub agent_task_allocations: HashMap<i32, Vec<(i32, usize)>>
}

pub trait DefineSolution<S> {
    fn get_prod_state_maps(&self) -> &HashMap<(i32, i32), HashMap<(S, i32), usize>>;

    fn set_prod_state_maps(&mut self, map: HashMap<(S, i32), usize>, agent: i32, task: i32);

    fn return_prod_state_map(&mut self, agent: i32, task: i32) -> HashMap<(S, i32), usize>;

    fn get_schedulers(&self) -> &HashMap<usize, HashMap<(i32, i32), Vec<f64>>>;

    fn set_schedulers(&mut self, schedulers: HashMap<usize, HashMap<(i32, i32), Vec<f64>>>);

    fn remove_scheduler(&mut self, agent: i32, task: i32, k: usize) -> Vec<f64>;

    fn get_weights(&self) -> &[f64];

    fn set_weights(&mut self, weights: Vec<f64>, l: usize);

    fn get_npoints(&self) -> usize;

    fn set_npoints(&mut self, k: usize);

    fn get_ntasks(&self) -> usize;

    fn set_ntasks(&mut self, ntasks: usize);

    fn get_agent_task_allocations(&mut self) -> &mut HashMap<i32, Vec<(i32, usize)>>;

    fn get_task_allocation(&mut self, agent: i32) -> Option<Vec<(i32, usize)>>;

    fn new_(ntasks: usize) -> Self;

}

impl<S> DefineSolution<S> for Solution<S> {

    fn new_(ntasks: usize) -> Self {
        Solution { 
            prod_state_maps: HashMap::new(), 
            schedulers: HashMap::new(), 
            weights: Vec::new(), 
            ntasks, 
            npoints: 0, 
            agent_task_allocations: HashMap::new() 
        }
    }

    fn get_prod_state_maps(&self) -> &HashMap<(i32, i32), HashMap<(S, i32), usize>> {
        &self.prod_state_maps
    }

    fn set_prod_state_maps(&mut self, map: HashMap<(S, i32), usize>, agent: i32, task: i32) {
        self.prod_state_maps.insert((agent, task), map);
    }

    fn return_prod_state_map(&mut self, agent: i32, task: i32) -> HashMap<(S, i32), usize> {
        self.prod_state_maps.remove(&(agent, task)).unwrap()
    }

    fn get_schedulers(&self) -> &HashMap<usize, HashMap<(i32, i32), Vec<f64>>> {
        &self.schedulers
    }

    fn set_schedulers(&mut self, schedulers: HashMap<usize, HashMap<(i32, i32), Vec<f64>>>) {
        self.schedulers = schedulers;
    }

    fn remove_scheduler(&mut self, agent: i32, task: i32, k: usize) -> Vec<f64> {
        self.schedulers.get_mut(&k).unwrap().remove(&(agent, task)).unwrap()
    }

    fn get_weights(&self) -> &[f64] {
        &self.weights[..]
    }

    fn set_weights(&mut self, weights: Vec<f64>, l: usize) {
        self.weights = weights;
        self.set_npoints(l);
    }

    fn get_npoints(&self) -> usize {
        self.npoints
    }

    fn set_npoints(&mut self, k: usize) {
        self.npoints = k;
    }

    fn get_ntasks(&self) -> usize {
        self.ntasks
    }

    fn set_ntasks(&mut self, ntasks: usize) {
        self.ntasks = ntasks;
    }

    fn get_agent_task_allocations(&mut self) -> &mut HashMap<i32, Vec<(i32, usize)>> {
        &mut self.agent_task_allocations
    }

    fn get_task_allocation(&mut self, agent: i32) -> Option<Vec<(i32, usize)>> {
        self.agent_task_allocations.remove(&agent)
    }


}

impl<S> GenericSolutionFunctions<S> for Solution<S> 
where S: Copy + Eq + std::hash::Hash { }

pub trait GenericSolutionFunctions<S>: DefineSolution<S>
where S: Copy + std::hash::Hash + Eq {
    /// Gets an index of the given product state (S, q) for a particular (agent, task) pair
    fn get_index_(
        &self,
        state: S,
        q: i32, 
        agent: i32,
        task: i32
    ) -> usize {
        let map = 
            self.get_prod_state_maps().get(&(agent, task)).unwrap();
        *map.get(&(state, q)).unwrap()
    }

    /// Gets the action for the scheduler at state index: state_index for a particular
    /// (agent, task, point) triple, where point corresponds to a Pareto soluction on the 
    /// convex hull. 
    fn get_action_(&self, state_index: usize, agent: i32, task: i32, point: usize) -> i32 {
        let point_ = self.get_schedulers().get(&point).unwrap();
        point_.get(&(agent, task)).unwrap()[state_index] as i32
    }

    /// For a particular point (scheduler), task combination returns the agent for which the
    /// task was allocated to optimally
    fn get_allocation_(&self, point: usize, task: i32) -> Vec<i32> {
        let point_keys = 
            self.get_schedulers().get(&point).unwrap();
        let mut keys_: Vec<_> = Vec::new();
        for (i, _j) in point_keys.keys().filter(|(_a, t)| *t == task) {
            keys_.push(*i)
        }
        keys_
    }

    fn add_to_agent_task_queues(&mut self) {
        // with the weights (row major format) sample a scheduler
        // then determine which agent the task is allocated to
        // then add the task to the agents queue
        // we probably don't need to return anything and just store the 
        // task_queue in the outputs
        let ntasks = self.get_ntasks();
        let npoints = self.get_npoints();
        for task in 0..ntasks {
            let w_ = &self.get_weights()[task * npoints..(task+1) * npoints];
            // sample and argmax from this weight 
            // so we need an array of indexes
            let choices : Vec<usize> = (0..npoints).collect();
            let mut rng = thread_rng();
            let dist = WeightedIndex::new(w_).unwrap();
            // Scheduler: k
            let k = choices[dist.sample(&mut rng)];
            // For the scheduler chosen, return the allocated agent
            let agent = self.get_allocation_(k, task as i32)[0];
            match self.get_agent_task_allocations().get_mut(&agent) {
                Some(t) => { 
                    // The agent index already exists, so we add to current tasks
                    t.push((task as i32, k));
                }
                None => {
                    self.get_agent_task_allocations()
                        .insert(agent, vec![(task as i32, k)]);
                }
            }
        }
    }
}


//----------------------------------------------------------------------------
// MOTAP Solution
//----------------------------------------------------------------------------

pub trait DefineExecutor<S> {
    /// Function to define a new Executor
    fn new_(nagents: usize) -> Self;
    
    /// For a particular task-agent pari get the sampled state-map for the product
    /// MDP
    fn get_state_map(&self, agent: i32, task: i32) -> &HashMap<(S, i32), usize>;

    /// From the MOTAP solution insert the sampled state-map into the executor
    /// memory
    fn set_state_map(&mut self, agent: i32, task: i32, map: HashMap<(S, i32), usize>);

    fn remove_state_map(&mut self, agent: i32, task: i32);

    /// Get the sampled scheduler for a particular task agent pair
    /// 
    /// This function will be paired with the allocation sampler
    fn get_scheduler(&self, agent: i32, task: i32) -> &[f64];
    
    /// Take the scheduler from the MOTAP solutions and insert it into the 
    /// executor struct
    fn set_scheduler(&mut self, agent: i32, task: i32, scheduler: Vec<f64>);

    fn remove_scheduler(&mut self, agent: i32, task: i32);

    /// Notice here that the task allocation does not keed the scheduler number
    /// because we have already sampled the scheduler distribution. 
    fn get_agent_task_allocations(&self, agent: i32) -> &[i32];

    fn get_next_task(&mut self, agent: i32) -> Option<i32>;

    /// Set the sampled task-agent allocation
    fn set_agent_task_allocation(&mut self, agent: i32, task: i32);

    fn dfa_next_state_(&mut self, task: i32, q: i32, word: String) -> i32;

    fn insert_dfa(&mut self, dfa: DFA, task: i32);

    fn check_done_(&self, task: i32) -> u8;

    fn dfa_current_state_(&self, task: i32) -> i32;
}

pub trait Execution<S>: DefineExecutor<S> 
where S: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    fn add_alloc_to_execution(&mut self, solution: &mut Solution<S>, nagents: usize) {
        for agent in 0..nagents {
            let allocs = solution.get_task_allocation(agent as i32);
            match allocs {
                Some(v) => { 
                    // Add the tasks to the set of schedulers
                    for (task, k) in v.into_iter() {
                        // For the executor:
                        // 1. Setting the Prod State Map
                        let prod_state_map = 
                        solution.return_prod_state_map(agent as i32, task);
                        self.set_state_map(agent as i32, task, prod_state_map);
                        // 2. Setting the Scheduler from the solution
                        let scheduler = 
                        solution.remove_scheduler(agent as i32, task, k);
                        self.set_scheduler(agent as i32, task, scheduler);
                        // 3. Set the task allocation
                        self.set_agent_task_allocation(agent as i32, task);
                    }
                }
                None => { 
                    // Do nothing
                }
            }
        }
    }

    /// Function to get the next task from the task queue for the particular 
    /// agent
    fn get_next_task_(&mut self, agent: i32) -> Option<i32> {
        let task = self.get_next_task(agent);
        task
    }

    /// Clean up after task-agent execution, to save memory
    fn remove_completed_task_agent_(&mut self, agent: i32, task: i32) {
        self.remove_state_map(agent, task);
        self.remove_scheduler(agent, task);
    }

    fn get_action_(&self, agent: i32, task: i32, state: S, q: i32) -> i32 {
        //println!("state: {:?}, q: {}", state, q);
        let sidx = *self.get_state_map(agent, task).get(&(state, q)).expect(&format!("failed at {:?}", (state, q)));
        let action = self.get_scheduler(agent, task)[sidx] as i32;
        action
    }
}