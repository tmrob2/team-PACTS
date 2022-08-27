use hashbrown::{HashMap, HashSet};
use crate::scpm::model::{SCPM, MOProductMDP, GridState};
use crate::{Mantissa, blas_dot_product, val_or_zero_one, solver, new_target};
//use pyo3::prelude::*;
use crate::parallel::threaded::process_mdps;

//#[pyfunction]
//#[pyo3(name="para_test")]
pub fn process_scpm(
    model: &SCPM, 
    w: &[f64], 
    eps: &f64, 
    num_agents: usize,
    num_tasks: usize,
    prods: Vec<MOProductMDP>
) -> (Vec<f64>, Vec<MOProductMDP>, HashMap<(i32, i32), Vec<f64>>) {
    // Algorithm 1 will need to follow this format
    // where we move the ownership of the product models into a function 
    // which processes them and then return those models again for reprocessing
    let (prods, mut pis, result) = process_mdps(prods, &w[..], &eps, num_agents, num_tasks).unwrap();
    //println!("result: {:?}", result);
    // compute a rewards model
    let rewards_function = model.insert_rewards(result);
    let nobjs = model.agents.size + model.tasks.size;
    let blas_transition_matrices = model.grid.create_dense_transition_matrix(
        model.tasks.size
    );
    let blas_rewards_matrices = model.grid.create_dense_rewards_matrix(
        nobjs, 
        model.agents.size, 
        model.tasks.size,
        &rewards_function,
    );
    //model.print_rewards_matrices(&blas_rewards_matrices);
    let (alloc, r) = model.value_iteration(
        eps, 
        &w[..], 
        &blas_transition_matrices, 
        &blas_rewards_matrices
    );
    //println!("Alloc: {:?}", alloc);
    let task_allocation = alloc_dfs(model, alloc);
    //println!("task alloc: {:?}", task_allocation);
    retain_alloc_policies(&task_allocation[..], &mut pis);
    //println!("pis returned aftern retain: {:?}", pis);
    (r, prods, pis)
}

fn retain_alloc_policies(alloc: &[GridState], pis: &mut HashMap<(i32, i32), Vec<f64>>) {
    // alloc
    pis.retain(|&(a, t), _| alloc.contains(&GridState::new(a, t)));
}

pub fn alloc_dfs(model: &SCPM, policy: Vec<f64>) -> Vec<GridState> {
    // Use a stack to record the states we will see
    let mut stack: Vec<GridState> = Vec::new();
    // Use a visited vector to record the states seen
    let mut visited: HashSet<GridState> = HashSet::new();
    let mut allocation: HashSet<GridState> = HashSet::new();
    // initialise the stack
    stack.push(model.init_state);
    visited.insert(model.init_state);
    while !stack.is_empty() {
        // retrieve an item from the back of the stack (FIFO)
        let state = stack.pop().unwrap();
        if state.task < model.tasks.size as i32 { // need to account for the terminal state
            // for the given scheduler determine the transition of the popped state
            let sidx = model.grid.state_mapping.get(&state).unwrap();
            if policy[*sidx] == 0. {
                allocation.insert(state);
            }
            match model.grid.transitions.get(&(state, policy[*sidx] as i32)) {
                Some(sprime) => { 
                    if !visited.contains(sprime) {
                        stack.push(*sprime);
                        visited.insert(*sprime);
                    }
                }
                None => {  
                    //panic!("No transitions found for ({:?}, {})", state, policy[*sidx])
                }
            }
        }
    }
    allocation.into_iter().collect::<Vec<GridState>>()
}


pub fn scheduler_synthesis(model: &SCPM, w: &[f64], eps: &f64, t: &[f64], prods_: Vec<MOProductMDP>) 
-> (Vec<HashMap<(i32, i32), Vec<f64>>>, HashMap<usize, Vec<f64>>, Vec<f64>) {
    println!("initial w: {:.3?}", w);
    let mut tnew = t.to_vec();
    let mut hullset: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut hullset_X: Vec<Vec<f64>> = Vec::new();
    let mut temp_hullset_X;
    let mut weights: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut weights_v: Vec<Vec<f64>> = Vec::new();
    let mut temp_weights_: Vec<Vec<f64>>;
    let mut X: HashSet<Vec<Mantissa>> = HashSet::new();
    let mut W: HashSet<Vec<Mantissa>> = HashSet::new();
    let mut schedulers: Vec<HashMap<(i32, i32), Vec<f64>>> = Vec::new();
    let num_agents = model.agents.size;
    let num_tasks = model.tasks.size;
    let mut prods = prods_;
    // set an initial weight
    let mut w = vec![0.; num_agents];
    let mut w2 = vec![1. / num_tasks as f64; num_tasks];
    w.append(&mut w2);

    // compute the initial point for the random weight vector
    let (r, prods_, pis) = process_scpm(
        model, &w[..], &eps, num_agents, t.len(), prods 
    );

    // every single sheduler has been returned at this point, but we only need those which 
    // correspond to alloc

    schedulers.insert(0, pis);

    X.insert(r.iter().cloned().map(|f| Mantissa::new(f)).collect::<Vec<Mantissa>>());
    W.insert(w.iter().cloned().map(|f| Mantissa::new(f)).collect::<Vec<Mantissa>>());
    println!("r init: {:.3?}", r);
    let wrl = blas_dot_product(&r[..], &w[..]);
    let wt = blas_dot_product(&tnew[..], &w[..]);
    println!("wt: {:.3?}, wrl: {:.3?}, wrl < wt: {}", wt, wrl, wrl < wt);
    if wrl < wt {
        temp_hullset_X = hullset_X.clone();
        temp_hullset_X.insert(0, r.to_vec());
        temp_weights_ = weights_v.clone();
        temp_weights_.insert(0, w.to_vec());
        let mut tnew_found = false;
        let mut iterations = 1;
        while !tnew_found {
            let tnew_result = new_target(
                temp_hullset_X.to_vec(), 
                temp_weights_.to_vec(), 
                tnew.to_vec(), 
                1, 
                model.tasks.size,
                model.agents.size,
                iterations,
                5.,
                0.01
            );
            match tnew_result {
                Ok(x) => {
                    println!("tnew: {:.3?}", x);
                    tnew = x;
                    tnew_found = true;
                    //return (schedulers, hullset, x)
                }
                Err(_) => {
                    iterations += 1;
                }
            }
        }
    }

    hullset.insert(0, r.to_vec());
    hullset_X.push(r);
    weights.insert(0, w.to_vec());
    weights_v.push(w.to_vec());

    let mut lpvalid = true;
    let tot_objs = model.agents.size + model.tasks.size;

    let mut w: Vec<f64> = vec![0.; tot_objs];
    prods = prods_;
    let mut count: usize = 1;

    
    
    while lpvalid {
        
        let lpresult = solver(hullset_X.to_vec(), tnew.to_vec(), tot_objs);
        //println!("LP result: {:?}", lpresult);
        match lpresult {
            Ok(sol) => {
                for (ix, val) in sol.iter().enumerate() {
                    if ix < tot_objs {
                        println!(" w[{:?}] = {:.3}", ix, val);
                        w[ix] = val_or_zero_one(val);
                    }
                }
                let new_w = w
                    .iter()
                    .clone()
                    .map(|f| Mantissa::new(*f))
                    .collect::<Vec<Mantissa>>();
                match W.contains(&new_w) {
                    true => {
                        println!("All points discovered");
                        lpvalid = false;
                    }
                    false => { 
                        let (r, prods_, pis) = process_scpm(
                            model, &w[..], &eps, num_agents, t.len(), prods
                        );
                        //println!("pis {:?}", pis);
                        println!("r new: {:.2?}", r);
                        prods = prods_;
                        let wrl = blas_dot_product(&r[..], &w[..]);
                        let wt = blas_dot_product(&tnew[..], &w[..]);

                        println!("wt: {:.3?}, wrl: {:.3?}, wrl < wt: {}", wt, wrl, wrl < wt);
                        if wrl < wt {
                            temp_hullset_X = hullset_X.clone();
                            temp_hullset_X.insert(0, r.to_vec());
                            temp_weights_ = weights_v.clone();
                            temp_weights_.insert(0, w.to_vec());
                            let mut tnew_found = false;
                            let mut iterations = 1;
                            while !tnew_found {
                                let tnew_result = new_target(
                                    temp_hullset_X.to_vec(), 
                                    temp_weights_.to_vec(), 
                                    tnew.to_vec(), 
                                    1, 
                                    model.tasks.size,
                                    model.agents.size,
                                    iterations,
                                    5.,
                                    0.01
                                );
                                match tnew_result {
                                    Ok(x) => {
                                        println!("tnew: {:?}", x);
                                        tnew = x;
                                        tnew_found = true;
                                        //return (schedulers, hullset, x)
                                    }
                                    Err(_) => {
                                        iterations += 1;
                                    }
                                }
                            }
                        } else {
                            schedulers.insert(count, pis);
                            hullset.insert(count, r.to_vec());
                            hullset_X.push(r);
                            W.insert(new_w);
                            weights.insert(count, w.to_vec());
                            weights_v.push(w.to_vec());
                            count += 1;
                        }
                    }
                }
            }
            Err(e) => {
                // the LP has finished and there are no more points which can be added to the
                // the polytope
                lpvalid = false;
            }
        }
    }
    (schedulers, hullset, tnew)
}