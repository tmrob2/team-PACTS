use hashbrown::{HashMap, HashSet};
use crate::scpm::model::{SCPM, MOProductMDP, GridState};
use crate::{Mantissa, blas_dot_product, val_or_zero_one, solver};
use pyo3::prelude::*;
use crate::parallel::threaded::process_mdps;
//use crate::lp::lp_solver::{new_target, gurobi_solver};

//#[pyfunction]
//#[pyo3(name="para_test")]
pub fn process_scpm(
    model: &SCPM, 
    w: &[f64], 
    eps: &f64, 
    num_agents: usize,
    prods: Vec<MOProductMDP>
) -> (Vec<f64>, Vec<MOProductMDP>, HashMap<(i32, i32), Vec<f64>>) {
    // Algorithm 1 will need to follow this format
    // where we move the ownership of the product models into a function 
    // which processes them and then return those models again for reprocessing
    let (prods, mut pis, result) = process_mdps(prods, &w[..], &eps, num_agents).unwrap();
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
    let (alloc, r) = model.value_iteration(
        eps, 
        &w[..], 
        &blas_transition_matrices, 
        &blas_rewards_matrices
    );
    let task_allocation = alloc_dfs(model, alloc);
    retain_alloc_policies(&task_allocation[..], &mut pis);
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
                None => { panic!("No transitions found for ({:?}, {})", state, policy[*sidx])}
            }
        }
    }
    allocation.into_iter().collect::<Vec<GridState>>()
}


pub fn scheduler_synthesis(model: &SCPM, w: &[f64], eps: &f64, t: &[f64], prods_: Vec<MOProductMDP>) 
-> (Vec<HashMap<(i32, i32), Vec<f64>>>, HashMap<usize, Vec<f64>>, Vec<f64>) {

    let mut tnew = t.to_vec();
    let mut hullset: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut hullset_X: Vec<Vec<f64>> = Vec::new();
    let mut weights: HashMap<usize, Vec<f64>> = HashMap::new();
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
        model, &w[..], &eps, num_agents, prods
    );

    // every single sheduler has been returned at this point, but we only need those which 
    // correspond to alloc

    schedulers.insert(0, pis);

    X.insert(r.iter().cloned().map(|f| Mantissa::new(f)).collect::<Vec<Mantissa>>());
    W.insert(w.iter().cloned().map(|f| Mantissa::new(f)).collect::<Vec<Mantissa>>());

    let wrl = blas_dot_product(&r[..], &w[..]);
    let wt = blas_dot_product(&tnew[..], &w[..]);

    if wrl < wt {
        return (schedulers, hullset, tnew)
        //let mut temp_hullset = hullset.clone();
        //temp_hullset.insert(0, r.to_vec());
        //let mut temp_weights = weights.clone();
        //temp_weights.insert(0, w.to_vec());
        //let mut tnew_found = false;
        //let mut iterations = 1;
        //while !tnew_found {
        //    let tnew_result = new_target(
        //        &temp_hullset, 
        //        &temp_weights, 
        //        &tnew[..], 
        //        1, 
        //        model.tasks.size,
        //        model.agents.size,
        //        5.,
        //        0.01,
        //        iterations
        //    );
        //    match tnew_result {
        //        Ok(x) => {
        //            println!("tnew: {:?}", x);
        //            tnew = x;
        //            tnew_found = true;
        //            //return (schedulers, hullset, x)
        //        }
        //        Err(_) => {
        //            iterations += 1;
        //        }
        //    }
        //}
    }

    hullset.insert(0, r.to_vec());
    hullset_X.push(r);
    weights.insert(0, w.to_vec());

    let mut lpvalid = true;
    let tot_objs = model.agents.size + model.tasks.size;

    let mut w: Vec<f64> = vec![0.; tot_objs];
    prods = prods_;
    let mut count: usize = 1;

    let lpresult = solver(hullset_X.to_vec(), tnew.to_vec(), tot_objs);

    println!("LP result: {:?}", lpresult);

    //while lpvalid {
    //    
    //    //let gurobi_result = gurobi_solver(&hullset, &tnew[..], &tot_objs);
    //    match gurobi_result {
    //        Some(sol) => {
    //            for (ix, val) in sol.iter().enumerate() {
    //                if ix < tot_objs {
    //                    //println!(" w[{:?}] = {}", ix, val);
    //                    //println!(" w[{:?}] = {:.3}", ix, val);
    //                    w[ix] = val_or_zero_one(val);
    //                }
    //            }
    //            let new_w = w
    //                .iter()
    //                .clone()
    //                .map(|f| Mantissa::new(*f))
    //                .collect::<Vec<Mantissa>>();
    //            match W.contains(&new_w) {
    //                true => {
    //                    //println!("All points discovered");
    //                    lpvalid = false;
    //                }
    //                false => { 
    //                    
    //                    let (r, prods_, pis) = process_scpm(
    //                        model, &w[..], &eps, num_agents, prods
    //                    );
    //                    prods = prods_;
    //                    let wrl = blas_dot_product(&r[..], &w[..]);
    //                    let wt = blas_dot_product(&tnew[..], &w[..]);
    //                    if wrl < wt {
    //                        //println!("Ran in t(s): {:?}", t1.elapsed().as_secs_f64());
    //                        let mut temp_hullset = hullset.clone();
    //                        temp_hullset.insert(0, r.to_vec());
    //                        let mut temp_weights = weights.clone();
    //                        temp_weights.insert(0, w.to_vec());
    //                        let mut tnew_found = false;
    //                        let mut iterations = 1;
    //                        while !tnew_found {
    //                            let tnew_result = new_target(
    //                                &temp_hullset,
    //                                &temp_weights,
    //                                &t[..],
    //                                temp_hullset.len(),
    //                                model.tasks.size,
    //                                model.agents.size,
    //                                5.,
    //                                0.05,
    //                                iterations
    //                            );
    //                            match tnew_result {
    //                                Ok(x) => {
    //                                    println!("tnew: {:?}", x);
    //                                    tnew = x;
    //                                    tnew_found = true;
    //                                    //return (schedulers, hullset, x)
    //                                }
    //                                Err(_) => {
    //                                    iterations += 1;
    //                                }
    //                            }
    //                            if iterations > 100 {
    //                                panic!("Solution not possible");
    //                            }
    //                        }
    //                    }
    //                    schedulers.insert(count, pis);
    //                    hullset.insert(count, r);
    //                    W.insert(new_w);
    //                    weights.insert(count, w.to_vec());
    //                    count += 1;
    //                }
    //            }
    //        }
    //        None => {
    //            // the LP has finished and there are no more points which can be added to the
    //            // the polytope
    //            lpvalid = false;
    //        }
    //    }
    //}
    (schedulers, hullset, tnew)
}