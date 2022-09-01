use std::sync::mpsc::{channel};
use threadpool::ThreadPool;
use crate::scpm::model::MOProductMDP;
use crate::blas_dot_product;
use hashbrown::HashMap;
//use rand::prelude::*;
//use std::thread;
use crate::algorithm::dp::value_iteration;
//use std::time::{Instant};


const NUM_THREADS: usize = 4;

pub fn process_mdps(
    mdps: Vec<MOProductMDP>, 
    w: &[f64], 
    eps: &f64,
    num_agents: usize,
    num_tasks: usize
) -> Result<(
        Vec<MOProductMDP>, 
        HashMap<(i32, i32), Vec<f64>>,
        HashMap<(i32, i32), Vec<f64>>,
        HashMap<i32, Vec<(i32,f64)>>,
    ), 
    Box<dyn std::error::Error>> {

    //let t1 = Instant::now();
    let mut mdp_return_vec: Vec<MOProductMDP> = Vec::new();
    let mut result: HashMap<i32, Vec<(i32,f64)>> = HashMap::new();
    let mut alloc_map: HashMap<(i32, i32), Vec<f64>> = HashMap::new();
    let mut pis: HashMap<(i32, i32), Vec<f64>> = HashMap::new();
    let pool = ThreadPool::new(NUM_THREADS);
    let mdp_count: usize = mdps.len();

    let (tx, rx) = channel();
    let eps_copy = *eps;
    for mdp in mdps.into_iter() {
        let tx = tx.clone();
        //let w_k_ = vec![w[mdp.agent_id as usize], w[num_agents + mdp.task_id as usize]];
        let w_ = w.to_vec();
        pool.execute(move || {
            let (mdp, pi, r) = compute_value(mdp, w_, eps_copy, num_agents, num_tasks);
            tx.send((mdp, pi, r)).expect("Could not send data!");
        });
    }

    for _ in 0..mdp_count {
        let (mdp, pi, r) = rx.recv()?;
        alloc_map.insert((mdp.agent_id, mdp.task_id), r.to_vec());
        // convert r to the complete multi-objective r
        let mut mo_exp_cost = vec![0.; num_tasks + num_agents];
        mo_exp_cost[mdp.agent_id as usize] = r[0];
        mo_exp_cost[num_tasks + mdp.task_id as usize] = r[1];
        let exp_w_tot_cost = blas_dot_product(&mo_exp_cost[..], w);
        // then multiply by w to get to 
        match result.get_mut(&mdp.task_id) {
            Some(v) => {
                v.push((mdp.agent_id, exp_w_tot_cost));
            }
            None => {
                result.insert(mdp.task_id, vec![(mdp.agent_id, exp_w_tot_cost)]);
            }
        }
        pis.insert((mdp.agent_id, mdp.task_id), pi);
        mdp_return_vec.push(mdp);
    }

    //println!("allocation hashmap: \n{:?}", alloc_map);
    
    //println!("time: {:?}, Result: {:.2?} \n {:?}", t1.elapsed().as_secs_f64(), result, pis);
    Ok((mdp_return_vec, pis, alloc_map, result))
}

/// This function could be turned into value iteration
fn compute_value(mdp: MOProductMDP, w: Vec<f64>, eps: f64, nagents: usize, ntasks: usize) 
    -> (MOProductMDP, Vec<f64>, Vec<f64>) {
    let (pi, r) = value_iteration(&mdp, &w[..], &eps, nagents, ntasks);
    //println!("mdp({},{}) -> {:.3?}", mdp.agent_id, mdp.task_id, r);
    (mdp, pi, r)
}