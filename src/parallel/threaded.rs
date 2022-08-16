use std::sync::mpsc::{channel, RecvError};
use threadpool::ThreadPool;
use crate::scpm::model::MOProductMDP;
use hashbrown::HashMap;
use rand::prelude::*;
use std::thread;
use std::time::{Instant};


const NUM_THREADS: usize = 4;

pub fn process_mdps(mdps: Vec<MOProductMDP>) -> Result<(), Box<dyn std::error::Error>> {

    let t1 = Instant::now();

    let mut result: HashMap<(i32, i32), Vec<f64>> = HashMap::new();
    let pool = ThreadPool::new(NUM_THREADS);
    let mdp_count: usize = mdps.len();

    let (tx, rx) = channel();
    for mdp in mdps.into_iter() {
        let tx = tx.clone();
        pool.execute(move || {
            let (i, j, vect) = compute_value(mdp);
            tx.send((i, j, vect)).expect("Could not send data!");
        });
    }

    for _ in 0..mdp_count {
        let (i, j, vect) = rx.recv()?;
        result.insert((i, j), vect);
    }
    
    println!("time: {:?}, Result: {:?}", t1.elapsed().as_secs(), result);
    Ok(())
}

/// This function could be turned into value iteration
fn compute_value(mdp: MOProductMDP) -> (i32, i32, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let time_ = rng.gen_range(1..=10);
    let time_secs = std::time::Duration::from_secs(time_);
    println!("Called compute value on MDP: ({},{}) compute time: {:?}", 
        mdp.agent_id, mdp.task_id, time_secs);
    thread::sleep(time_secs);
    (mdp.agent_id, mdp.task_id, vec![1., 1.])
}

// We need a function tol assemble the data into an MDP 