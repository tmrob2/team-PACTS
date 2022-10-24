use rand::seq::SliceRandom;
use std::hash::Hash;
use crate::{gpu_scpm::gpu_model::MOProductMDP, gather_policy_ffi};
use pyo3::prelude::*;

pub fn random_policy<S>(prod: &MOProductMDP<S>) -> Vec<f64>
where S: Copy + Hash + Eq {
    let mut pi: Vec<f64> = vec![0.; prod.states.len()];
    for state in prod.states.iter() {
        let state_idx = prod.get_state_map().get(state).unwrap();
        // choose a random action at state s
        let actions = prod.get_available_actions(state);
        let act = actions.choose(&mut rand::thread_rng()).unwrap();
        pi[*state_idx] = *act as f64;
    }
    pi
}

#[pyfunction]
pub fn map_policy_to_gather(pi: Vec<i32>, nc: i32, nr: i32, prod_block_size: i32) -> Vec<i32> {
    let mut output: Vec<i32> = vec![0; prod_block_size as usize * nc as usize];
    //gather_policy_ffi(&pi, &mut output, size, nr, nc, prod_block_size);
    /*let mut rows = vec![0; prod_block_size as usize];
    for i in 0..prod_block_size as usize {
        rows[i] = i as i32 + prod_block_size * pi[i];
    }

    let mut idx: usize = 0;
    for r in 0..prod_block_size {
        for c in 0..nc as usize {
            output[idx] = c as i32 * nr + rows[r as usize];
            idx += 1;
        }
    }*/
    gather_policy_ffi(&pi, &mut output, nc, nr, prod_block_size);
    output
}