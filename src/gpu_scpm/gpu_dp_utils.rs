use rand::seq::SliceRandom;
use std::hash::Hash;
use crate::{gpu_scpm::gpu_model::MOProductMDP, sparse::{definition::CxxMatrixf32, argmax::argmaxM}};

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