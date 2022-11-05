use hashbrown::HashMap;

use super::gpu_model;
use crate::{Env, Solution, GenericSolutionFunctions,
    DefineSolution, initial_policy_value_ffi,
    policy_optimisation_ffi, multi_objective_values_ffi, sparse::definition::CxxMatrixf32, GPUProblemMetaData};
use std::hash::Hash;
use crate::sparse::argmax;

pub fn gpu_value_iteration<S, E>(
    model: &mut gpu_model::GPUSCPM,
    env: &mut E,
    w: &[f32],
    eps: f32,
    argmaxP: &CxxMatrixf32,
    argmaxR: &CxxMatrixf32,
    Pcsr: &CxxMatrixf32,
    Rcsr: &CxxMatrixf32, 
    data: &GPUProblemMetaData,
    init_pi: &[i32]
) ->  HashMap<(i32, i32, i32), f32>
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S> {
    let nobjs = model.num_agents + model.tasks.size;
    let num_actions = env.get_action_space().len();
    let num_models = model.num_agents * model.tasks.size;
    
    //let mut value = vec![0.; data.transition_prod_block_size];
    let mut x = vec![0.; data.transition_prod_block_size];
    let mut y = vec![0.; data.transition_prod_block_size];
    let mut rmv = vec![0.; data.transition_prod_block_size];
    initial_policy_value_ffi(
        &argmaxP,
        &argmaxR,
        &mut x,
        &mut y,
        &w,
        &mut rmv,
        eps
    );
    
    // Preprocess the weight vector
    let mut w_: Vec<f32> = 
    Vec::with_capacity(w.len() * (num_actions * num_models) as usize);
    
    // Preprocess the value vector
    let mut init_value_: Vec<f32> = 
    Vec::with_capacity(y.len() * (num_actions * num_models) as usize);
    
    for _ in 0..(num_actions * num_models) {
        w_.extend(w); // this makes (|W|.|A|) x 1
    }
    for _ in 0..(num_actions) {
        init_value_.extend(&y);
    }
    //println!("init value\n:{:?}", init_value_);

    // Reinitialise the input mem alloc vectors into the policy optimisation
    // interface
    let mut x = vec![0.; Pcsr.m as usize];
    let mut y = vec![0.; Pcsr.m as usize];
    let rmv = vec![0.; Pcsr.m as usize];
    let mut policy = init_pi.to_vec();

    // Do policy optimisation to find optimal pi under w
    policy_optimisation_ffi(
        &init_value_, 
        &mut policy,
        &Pcsr, 
        &Rcsr, 
        &w_, 
        &mut x, 
        &mut y,
        &rmv, 
        eps,
        data.transition_prod_block_size as i32,
        num_actions as i32
    );
    
    let argmaxP = argmax::multiobj_argmaxP(
        &Pcsr, 
        &policy, 
        data.transition_prod_block_size as i32, 
        data.transition_prod_block_size as i32, 
        nobjs
    );
    
    let argmaxR = argmax::multiobj_argmaxR(
        &Rcsr, 
        &policy, 
        data.transition_prod_block_size as i32, 
        nobjs
    );
    
    let mut x = vec![0.; data.transition_prod_block_size * nobjs];
    assert_eq!(argmaxR.len(), argmaxP.m as usize);
    multi_objective_values_ffi(
        &argmaxR, 
        &argmaxP, 
        eps, 
        &mut x
    );

    //println!("x:\n{:?}", x);

    // get the indices of x which correspond to the initial values
    // for each of the product models
    let mut pid = 0;
    let bl = data.transition_prod_block_size;
    let mut output: HashMap<(i32, i32, i32), f32> = 
        HashMap::new();
    for k in 0..nobjs {
        // get the initial indices
        for agent in 0..model.num_agents {
            for task in 0..model.tasks.size {
                //println!("A: {}, T: {} Obj: {} Val: {:.2}",
                //    agent, task, k, x[k * bl + init_states[pid]]);
                output.insert((agent as i32, task as i32, k as i32), 
                    x[k * bl + data.init_state_idx[pid]]);
                pid+=1;
            }
        }
        pid = 0;
    }
    output
}