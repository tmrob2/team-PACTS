use super::gpu_model;
use crate::{Env, Solution, GenericSolutionFunctions,
    DefineSolution, construct_gpu_problem, initial_policy_value_ffi,
    policy_optimisation_ffi, multi_objective_values_ffi};
use std::hash::Hash;
use crate::sparse::argmax;

pub fn gpu_value_iteration<S, E>(
    model: &mut gpu_model::GPUSCPM,
    env: &mut E,
    w: &[f32],
    eps: f32,
)
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S> {
    let nobjs = model.num_agents + model.tasks.size;
    let num_actions = env.get_action_space().len();
    let num_models = model.num_agents * model.tasks.size;

    let (mat, init_pi, r, data) = 
        construct_gpu_problem(model, env);

    // Transition and rewards matrices are consumed and replaces by 
    // their CSR equivalents. 
    println!("init pi: \n{:?}", init_pi);
    let Pcsr = crate::sparse::compress::compress(mat);
    let Rcsr = crate::sparse::compress::compress(r);
    
    let argmaxP = argmax::argmaxM(
        &Pcsr, 
        &init_pi, 
        data.transition_prod_block_size as i32, 
        data.transition_prod_block_size as i32
    );
    
    let argmaxR = argmax::argmaxM(
        &Rcsr, 
        &init_pi, 
        data.transition_prod_block_size as i32, 
        data.reward_obj_prod_block_size as i32
    );
    
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
    println!("init value\n:{:?}", init_value_);

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
    println!("updated policy: \n{:?}", policy);
    println!("output value: \n{:?}", y);
    
    
    let argmaxP = argmax::multiobj_argmaxP(
        &Pcsr, 
        &policy, 
        data.transition_prod_block_size as i32, 
        data.transition_prod_block_size as i32, 
        1
    );
    
    let argmaxR = argmax::multiobj_argmaxR(
        &Rcsr, 
        &policy, 
        data.transition_prod_block_size as i32, 
        1
    );

    //let bl_ = data.transition_prod_block_size;
    println!("argmax R: \n{:?}", &argmaxR); //[2 * bl_..3 * bl_]);
    
    let mut x = vec![0.; data.transition_prod_block_size];
    assert_eq!(argmaxR.len(), argmaxP.m as usize);
    multi_objective_values_ffi(
        &argmaxR, 
        &argmaxP, 
        eps, 
        &mut x
    );

    println!("x:\n{:?}", x);

    // get the indices of x which correspond to the initial values
    // for each of the product models
    let mut pid = 0;
    let init_states = data.init_state_idx;
    let bl = data.transition_prod_block_size;
    for k in 0..1 {
        // get the initial indices
        for agent in 0..model.num_agents {
            for task in 0..model.tasks.size {
                println!("A: {}, T: {} Obj: {} Val: {:.2}",
                    agent, task, k, x[k * bl + init_states[pid]]);
                pid+=1;
            }
        }
        pid = 0;
    }
    
}