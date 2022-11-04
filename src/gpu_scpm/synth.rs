use super::gpu_model::GPUSCPM;
use super::dp::gpu_value_iteration;
use std::{hash::Hash, time::Instant};
use crate::{Env, Solution, GenericSolutionFunctions,
    DefineSolution, blas_dot_productf32, new_targetf32, solverf32};
use hashbrown::{HashMap, HashSet};
use ordered_float::{self, OrderedFloat};

/*
The sheduler synthesis algorithm for GPU should be much simpler

we just call  gpu value iteration with a new weight vector 
and references to model, and env.
*/

fn gpu_value_iteration_wrapper<S, E>(
    model: &mut GPUSCPM, env: &mut E, w: &[f32], eps: f32, t: &[f32]
) -> Vec<f32>
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S> {
    let output = gpu_value_iteration(model, env, w, eps);
    // with the hashmap returned from value iteration determine an ordering
    println!("# Agents: {}, # Tasks: {}", model.num_agents, model.tasks.size);
    // for a particular task determine the total cost for a particular agent
    let mut cost: Vec<f32> = vec![0.; model.tasks.size + model.num_agents];
    for task in 0..model.tasks.size {
        // determine 
        // for each agent determine which task probability is the best
        let mut task_cmp = vec![0.; model.num_agents];
        for agent in 0..model.num_agents {
            // tghe objective is always fixed to task + num_agents
            let val = output.get(
                &(agent as i32, task as i32, (task + model.num_agents) as i32))
                .unwrap();
            task_cmp[agent] = *val;
        }
        // max
        task_cmp.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // first try agent 1
        // if agent 1 already has a cost and agent 2 does not
        // allocate to agent 2
        let mut allocated = false;
        for agent in 0..model.num_agents {
            if cost[agent] + task_cmp[agent] < t[agent] {
                // it is fine to allocate to this agent
                cost[model.num_agents + task] += task_cmp[agent];
                cost[agent] += *output.get(&(agent as i32, task as i32, agent as i32)).unwrap();
                allocated = true;
                break;
            }
        }
        if !allocated {
            // allocate the task to the agent with the lowest absolute
            // difference between the added task cost and the target
            let mut curr_agent_cost: Vec<(usize, f32)> = cost.iter().cloned()
                .enumerate()
                .filter(|(i, _x)| *i < model.num_agents)
                .collect();

            curr_agent_cost.sort_by(
                // Actually want the maximum cost because it is negative
                |(_, a), (_, b)| b.partial_cmp(a).unwrap()
            );
            
            // with the current cost agent vector select the first one and 
            // assign the task cost
            cost[model.num_agents + task] += task_cmp[curr_agent_cost[0].0];
            cost[curr_agent_cost[0].0] += 
                *output.get(&(curr_agent_cost[0].0 as i32, task as i32, curr_agent_cost[0].0 as i32)).unwrap();
        }
    }
    cost
}

pub fn gpu_synth<S, E>(
    model: &mut GPUSCPM, 
    env: &mut E,
    w: &mut [f32],
    eps: f32,
    t: &[f32]
)
where S: Copy + std::fmt::Debug + Hash + Eq + Send + Sync + 'static, 
E: Env<S>, Solution<S>: DefineSolution<S> + GenericSolutionFunctions<S> {
    let t1 = Instant::now();
    let mut tnew = t.to_vec();
    let mut hullset: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut hullset_X: Vec<Vec<f32>> = Vec::new();
    let mut temp_hullset_X;
    let mut weights: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut weights_v: Vec<Vec<f32>> = Vec::new();
    let mut temp_weights_: Vec<Vec<f32>>;
    //let mut X: HashSet<Vec<Mantissa>> = HashSet::new();
    let mut W: HashSet<Vec<OrderedFloat<f32>>> = HashSet::new();
    let mut schedulers: HashMap<usize, HashMap<(i32, i32), Vec<f32>>> = HashMap::new();
    let mut allocation_acc: Vec<(i32, i32, i32, Vec<f32>)> = Vec::new();

    // Starting with the initial w generate new hull points
    let r = gpu_value_iteration_wrapper(model, env, w, eps, t);
    W.insert(w.iter().cloned().map(|x| OrderedFloat(x)).collect::<Vec<OrderedFloat<f32>>>());
    let wrl = blas_dot_productf32(&r[..], &w[..]);
    let wt = blas_dot_productf32(&tnew, &w);
    println!("wt: {:.3?}, wrl: {:.3?}, wrl < wt: {}", wt, wrl, wrl < wt);
    if wrl < wt {
        println!("wrl < wt, new target required!");
        temp_hullset_X = hullset_X.clone();
        temp_hullset_X.insert(0, r.to_vec());
        temp_weights_ = weights_v.clone();
        temp_weights_.insert(0, w.to_vec());
        let tnew_ = new_targetf32(
            temp_hullset_X, 
            temp_weights_, 
            tnew.to_vec(), 
            1, 
            //model.tasks.size,
            model.num_agents, 
        );
        match tnew_ {
            Ok(x) => {
                println!("tnew: {:.2?}", x);
                tnew = x;
                //return (schedulers, hullset, x)
            }
            Err(_) => {
                panic!("A solution to the convex optimisation could not be found!");
            }
        }
    }

    hullset.insert(0, r.to_vec());
    hullset_X.push(r);
    weights.insert(0, w.to_vec());
    weights_v.push(w.to_vec());

    let mut lpvalid = true;
    let tot_objs = model.num_agents + model.tasks.size;

    //let mut w: Vec<f64> = vec![0.; tot_objs];
    let mut count: usize = 1;
    
    while lpvalid {
        
        let lpresult =
            solverf32(hullset_X.to_vec(), tnew.to_vec(), tot_objs);
        //println!("LP result: {:.2?}", lpresult);
        match lpresult {
            Ok(_sol) => {
                let new_w: Vec<OrderedFloat<f32>> = w.iter().cloned().map(|x| OrderedFloat(x)).collect();
                match W.contains(&new_w) {
                    true => {
                        println!("All points discovered");
                        lpvalid = false;
                    }
                    false => { 
                        // generate a new hull point
                        let r = gpu_value_iteration_wrapper(model, env, w, eps, t);
                        /*for (agent_, task_, r_) in alloc.into_iter() {
                            allocation_acc.push((count as i32, agent_, task_, r_));
                        }*/

                        let wrl = blas_dot_productf32(&r[..], &w[..]);
                        let wt = blas_dot_productf32(&tnew[..], &w[..]);
                        println!("wt: {:.3?}, wrl: {:.3?}, wrl < wt: {}", wt, wrl, wrl < wt);
                        if wrl < wt {
                            temp_hullset_X = hullset_X.clone();
                            temp_hullset_X.insert(0, r.to_vec());
                            temp_weights_ = weights_v.clone();
                            temp_weights_.insert(0, w.to_vec());
                            let tnew_ = new_targetf32(
                                temp_hullset_X, 
                                temp_weights_, 
                                tnew.to_vec(), 
                                hullset_X.len(), 
                                //model.tasks.size,
                                model.num_agents, 
                            );
                            match tnew_ {
                                Ok(x) => {
                                    println!("tnew: {:.3?}", x);
                                    tnew = x;
                                    //return (schedulers, hullset, x)
                                }
                                Err(_) => {
                                    panic!("A solution to the convex optimisation could not be found!");
                                }
                            }
                        } else {
                            //schedulers.insert(count, pis);
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
            Err(e) => {lpvalid = false;}
        }
    }
    println!("Time: {:.3}", t1.elapsed().as_secs_f32());
}