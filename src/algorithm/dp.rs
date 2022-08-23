#![allow(non_snake_case)]
use crate::scpm::model::MOProductMDP;
use crate::*;
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use float_eq::float_eq;

/// Given a Product MDP, and a weight vector perform a multi-objective 
/// value-policy iteration to generate a simple scheduler which is an optimal 
/// solution to this problem
pub fn value_iteration(prod: &MOProductMDP, w: &[f64], eps: &f64) 
    -> (Vec<f64>, Vec<f64>) {
    let size = prod.states.len();

    // we will need to construct a set of size x size 
    // transition matrices for each action
    let P = construct_spblas_matrices(prod);
    let R = construct_rewards_matrices(prod);

    let mut x = vec![0f64; size]; // value vector for agent-task pair
    let mut xnew = vec![0f64; size]; // new value vector for agent-task pair
    let mut xtemp = vec![0f64; size]; // a temporary vector used for copying

    // initialise a random policy and compute its expected value
    let mut pi = random_policy(prod);

    // Determine the value of the initial vector
    let argmaxP = construct_argmax_spPmatrix(prod, &pi[..]);
    let mut argmaxR = construct_argmax_Rvector(prod, &pi[..]);

    let mut X: Vec<f64> = vec![0.; size * 2];
    let mut Xnew: Vec<f64> = vec![0.; size * 2];
    let mut Xtemp: Vec<f64> = vec![0.; size * 2];
    let mut epsold: Vec<f64> = vec![0.; size * 2];

    let mut inf_indices: Vec<f64>;
    let mut inf_indices_old: Vec<f64> = Vec::new();
    let mut unstable_count: i32 = 0;

    value_for_init_policy(&mut argmaxR[..], &mut x[..], eps, &argmaxP);

    let mut pi_new: Vec<f64> = vec![-1.0; size];
    let mut q = vec![0f64; size * prod.actions.len()];
    let mut epsilon: f64; // = 1.0;
    let mut policy_stable = false;

    while !policy_stable {
        policy_stable = true;
        for (ii, a) in prod.actions.iter().enumerate() {
            // new matrix instantiation is fine
            let mut S = P.get(a).unwrap();
            let Pa = sparse_to_cs(&mut S);
            // we use the rhs matrix dim as this represents the modified state space and
            // value vector
            // Perform the operation P.x
            let mut vmv = vec![0f64; size];
            sp_mv_multiply_f64(Pa, &x[..], &mut vmv);
            //println!("action: {}, idx: {}", a, ii);
            //println!("vmv: {:?}", vmv);
            // Perform the operation R.w
            let mut rmv = vec![0f64; S.nr as usize];
            let Ra = R.get(&a).unwrap();
            //println!("R: {:?}", R.m);
            //println!("w: {:?}", w);
            //println!("r nr: {}, r nc: {}", R.nr, R.nc);
            blas_matrix_vector_mulf64(
                &Ra.m[..],
                &w[..],
                Ra.rows as i32,
                Ra.cols as i32,
                &mut rmv[..]
            );
            assert_eq!(vmv.len(), rmv.len());
            // Perform the operation R.w + P.x
            add_vecs(&rmv[..], &mut vmv[..], S.nr as i32, 1.0);
            //println!("vmv after addition with rmv:\n{:?}", vmv);
            // Add the value vector to the Q table
            update_qmat(&mut q[..], &vmv[..], ii, prod.actions.len() as usize).unwrap();
        }
        // determine the maximum values for each state in the matrix of value estimates
        max_values(&mut xnew[..], &q[..], &mut pi_new[..], size, prod.actions.len());
        copy(&xnew[..], &mut xtemp[..], size as i32);
        // copy the new value vector to calculate epsilon
        add_vecs(&x[..], &mut xnew[..], size as i32, -1.0);
        update_policy(&xnew[..], &eps, &mut pi[..], &pi_new[..], size, &mut policy_stable);
        // Calculate the max epsilon
        //epsilon = self.max_eps(&xnew[..]);
        //println!("eps: {:?}", epsilon);
        // Copy x <- xnew
        copy(&xtemp[..], &mut x[..], size as i32);
        //println!("\rx: {:?}", x);
    }

    let argmaxP = construct_argmax_spPmatrix(prod, &pi[..]);
    let argmaxR = construct_argmax_Rmatrix(prod,  &pi[..], &R);

    epsilon = 1.0;

    let mut epsilon_old: f64 = 1.0;
    while epsilon > *eps && unstable_count < UNSTABLE_POLICY {
        for k in 0..2 {
            // Perform the operation argmaxP.x_k
            let mut vobjvec = vec![0f64; argmaxP.nr];
            sp_mv_multiply_f64(argmaxP.m, &X[k*size..(k+1)*size], &mut vobjvec[..]);
            // Perform the operation R + P.x_k
            add_vecs(&argmaxR.m[k*size..(k+1)*size], &mut vobjvec[..], size as i32, 1.0);
            copy(&vobjvec[..], &mut Xnew[k*size..(k+1)*size], size as i32);
        }
        // determine the difference between X, Xnew
        let obj_len = (size * 2) as i32;
        copy(&Xnew[..], &mut Xtemp[..], obj_len);
        add_vecs(&Xnew[..], &mut X[..], obj_len, -1.0);
        epsilon = max_eps(&X[..]);
        inf_indices = X.iter()
            .zip(epsold.iter())
            .enumerate()
            .filter(|(_ix, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
            .map(|(ix, _)| ix as f64)
            .collect::<Vec<f64>>();

        if inf_indices.len() == inf_indices_old.len() {
            if inf_indices.iter().zip(inf_indices_old.iter()).all(|(val1, val2)| val1 == val2) {
                //println!("eps: {} eps old: {}, inf: {:?}", epsilon, epsilon_old, inf_indices);
                if epsilon < epsilon_old {
                    // the value function is still contracting an this is converging, therefore
                    // not unstable
                    unstable_count = 0;
                } else {
                    unstable_count += 1;
                }
            } else {
                unstable_count = 0;
            }
        } else {
            unstable_count = 0;
        }
        //println!("{:?}", t5.elapsed().as_secs_f64());
        copy(&X[..], &mut epsold[..], obj_len);
        // Copy X <- Xnew
        copy(&Xtemp[..], &mut X[..], obj_len);
        // copy the unstable indices
        inf_indices_old = inf_indices;
        epsilon_old = epsilon;
        //println!("X: {:.2?}", X);
    }
    if unstable_count >= UNSTABLE_POLICY {
        //println!("inf indices: {:?}", inf_indices_old);
        for ix in inf_indices_old.iter() {
            if X[*ix as usize] < 0. {
                X[*ix as usize] = -f32::MAX as f64;
            }
        }
    }

    let mut r: Vec<f64> = vec![0.; 2];
    // get the index of the initial state
    let init_idx = prod.get_state_map().get(&prod.initial_state).unwrap();
    r[0] = X[*init_idx];
    r[1] = X[size + *init_idx];
    (pi, r)
}

fn random_policy(prod: &MOProductMDP) -> Vec<f64> {
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

fn construct_spblas_matrices(prod: &MOProductMDP) -> HashMap<i32, COO> {
    let mut sparse_matrices: HashMap<i32, COO> = HashMap::new();

    let size = prod.states.len();

    for action in prod.actions.iter() {
        let mut r: Vec<i32> = Vec::new();
        let mut c: Vec<i32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();

        for state in prod.states.iter() {
            let row_idx = prod.get_state_map().get(state).unwrap();
            match prod.transitions.get(&(*state, *action)) {
                None => { }
                Some(v) => { 
                    for (sprime, p) in v.iter() {
                        let col_idx = prod.get_state_map().get(&sprime).unwrap();
                        r.push(*row_idx as i32);
                        c.push(*col_idx as i32);
                        vals.push(*p);
                    }
                }
            }
        }

        let nnz = vals.len();
        let S = COO {
            nzmax: nnz as i32,
            nr: size as i32,
            nc: size as i32,
            i: r,
            j: c,
            x: vals,
            nz: nnz as i32
        };
        sparse_matrices.insert(*action, S);
    }
    sparse_matrices
}

#[allow(non_camel_case_types, non_snake_case)]
fn construct_argmax_spPmatrix(prod: &MOProductMDP, pi: &[f64]) -> SparseMatrix {
    let mut r: Vec<i32> = Vec::new();
    let mut c: Vec<i32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    
    let size = prod.states.len();

    for state in prod.states.iter() {
        let row_idx = prod.get_state_map().get(state).unwrap();
        let action = pi[*row_idx] as i32;
        match prod.transitions.get(&(*state, action)) {
            Some(v) => {
                for (sprime, p) in v.iter() {
                    let col_idx = prod.get_state_map().get(sprime).unwrap();
                    r.push(*row_idx as i32);
                    c.push(*col_idx as i32);
                    vals.push(*p); 
                }
            }
            None => { }
        }
    }

    let T = create_sparse_matrix(
        size as i32,
        size as i32,
        &r[..],
        &c[..],
        &vals[..]
    );
    let A = convert_to_compressed(T);
    SparseMatrix {
        m: A,
        nr: size,
        nc: size,
        nnz: vals.len()
    }
}

#[allow(non_snake_case)]
fn construct_rewards_matrices(prod: &MOProductMDP) -> HashMap<i32, DenseMatrix> {
    let mut rewards: HashMap<i32, DenseMatrix> = HashMap::new();
    let size: usize = prod.states.len();
    for action in prod.actions.iter() {
        let mut R: Vec<f64> = vec![-f32::MAX as f64; size * 2];
        for state in prod.states.iter() {
            let row_idx = prod.get_state_map().get(state).unwrap();
            match prod.rewards.get(&(*state, *action)) {
                Some(r) => {
                    R[*row_idx] = r[0];
                    R[size + *row_idx] = r[1];
                }
                None => { }
            }
        }
        rewards.insert(*action, DenseMatrix {
            m: R,
            rows: size,
            cols: 2
        });
    }
    rewards
}

#[allow(non_snake_case)]
fn construct_argmax_Rmatrix(
    prod: &MOProductMDP, 
    pi: &[f64], 
    rmatricies: &HashMap<i32, DenseMatrix>
) -> DenseMatrix {
    let size = prod.states.len();
    let mut R: Vec<f64> = vec![0.; size * 2];
    for state in prod.states.iter() {
        let row_idx = prod.get_state_map().get(state).unwrap();
        let action = pi[*row_idx] as i32;
        let rewards = rmatricies.get(&action).unwrap();
        for c in 0..2 {
            R[c * size + *row_idx] = rewards.m[c * size + *row_idx];
        }
    }
    DenseMatrix {
        m: R,
        rows: size,
        cols: 2
    }
}

#[allow(non_snake_case)]
fn construct_argmax_Rvector(prod: &MOProductMDP, pi: &[f64]) -> Vec<f64> {
    let size = prod.states.len();
    let mut R: Vec<f64> = vec![0.; size];
    for state in prod.states.iter() {
        let row_idx = prod.get_state_map().get(state).unwrap();
        let action = pi[*row_idx] as i32;
        match prod.rewards.get(&(*state, action)) {
            Some(r) => {
                R[*row_idx] = r[0];
            }
            None => { }
        }
    }
    R
}

fn value_for_init_policy(
    R: &mut [f64], 
    x: &mut [f64], 
    eps: &f64, 
    argmaxP: &SparseMatrix
) {
    let mut epsilon: f64 = 1.0;
    let mut xnew: Vec<f64> = vec![0.; argmaxP.nc];
    let mut epsold: Vec<f64> = vec![0.; argmaxP.nc];
    let mut unstable_count: i32 = 0;
    let mut inf_indices: Vec<f64>;
    let mut inf_indices_old: Vec<f64> = Vec::new();
    let mut epsilon_old: f64 = 1.0;
    while (epsilon > *eps) && (unstable_count < UNSTABLE_POLICY) {
        let mut vmv = vec![0f64; argmaxP.nr];
        sp_mv_multiply_f64(argmaxP.m, &x[..], &mut vmv[..]);
        add_vecs(&mut R[..], &mut vmv[..], argmaxP.nr as i32, 1.0);
        copy(&vmv[..], &mut xnew[..argmaxP.nr], argmaxP.nr as i32);
        add_vecs(&xnew[..], &mut x[..], argmaxP.nc as i32, -1.0);
        epsilon = max_eps(&x[..]);
        inf_indices = x[..argmaxP.nr].iter()
            .zip(epsold.iter())
            .enumerate()
            .filter(|(_, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
            .map(|(ix, _)| ix as f64)
            .collect::<Vec<f64>>();

        if inf_indices.len() == inf_indices_old.len() {
            if inf_indices.iter().zip(inf_indices_old.iter()).all(|(val1, val2)| val1 == val2) {
                if epsilon < epsilon_old {
                    unstable_count = 0;
                } else {
                    unstable_count += 1;
                }
            } else {
                unstable_count = 0;
            }
        } else {
            unstable_count = 0;
        }

        copy(&x[..], &mut epsold[..], argmaxP.nc as i32);
        // replace all of the values where x and epsold are equal with NEG_INFINITY or INFINITY
        // depending on sign
        copy(&vmv[..], &mut x[..argmaxP.nr], argmaxP.nr as i32);

        inf_indices_old = inf_indices;
        epsilon_old = epsilon;
    }
    if unstable_count >= UNSTABLE_POLICY {
        for ix in inf_indices_old.iter() {
            if x[*ix as usize] < 0. {
                x[*ix as usize] = -f32::MAX as f64;
            }
        }
    }
}