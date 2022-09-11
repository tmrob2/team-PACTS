#![allow(non_snake_case)]
use crate::scpm::model::MOProductMDP;
use crate::*;
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use float_eq::float_eq;

/// Given a Product MDP, and a weight vector perform a multi-objective 
/// value-policy iteration to generate a simple scheduler which is an optimal 
/// solution to this problem
pub fn value_iteration<S>(
    prod: &MOProductMDP<S>, 
    w: &[f64], 
    eps: &f64, 
    nagents: usize, 
    ntasks:usize
) -> (Vec<f64>, Vec<f64>) 
where S: Copy + Hash + Eq {
    let size = prod.states.len();
    let nobjs: usize = nagents + ntasks;
    // convert the matrices to cs_di fmt
    let mut P: HashMap<i32, SparseMatrix> = HashMap::new();
    for action in prod.actions.iter() {
        let TP = prod.transition_mat.get(action).unwrap();
        let A = sparse_to_cs(TP);
        P.insert(*action, SparseMatrix {
            m: A,
            nr: TP.nr as usize,
            nc: TP.nc as usize,
            nnz: TP.nz as usize
        });
    }

    let mut x = vec![0f64; size]; // value vector for agent-task pair
    let mut xnew = vec![0f64; size]; // new value vector for agent-task pair
    let mut xtemp = vec![0f64; size]; // a temporary vector used for copying

    // initialise a random policy and compute its expected value
    let mut pi = random_policy(prod);

    // Determine the value of the initial vector
    let argmaxP = construct_argmax_spmatrix(prod, &pi[..], &P, size);
    let argmaxR = construct_argmax_Rvector(prod, &pi[..]);

    let mut X: Vec<f64> = vec![0.; size * 2];
    let mut Xnew: Vec<f64> = vec![0.; size * 2];
    let mut Xtemp: Vec<f64> = vec![0.; size * 2];
    let mut epsold: Vec<f64> = vec![0.; size * 2];

    let mut inf_indices: Vec<f64>;
    let mut inf_indices_old: Vec<f64> = Vec::new();
    let mut unstable_count: i32 = 0;

    value_for_init_policy(&argmaxR[..], &mut x[..], eps, &argmaxP);

    let mut pi_new: Vec<f64> = vec![-1.0; size];
    let mut q = vec![0f64; size * prod.actions.len()];
    let mut epsilon: f64; // = 1.0;
    let mut policy_stable = false;

    while !policy_stable {
        policy_stable = true;
        for (ii, a) in prod.actions.iter().enumerate() {
            // new matrix instantiation is fine
            let Pa = P.get(a).unwrap();
            //let Pa = sparse_to_cs(&mut S);
            // we use the rhs matrix dim as this represents the modified state space and
            // value vector
            // Perform the operation P.x
            let mut vmv = vec![0f64; size];
            sp_mv_multiply_f64(Pa.m, &x[..], &mut vmv);
            // Perform the operation R.w
            let mut rmv = vec![0f64; size as usize];
            let Ra = prod.rewards_mat.get(&a).unwrap();
            blas_matrix_vector_mulf64(
                &Ra.m[..],
                &w[..],
                Ra.rows as i32,
                Ra.cols as i32,
                &mut rmv[..]
            );
            assert_eq!(vmv.len(), rmv.len());
            // Perform the operation R.w + P.x
            add_vecs(&rmv[..], &mut vmv[..], size as i32, 1.0);
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
        // Copy x <- xnew
        copy(&xtemp[..], &mut x[..], size as i32);
    }

    let argmaxP = construct_argmax_spmatrix(prod, &pi[..], &P, size);
    let argmaxR = construct_argmax_Rmatrix(prod, &pi[..], nobjs);

    epsilon = 1.0;
    let agent_idx = prod.agent_id as usize;
    let task_idx = nagents + prod.task_id as usize;
    let mut epsilon_old: f64 = 1.0;
    while epsilon > *eps && unstable_count < UNSTABLE_POLICY {
        for k in 0..2 {
            // Perform the operation argmaxP.x_k
            let mut vobjvec = vec![0f64; argmaxP.nr];
            sp_mv_multiply_f64(argmaxP.m, &X[k*size..(k+1)*size], &mut vobjvec[..]);
            // Perform the operation R + P.x_k}
            let R_ = if k == 0 {
                &argmaxR.m[agent_idx*size..(agent_idx+1)*size]
            } else {
                &argmaxR.m[task_idx*size..(task_idx+1)*size]
            };
            add_vecs(&R_[..], &mut vobjvec[..], size as i32, 1.0);
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
        copy(&X[..], &mut epsold[..], obj_len);
        // Copy X <- Xnew
        copy(&Xtemp[..], &mut X[..], obj_len);
        // copy the unstable indices
        inf_indices_old = inf_indices;
        epsilon_old = epsilon;
    }
    if unstable_count >= UNSTABLE_POLICY {
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

fn random_policy<S>(prod: &MOProductMDP<S>) -> Vec<f64>
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

#[allow(non_camel_case_types, non_snake_case)]
fn construct_argmax_spmatrix<S>(
    prod: &MOProductMDP<S>, 
    pi: &[f64], 
    matrices: &HashMap<i32, SparseMatrix>,
    size: usize
) -> SparseMatrix {
    // todo convert Sparse to argmaxSparse

    let mut transposes: HashMap<i32, SparseMatrixComponents> = HashMap::new();

    for action in prod.actions.iter() {
        let A = matrices.get(action).unwrap();
        transposes.insert(
            *action, 
            deconstruct(transpose(A.m, A.nnz as i32), A.nnz, A.nr)
        );
    }

    let mut argmax_i: Vec<i32> = Vec::new();
    let mut argmax_j: Vec<i32> = Vec::new();
    let mut argmax_vals: Vec<f64> = Vec::new();

    for c in 0..size {
        let matcomp = transposes.get(&(pi[c] as i32)).unwrap();
        let p = &matcomp.p;
        let i = &matcomp.i;
        let x = &matcomp.x;
        if p[c + 1] - p[c] > 0 {
            // for each row recorder in CSS add the transpose of the coord
            for r in p[c]..p[c+1] {
                argmax_j.push(i[r as usize]);
                argmax_i.push(c as i32);
                argmax_vals.push(x[r as usize]);
            }
        }
    }

    let nnz = argmax_vals.len();

    let T = create_sparse_matrix(
        size as i32,
        size as i32,
        &argmax_i[..],
        &argmax_j[..],
        &argmax_vals[..]
    );

    let A = convert_to_compressed(T);
    SparseMatrix {
        m: A,
        nr: size,
        nc: size,
        nnz: nnz
    }
}

fn construct_argmax_Rmatrix<S>(
    prod: &MOProductMDP<S>, 
    pi: &[f64], 
    nobjs: usize
) -> DenseMatrix {
    let size = prod.states.len();
    let mut R: Vec<f64> = vec![0.; size * nobjs];
    for r in 0..size {
        let rewards = prod.rewards_mat.get(&(pi[r] as i32)).unwrap();
        for c in 0..nobjs {
            R[c * size + r] = rewards.m[c * size + r]; 
        }
    }
    DenseMatrix {
        m: R,
        rows: size,
        cols: nobjs
    }
}

fn construct_argmax_Rvector<S>(prod: &MOProductMDP<S>, pi: &[f64]) -> Vec<f64> {
    let size = prod.states.len();
    let agent_idx = prod.agent_id as usize;
    let mut R: Vec<f64> = vec![0.; size];
    for r in 0..size {
        let Ra = prod.rewards_mat.get(&(pi[r] as i32)).unwrap();
        R[r] = Ra.m[agent_idx * size + r];
    }
    R
}

fn value_for_init_policy(
    R: &[f64], 
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
        add_vecs(&R[..], &mut vmv[..], argmaxP.nr as i32, 1.0);
        copy(&vmv[..], &mut xnew[..argmaxP.nr], argmaxP.nr as i32);
        //println!("vmv: {:?}", vmv);
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

pub fn value_for_init_policy_dense(
    R: &[f64], 
    x: &mut [f64], 
    eps: &f64, 
    argmaxP: &DenseMatrix
) {
    let mut epsilon: f64 = 1.0;
    let mut xnew: Vec<f64> = vec![0.; argmaxP.cols];
    let mut epsold: Vec<f64> = vec![0.; argmaxP.cols];
    let mut unstable_count: i32 = 0;
    let mut inf_indices: Vec<f64>;
    let mut inf_indices_old: Vec<f64> = Vec::new();
    let mut epsilon_old: f64 = 1.0;
    while (epsilon > *eps) && (unstable_count < UNSTABLE_POLICY) {
        let mut vmv = vec![0f64; argmaxP.rows];
        (argmaxP.rows, &x[..], &mut vmv[..]);
        add_vecs(&R[..], &mut vmv[..], argmaxP.rows as i32, 1.0);
        copy(&vmv[..], &mut xnew[..argmaxP.rows], argmaxP.rows as i32);
        add_vecs(&xnew[..], &mut x[..], argmaxP.cols as i32, -1.0);
        epsilon = max_eps(&x[..]);
        inf_indices = x[..argmaxP.rows].iter()
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

        copy(&x[..], &mut epsold[..], argmaxP.cols as i32);
        // replace all of the values where x and epsold are equal with NEG_INFINITY or INFINITY
        // depending on sign
        copy(&vmv[..], &mut x[..argmaxP.rows], argmaxP.rows as i32);

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

pub fn print_rewards_matrices(
    blas_rewards_matrices: &HashMap<i32, DenseMatrix>,
    state_map: &HashMap<usize, (i32, i32)>,
    actions: &[i32],
) {
    for action in actions.iter() {
        let m = blas_rewards_matrices.get(action).unwrap();
        for r in 0..m.rows + 1 {
            for c in 0..m.cols + 1 {
                if r == 0 {
                    if c == 0 {
                        print!("{0:width$}", "",width=5);
                    } else {
                        print!("[o[{}]]", c-1);
                    }
                } else {
                    if c == 0 {
                        // let g = states[r - 1];
                        let (s, q) = state_map.get(&(r-1)).unwrap();
                        print!("[{},{}]", s, q);
                    } else {
                        let pval = if m.m[(c-1) * m.rows + (r-1)] == -f32::MAX as f64 {
                            f64::NEG_INFINITY
                        } else {
                            m.m[(c-1) * m.rows + (r-1)]
                        };
                        print!("{:.2} ", pval);
                    }
                }
            }
            println!();
        }
    }
}