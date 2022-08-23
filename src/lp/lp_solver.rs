use hashbrown::HashMap;
use crate::blas_dot_product;
use gurobi::{attr, Continuous, LinExpr, Model, param, Status, Greater, Equal, Less, Var};


/// l is the size of the hullset
/// n is the number of agents
/// m is the number of tasks
pub fn new_target(
    hullset: &HashMap<usize, Vec<f64>>,
    weights: &HashMap<usize, Vec<f64>>,
    target: &[f64],
    l: usize,
    m: usize,
    n: usize,
    cost_step: f64,
    prob_step: f64,
    iterations: i32
) -> Result<Vec<f64>, Box<dyn std::error::Error>> { 
    let mut env = gurobi::Env::new("").unwrap();
    env.set(param::OutputFlag, 0).unwrap();
    env.set(param::LogToConsole, 0).unwrap();
    let mut model = Model::new("model2", &env).unwrap();
    let mut v: HashMap<String, gurobi::Var> = HashMap::new();
    for k in 0..l {
        v.insert(
            format!("lambda{}", k),
            model
                .add_var(
                    &*format!("lambda{}", k),
                    Continuous,
                    0.0,
                    0.00,
                    1.0,
                    &[],
                    &[]
                )?);
    }
    for j in 0..n {
        let cost_upper_bound: f64 = target[j] - iterations as f64 * cost_step;
        v.insert(
            format!("z{}", j),
            model.add_var(
                &*(format!("z{}", j)),
                Continuous,
                0.,
                -gurobi::INFINITY,
                cost_upper_bound,
                &[], &[]
            )?);
    }
    for j in n..n + m {
        let prob_lower_bound: f64 = if target[j] - iterations as f64 * prob_step >= 0. {
            target[j] - iterations as f64 * prob_step
        } else {
            panic!("target doesn't exist")
        };
        v.insert(
            format!("z{}", j),
            model.add_var(
                &*(format!("z{}", j)),
                Continuous,
                0.,
                prob_lower_bound,
                1.,
                &[], &[]
            )?);
    }
    let epsilon = model.add_var(
        "epsilon", Continuous, 1.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
    )?;

    println!("added variables");

    model.update()?;

    for j in 0..n + m {
        let mut e = LinExpr::new();
        // greater than epsilon constraint
        for k in 0..l {
            let x_kj = hullset.get(&k).unwrap()[j];
            let lambda = v.get(&*format!("lambda{}",k)).unwrap();
            e = e.clone().add_term(x_kj, lambda.clone());
        }
        e = e.clone().add_term(1.0, epsilon.clone());
        let zj = v.get(&*format!("z{}", j)).unwrap().clone();
        e = e.clone().add_term(-1., zj);
        model.add_constr(
            &*format!("c_0_{}", j),
            e,
            Greater,
            0.
        );
        let mut e = LinExpr::new();
        // greater than epsilon constraint
        for k in 0..l {
            let x_kj = hullset.get(&k).unwrap()[j];
            let lambda = v.get(&*format!("lambda{}",k)).unwrap();
            e = e.clone().add_term(x_kj, lambda.clone());
        }
        e = e.clone().add_term(-1.0, epsilon.clone());
        let zj = v.get(&*format!("z{}", j)).unwrap().clone();
        e = e.clone().add_term(-1., zj);
        model.add_constr(
            &*format!("c_1_{}", j),
            e,
            Less,
            0.
        );
    }
    model.update()?;

    println!("added distance constraints");

    for k in 0..l {
        let wr = blas_dot_product(&weights.get(&k).unwrap()[..], &hullset.get(&k).unwrap()[..]);
        let mut e = LinExpr::new();
        //e = e.add_constant(wr);
        let mut zs: Vec<_> = Vec::new();
        for j in 0..n + m {
            zs.push(v.get(&*format!("z{}", j)).unwrap().clone());
        }
        e = e.clone().add_terms(&weights.get(&k).unwrap()[..], &zs[..]);
        model.add_constr(
            &*format!("c_z{}", k),
            e,
            Less,
            wr
        );
    }

    model.update()?;

    println!("added inside hull constraint");

    let mut e = LinExpr::new();
    let coeffs: Vec<f64> = vec![1.0; l];
    let mut lambdas: Vec<_> = Vec::new();
    for k in 0..l {
        lambdas.push(v.get(&*format!("lambda{}",k)).unwrap().clone());
    }
    e = e.add_terms(&coeffs[..], &lambdas[..]);
    model.add_constr("lambda_sum", e, Equal, 1.0)?;

    model.update()?;
    //model.write("model output.txt");
    model.set_objective(epsilon.clone(), gurobi::Minimize)?;
    model.optimize()?;
    let mut vars: Vec<Var> = Vec::new();
    for j in 0..n + m {
        vars.push(v.get(&*format!("z{}", j)).unwrap().clone());
    }
    let result = match model.get_values(attr::X, &vars[..]) {
        Ok(x) => { x }
        Err(e) => { panic!("The variables could not be extracted from the solution!") }
    };
    Ok(result)
}

pub fn gurobi_solver(
    h: &HashMap<usize, Vec<f64>>, 
    t: &[f64], 
    dim: &usize
) -> Option<Vec<f64>> { 
    let mut env = gurobi::Env::new("").unwrap();
    env.set(param::OutputFlag, 0).ok();
    env.set(param::LogToConsole, 0).ok();
    env.set(param::InfUnbdInfo, 1).ok();
    //env.set(param::FeasibilityTol,10e-9).unwrap();
    env.set(param::NumericFocus,2).ok();
    let mut model = Model::new("model1", &env).unwrap();

    // add variables
    let mut vars: HashMap<String, gurobi::Var> = HashMap::new();
    for i in 0..*dim {
        vars.insert(format!("w{}", i), model.add_var(
            &*format!("w{}", i),
            Continuous,
            0.0,
            0.0,
            1.0,
            &[],
            &[]).unwrap()
        );
    }
    let d = model.add_var(
        "d", Continuous, 0.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
    ).unwrap();

    model.update().unwrap();
    let mut w_vars = Vec::new();
    for i in 0..*dim {
        let w = vars.get(&format!("w{}", i)).unwrap();
        w_vars.push(w.clone());
    }
    let t_expr = LinExpr::new();
    let t_expr1 = t_expr.add_terms(&t[..], &w_vars[..]);
    let t_expr2 = t_expr1.add_term(1.0, d.clone());
    let t_expr3 = t_expr2.add_constant(-1.0);
    model.add_constr("t0", t_expr3, gurobi::Greater, 0.0).ok();

    for ii in 0..h.len() {
        let expr = LinExpr::new();
        let expr1 = expr.add_terms(&h.get(&ii).unwrap()[..], &w_vars[..]);
        let expr2 = expr1.add_term(1.0, d.clone());
        let expr3 = expr2.add_constant(-1.0);
        model.add_constr(&*format!("c{}", ii), expr3, gurobi::Less, 0.0).ok();
    }
    let w_expr = LinExpr::new();
    let coeffs: Vec<f64> = vec![1.0; *dim];
    let final_expr = w_expr.add_terms(&coeffs[..], &w_vars[..]);
    model.add_constr("w_final", final_expr, gurobi::Equal, 1.0).ok();

    model.update().unwrap();
    model.set_objective(&d, gurobi::Maximize).unwrap();
    model.optimize().unwrap();
    let mut varsnew = Vec::new();
    for i in 0..*dim {
        let var = vars.get(&format!("w{}", i)).unwrap();
        varsnew.push(var.clone());
    }
    let val = model.get_values(attr::X, &varsnew[..]).unwrap();
    //println!("model: {:?}", model.status());
    if model.status().unwrap() == Status::Infeasible {
        None
    } else {
        Some(val)
    }
}
