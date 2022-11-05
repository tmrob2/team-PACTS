#from docplex.mp.model import Model
import numpy as np
from pulp import *
cplex_dir = "/opt/ibm/ILOG/CPLEX_Studio221/cplex/bin/x86-64_linux/cplex"

def hyperplane_solver(X, t, nobj):
    # Define the model
    #print(X)
    model = LpProblem("NoName", LpMaximize)
    solver = CPLEX_CMD(path=cplex_dir, msg=False)
    # Define the decision variables
    w = {i: LpVariable(name=f"w{i}", lowBound=0., upBound=1.) for i in range(nobj)}
    d = LpVariable(name="d")

    for x in X:
        for i in range(nobj):
            model += w[i] * (t[i] - x[i]) >= d

    model += lpSum([w[i] for i in range(nobj)]) == 1

    model += d

    #print(model)

    # solve the problem
    status = model.solve(solver)
    solution = [value(w[i]) for i in range(nobj)]
    #print("w: ", solution)
    return solution


import cvxpy as cp

def eucl_new_target(X, W, t, l, n):
    z = cp.Variable(len(t)) # vector variable with shape (5,)
    #obj = cp.sum_squares(A @ z - t)
    obj = cp.norm(z - t, 2)
    constraints = [ ]

    for i in range(n):
        constraints.append(z[i] <= t[i] - 5.)
    
    for k in range(l):
        constraints.append(np.dot(W[k], X[k]) >= W[k] @ z)

    for i in range(n, len(t)):
        constraints.append(z[i] <= t[i])
        constraints.append(z[i] >= 0)
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    #print(prob)
    prob.solve()
    print("status", prob.status)
    print("optimal value", prob.value)
    print("optimal var", z.value)
    return z.value


def randomised_scheduler(r, t, l, m, n):
    model = LpProblem("NoName", LpMinimize)
    solver = CPLEX_CMD(path=cplex_dir, msg=False)
    c_dict = {}
    p_dict = {}
    #I = set([])

    # convert r to a dictionary
    # we also have to collect the set of I, J, K here as well
    for (k, i, j, r) in r:
        c_dict[f"{i}_{j}_{k}"] = r[0]
        if f"{j}_{k}" not in p_dict.keys():
            p_dict[f"{j}_{k}"] = r[1]
        else:
            p_dict[f"{j}_{k}"] += r[1]
            
    # Define the decision variables
    v = {f"{j}_{k}": LpVariable(name=f"v{j}_{k}", lowBound=0., upBound=1.) 
            for j in range(m) for k in range(l)}

    for i in range(n):
        sum_var = []
        for j in range(m):
            for k in range(l):
                if f"{i}_{j}_{k}" in c_dict.keys():
                    sum_var.append((v[f"{j}_{k}"], c_dict[f"{i}_{j}_{k}"]))
                    #sum_var.append(v[f"{j}_{k}"] * c_dict[f"{i}_{j}_{k}"])
        #model += lpSum(sum_var) >= t[i]
        exp = LpAffineExpression(sum_var)
        c = LpConstraint(exp, 1, name=f"cost{i}", rhs=t[i])
        ep = c.makeElasticSubProblem(penalty=100, proportionFreeBoundList=[t[i], 0])
        model.extend(ep)

    for j in range(m):
        model += lpSum([v[f"{j}_{k}"]*p_dict[f"{j}_{k}"] for k in range(l)]) >= t[n + j]
    
    for j in range(m):
        model += lpSum([v[f"{j}_{k}"] for k in range(l)]) == 1

    model += 0

    #print(model)

    status = model.solve(solver)
    print("status", status)
    solution = [value(v[f"{j}_{k}"]) for j in range(m) for k in range(l)]
    print("solution", solution)
    return solution

