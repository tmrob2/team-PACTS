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
    return solution

def new_target_(X, W, t, l, m, n, ii, cstep, pstep):
    # n - num agents
    # m - num tasks
    # ii - iterations
    # cstep - cost step
    model = LpProblem("NoName", LpMinimize)
    solver = CPLEX_CMD(path=cplex_dir, msg=False)

    lambda_ = {i: LpVariable(name=f"lambda{i}", lowBound=0, upBound=1.0) for i in range(l) }
    z = {i: LpVariable(name=f"z{i}", upBound=upper_bound(t, ii, pstep, i, cstep, n)) for i in range(n + m)}
    epsilon = LpVariable(name="epsilon")

    for j in range(n+m):
        model += lpSum([X[k][j] * lambda_[k] for k in range(l)]) - epsilon - z[j] <= 0

    for k in range(l):
        model += lpDot(W[k], z[k]) <= np.dot(W[k], X[k])

    model += lpSum([lambda_[k] for k in range(l)]) == 1

    model += epsilon

    print(model)
    status = model.solve(solver)
    #print(status)
    solution = [value(z[j]) for j in range(n + m)]
    #print(solution)
    return solution

def upper_bound(t, ii, prob_step, j, cstep, n):
    if j < n:
        return t[j] - ii * cstep
    else:
        if t[j] - ii * prob_step:
            return t[j] - ii * prob_step
        else:
            raise("target doesn't exist")