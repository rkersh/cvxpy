from __future__ import print_function
import cvxpy

# Data
fixed_cost = [1900.0, 820.0, 805.0, 464.0, 3912.00, 556.0]
var_cost = [15.0, 20.0, 45.0, 64.0, 12.0, 56.0]
capacity = [100.0, 20.0, 405.0, 264.0, 12.0, 256.0]
demand = 22.0
n_machines = len(var_cost)
epsilon = 1e-6

# Variables
operate = cvxpy.Variable(n_machines)
use = cvxpy.Bool(n_machines)

# Objective
obj = cvxpy.Minimize(
    cvxpy.sum_entries(cvxpy.mul_elemwise(var_cost, operate)) +
    cvxpy.sum_entries(cvxpy.mul_elemwise(fixed_cost, use)))

# Constraints
constraints = []
for var in operate:
    constraints.append(0.0 <= var)
for var in use:
    constraints.append(0.0 <= var)
    constraints.append(var <= 1.0)
for var, cap in zip(operate, capacity):
    constraints.append(var <= cap)
for opvar, usevar, cap in zip(operate, use, capacity):
    constraints.append(opvar - cap * usevar <= 0.0)
constraints.append(cvxpy.sum_entries(operate) == demand)

# Solve and display results
prob = cvxpy.Problem(obj, constraints)
prob.solve(solver=cvxpy.CPLEX, verbose=True)

print("Solution status = ", prob.status)
print("Obj", prob.value)
for i, var in enumerate(operate):
    if var.value > epsilon:
        print("E", i, "is used for", var.value)
print("--------------------------------------------")
