from __future__ import print_function
from cvxpy import *

x1, x2, x3 = Variable(), Variable(), Variable()

constraints = [-x1 + x2 + x3 <= 20.0,
               x1 - 3.0 * x2 + x3 <= 30.0,
               0.0 <= x1, x1 <= 40.0,
               0.0 <= x2,
               0.0 <= x3]

obj = Maximize(x1 + 2.0 * x2 + 3.0 * x3)

prob = Problem(obj, constraints)
prob.solve(solver=CPLEX)
print("status:", prob.status)
print("objective:", prob.value)
print("values: x1 = {0}, x2 = {1}, x3 = {2}".format(
    x1.value, x2.value, x3.value))
