"""
Entering and optimizing a mixed integer programming problem.

The MIP problem solved in this example is:

  Maximize  x1 + 2 x2 + 3 x3 + x4
  Subject to
     - x1 +   x2 + x3 + 10 x4 <= 20
       x1 - 3 x2 + x3         <= 30
              x2      - 3.5x4  = 0
  Bounds
       0 <= x1 <= 40
       0 <= x2
       0 <= x3
       2 <= x4 <= 3
  Integers
      x4
"""
from __future__ import print_function
import cvxpy
from cvxpy import (Variable, Maximize, Problem, CPLEX)


INF = 1.0e+20
NUMCOLS = 4
OBJ = [1.0, 2.0, 3.0, 1.0]
LB = [0.0, 0.0, 0.0, 2.0]
UB = [40.0, INF, INF, 3.0]

x = Variable(NUMCOLS)

constr = []
# Set bounds
for lb, ub, var in zip(LB, UB, x):
    constr.append(lb <= var)
    constr.append(var <= ub)

# Specify the constraints
constr.append(-x[0] + x[1] + x[2] + 10 * x[3] <= 20.0)
constr.append(x[0] - 3.0 * x[1] + x[2] <= 30.0)
constr.append(x[1] - 3.5 * x[3] == 0.0)

# Force x[3] to be integral. This is equivalent to changing the type
# of the variable from continous to integer.
constr.append(x[3] == Variable(integer=True))

# Objective
obj = Maximize(cvxpy.sum(cvxpy.multiply(OBJ, x)))

# Solve and display results
prob = Problem(obj, constr)
prob.solve(solver=CPLEX, verbose=True)

print("Solution status = ", prob.status)
print("Solution value  = ", prob.value)
for idx, var in enumerate(x):
    print("Column {0}: Value = {1:10f}".format(idx, var.value))
