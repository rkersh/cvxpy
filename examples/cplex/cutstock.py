"""
Using column generation

Problem Description:

The cutting stock problem in this example is sometimes known in math
programming terms as a knapsack problem with reduced costs in the
objective function. Generally, a cutting stock problem begins with a
supply of rolls of material of fixed length (the stock). Strips are cut
from these rolls. All the strips cut from one roll are known together as
a pattern. The point of this example is to use as few rolls of stock as
possible to satisfy some specified demand of strips. By convention, it is
assumed that only one pattern is laid out across the stock; consequently,
only one dimension (the width) of each roll of stock is important.
"""
from __future__ import print_function
from cvxpy import (Variable, Minimize, sum_entries, Int, mul_elemwise, Problem,
                   CPLEX)

# Data
EPSILON = 1.0e-6
WIDTH = 115
SIZE = [25, 40, 50, 55, 70]
LENSIZE = len(SIZE)
AMOUNT = [50, 36, 24, 8, 30]

def report1(cutprob, cutvars):
    """Print a report about the current solution in the cutting
    optimization problem given by the cut argument.
    """
    print()
    print("Using {0} rolls".format(cutprob.value))
    print()
    for idx, var in enumerate(cutvars):
        print("  Cut {0} = {1}".format(idx, var.value))
    print()
    for idx, constr in enumerate(cutprob.constraints[-LENSIZE:]):
        print("  Fill {0} = {1}".format(idx, constr.dual_value))
    print()

def report2(patprob, patvars):
    """Print a report about the current solution in the pattern generation
    problem given by the pat argument. The use argument specifies the indices
    of variables that shall appear in the report.
    """
    print()
    print("Reduced cost is {0}".format(patprob.value))
    print()
    if patprob.value <= -EPSILON:
        for idx, var in enumerate(patvars):
            print("  Use {0} = {1}".format(idx, var.value))
        print()

def report3(cutprob, cutvars):
    """Print the final report for the current solution in the cutting
    optimization problem given by the cut argument.
    """
    print()
    print("Best integer solution uses {0} rolls".format(cutprob.value))
    print()
    for idx, var in enumerate(cutvars):
        print("  Cut {0} = {1}".format(idx, var.value))

def create_cut_constr(cutvars, cutmatind, cutmatval):
    assert len(cutmatind) == len(cutmatval)
    assert len(cutmatind) == len(AMOUNT)
    constr = []
    for var in cutvars:
        constr.append(0.0 <= var)
    for i in range(len(cutmatind)):
        con = None
        for ind, val in zip(cutmatind[i], cutmatval[i]):
            expr = (val * ind)
            if con is None:
                con = expr
            else:
                con += expr
        constr.append(con >= AMOUNT[i])
    return constr

# Setup cutting optimization (master) problem.
# This is the problem to which columns will be added in the loop below.

# Cut Variables
cutvars = []
for i in range(LENSIZE):
    cutvars.append(Variable())

# Cut Constraints
cutmatind = []
cutmatval = []
for var, size, amount in zip(cutvars, SIZE, AMOUNT):
    coef = WIDTH // size
    cutmatind.append([var])
    cutmatval.append([coef])
cutconstr = create_cut_constr(cutvars, cutmatind, cutmatval)

# Cut Objective
cutobj = Minimize(sum(cutvars))
cutprob = Problem(cutobj, cutconstr)

# Setup pattern generation (worker) problem.
# The constraints and variables in this problem always stay the same but
# the objective function will change during the column generation loop.

# Pat Variables
patvars = Int(LENSIZE)
patobjvar = Variable()

# Pat Bounds
patconstr = [patobjvar == 1.0,]
for var in patvars:
    patconstr.append(0.0 <= var)

# Pat Constraints
# Single constraint: total size must not exceed the width.
patconstr.append(sum_entries(mul_elemwise(SIZE, patvars)) <= WIDTH)

# Pat Objective
patprob = Problem(Minimize(patobjvar), patconstr)

# Column generation procedure
while True:
    # Optimize over current patterns
    cutprob.solve(solver=CPLEX, verbose=True)
    report1(cutprob, cutvars)

    # Find and add new pattern. The objective function of the
    # worker problem is constructed from the dual values of the
    # constraints of the master problem.
    price = [-d.dual_value for d in cutprob.constraints[-LENSIZE:]]
    patprob = Problem(
        Minimize(sum_entries(mul_elemwise(price, patvars)) + patobjvar),
        patprob.constraints)
    patprob.solve(solver=CPLEX)
    report2(patprob, patvars)

    # If reduced cost (worker problem objective function value) is
    # non-negative we are optimal. Otherwise we found a new column
    # to be added. Coefficients of the new column are given by the
    # optimal solution vector to the worker problem.
    if patprob.value > -EPSILON:
        break
    newpat = [v.value for v in patvars]

    # The new pattern constitutes a new variable in the cutting
    # optimization problem. Create that variable and add it to all
    # constraints with the coefficients read from the optimal solution
    # of the pattern generation problem.
    newcol = Variable()
    cutvars.append(newcol)
    cutprob.objective += Minimize(newcol)
    for i, coef in enumerate(newpat):
        cutmatind[i].append(newcol)
        cutmatval[i].append(coef)
    cutprob.constraints = create_cut_constr(cutvars, cutmatind, cutmatval)

# Perform a final solve on the cutting optimization problem.
# Turn all variables into integers before doing that.
for var in cutvars:
    cutprob.constraints.append(var == Int())
cutprob.solve(solver=CPLEX, verbose=True)
report3(cutprob, cutvars)
print("Solution status = ", cutprob.status)
