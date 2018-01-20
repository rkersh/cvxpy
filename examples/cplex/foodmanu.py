from __future__ import print_function
import cvxpy

# Data
cost = [[110.0, 120.0, 130.0, 110.0, 115.0],  # Cost for January
        [130.0, 130.0, 110.0,  90.0, 115.0],  # Cost for February
        [110.0, 140.0, 130.0, 100.0,  95.0],  # Cost for March
        [120.0, 110.0, 120.0, 120.0, 125.0],  # Cost for April
        [100.0, 120.0, 150.0, 110.0, 105.0],  # Cost for May
        [90.0,  100.0, 140.0,  80.0, 135.0]]  # Cost for June
hardness = [8.8, 6.1, 2.0, 4.2, 5.0]  # Hardness of each product
nmonths = len(cost)
nproducts = len(cost[0])
# indices for vegetable oils and non-vegetable oils
v1, v2, o1, o2, o3 = list(range(5))
BIGM = 999.0

# Variables
produce = cvxpy.Variable(nmonths)
use = cvxpy.Variable(nmonths, nproducts)
is_used = cvxpy.Bool(nmonths, nproducts)
buy = cvxpy.Variable(nmonths, nproducts)
store = cvxpy.Variable(nmonths, nproducts)

# Objective
obj = None

# Bounds
constraints = []
for var in produce:
    constraints.append(0.0 <= var)
for i in range(nmonths):
    for j in range(nproducts):
        constraints.append(0.0 <= use[(i,j)])
for i in range(nmonths):
    for j in range(nproducts):
        constraints.append(0.0 <= is_used[(i,j)])
        constraints.append(is_used[(i,j)] <= 1.0)
for i in range(nmonths):
    for j in range(nproducts):
        constraints.append(0.0 <= buy[(i, j)])
for i in range(nmonths):
    for j in range(nproducts):
        if i == nmonths - 1:
            constraints.append(store[(i, j)] == 500.0)
        else:
            constraints.append(0.0 <= store[(i, j)])
            constraints.append(store[(i, j)] <= 1000.0)

# Constraints
for i in range(nmonths):

    # Not more than 200 tons of vegetable oil can be refined
    constraints.append(use[(i, v1)] + use[(i, v2)] <= 200.0)

    # Not more than 250 tons of non-vegetable oil can be refined
    constraints.append(use[(i, o1)] + use[(i, o2)] + use[(i, o3)] <= 250.0)

    # Constraints on food composition
    constraints.append(
        3.0 * produce[i] <=
        cvxpy.sum_entries(cvxpy.mul_elemwise(hardness, use[i, :].T)))

    constraints.append(
        6.0 * produce[i] >=
        cvxpy.sum_entries(cvxpy.mul_elemwise(hardness, use[i, :].T)))

    constraints.append(produce[i] == cvxpy.sum_entries(use[i, :]))

    # Raw oil can be stored for later use
    if i == 0:
        for p in range(nproducts):
            constraints.append(
                500.0 + buy[(i, p)] == use[(i, p)] + store[(i, p)])
    else:
        for p in range(nproducts):
            constraints.append(
                store[(i-1, p)] + buy[(i, p)] == use[(i, p)] + store[(i, p)])

    # Logical constraints:
    # When an oil is used, the quantity must be at least 20 tons.
    # for p in range(nproducts):
    for p in range(nproducts):
        constraints.append(use[i, p] - BIGM * is_used[i, p] <= 0.0)
        constraints.append(use[i, p] - 20.0 * is_used[i, p] >= 0.0)

    # The food cannot use more than 3 oils (or at least two oils must not
    # be used).
    constraints.append(cvxpy.sum_entries(is_used[i, :]) <= 3.0)

    # If products v1 or v2 are used, then product o3 is also used.
    constraints.append(is_used[i, o3] - is_used[i, v1] >= 0.0)
    constraints.append(is_used[i, o3] - is_used[i, v2] >= 0.0)

    # Objective Function
    expr = (150.0 * produce[i])
    expr -= cvxpy.sum_entries(cvxpy.mul_elemwise(cost[i], buy[i, :].T))
    expr -= 5.0 * cvxpy.sum_entries(store[i, :])
    if obj is None:
        obj = expr
    else:
        obj += expr


# Solve and display results
prob = cvxpy.Problem(cvxpy.Maximize(obj), constraints)
prob.solve(solver=cvxpy.CPLEX, verbose=True)

print("Solution status = ", prob.status)
print("Maximum profit = ", prob.value)
for i in range(nmonths):
    print("Month", i)
    print("  . buy   ", end=' ')
    for j in range(nproducts):
        print("{0:15.6f}\t".format(buy[i, j].value), end=' ')
    print()
    print("  . use   ", end=' ')
    for j in range(nproducts):
        print("{0:15.6f}\t".format(use[i, j].value), end=' ')
    print()
    print("  . store ", end=' ')
    for j in range(nproducts):
        print("{0:15.6f}\t".format(store[i, j].value), end=' ')
    print()
