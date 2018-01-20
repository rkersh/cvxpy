from __future__ import print_function

import cvxpy

# Amount of alloy requested
ALLOY_AMT = 71.0

pure1 = cvxpy.Variable(name="Pure1")
pure2 = cvxpy.Variable(name="Pure2")
pure3 = cvxpy.Variable(name="Pure3")
raw1 = cvxpy.Variable(name="Raw1")
raw2 = cvxpy.Variable(name="Raw2")
scrap1 = cvxpy.Variable(name="Scrap1")
scrap2 = cvxpy.Variable(name="Scrap2")
ingots = cvxpy.Variable(name="Ingots")
elemnt1 = cvxpy.Variable(name="Element1")
elemnt2 = cvxpy.Variable(name="Element2")
elemnt3 = cvxpy.Variable(name="Element3")

constraints = [
    0.0 <= pure1, 0.0 <= pure2, 0.0 <= pure3,
    0.0 <= raw1, 0.0 <= raw2,
    0.0 <= scrap1, 0.0 <= scrap2,
    0.0 <= ingots, ingots <= 100000,
    0.0 <= elemnt1, 0.0 <= elemnt2, 0.0 <= elemnt3,
    0.05 * ALLOY_AMT <= elemnt1, elemnt1 <= 0.10 * ALLOY_AMT,
    0.30 * ALLOY_AMT <= elemnt2, elemnt2 <= 0.40 * ALLOY_AMT,
    0.60 * ALLOY_AMT <= elemnt3, elemnt3 <= 0.80 * ALLOY_AMT,
    elemnt1 + elemnt2 + elemnt3 == ALLOY_AMT,
    -elemnt1 + pure1 +
    0.20 * raw1 + 0.01 * raw2 +
    0.00 * scrap1 + 0.01 * scrap2 + 0.10 * ingots == 0.0,
    -elemnt2 + pure2 +
    0.05 * raw1 + 0.00 * raw2 +
    0.60 * scrap1 + 0.00 * scrap2 + 0.45 * ingots == 0.0,
    -elemnt3 + pure3 +
    0.05 * raw1 + 0.30 * raw2 +
    0.40 * scrap1 + 0.70 * scrap2 + 0.45 * ingots == 0.0,
]

# Minimize costs of sources
obj = cvxpy.Minimize(22.0 * pure1 + 10.0 * pure2 + 13.0 * pure3 +
                     6.0 * raw1 + 5.0 * raw2 +
                     7.0 * scrap1 + 8.0 * scrap2 +
                     9.0 * ingots)

prob = cvxpy.Problem(obj, constraints)
prob.solve(solver=cvxpy.CPLEX, verbose=True)

print("Solution status: ", prob.status)
print("Cost: ", prob.value)
print("Pure metal:")
for var in (pure1, pure2, pure3):
    print("{0}) {1}".format(var.name(), var.value))
print("Raw Material:")
for var in (raw1, raw2):
    print("{0}) {1}".format(var.name(), var.value))
print("Scrap:")
for var in (scrap1, scrap2):
    print("{0}) {1}".format(var.name(), var.value))
print("Ingots:")
print("{0}) {1}".format(ingots.name(), ingots.value))
print("Elements:")
for var in (elemnt1, elemnt2, elemnt3):
    print("{0}) {1}".format(var.name(), var.value))
