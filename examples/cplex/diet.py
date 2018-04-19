"""
Mimimize the cost of a diet subject to nutritional constraints.
"""
from __future__ import print_function
from cvxpy import (Variable, Problem, Minimize, mul_elemwise, CPLEX,
                   sum_entries)

FOOD_COST = [1.84, 2.19, 1.84, 1.44, 2.29, 0.77, 1.29, 0.6, 0.72]
FOOD_MIN = [0, 0, 0, 0, 0, 0, 0, 0, 0]
FOOD_MAX = [10, 10, 10, 10, 10, 10, 10, 10, 10]
NUTR_MIN = [2000, 350, 55, 100, 100, 100, 100]
NUTR_MAX = [9999, 375, 9999, 9999, 9999, 9999, 9999]
NUTR_PER = [[510, 370, 500, 370, 400, 220, 345, 110, 80],
            [34, 35, 42, 38, 42, 26, 27, 12, 20],
            [28, 24, 25, 14, 31, 3,  15, 9,  1],
            [15, 15, 6,  2,  8 , 0,  4 , 10, 2],
            [6,  10, 2,  0,  15, 15, 0 , 4,  120],
            [30, 20, 25, 15, 15, 0,  20, 30, 2],
            [20, 20, 20, 10, 8 , 2,  15, 0,  2]]


def main():
    num_foods = len(FOOD_COST)

    # add variables to decide how much of each type of food to buy
    x = Variable(num_foods)

    constraints = [food_min <= x_i
                   for x_i, food_min
                   in zip(x, FOOD_MIN)]
    constraints += [x_i <= food_max
                   for x_i, food_max
                   in zip(x, FOOD_MAX)]

    # add constraints to specify limits for each of the nutrients
    for nutr_min, nutr_per, nutr_max in zip(NUTR_MIN, NUTR_PER, NUTR_MAX):
        expr = sum_entries(mul_elemwise(nutr_per, x))
        constraints += [nutr_min <= expr,
                        expr <= nutr_max]

    # we want to minimize costs
    prob = Problem(Minimize(sum_entries(mul_elemwise(FOOD_COST, x))),
                   constraints)
    prob.solve(solver=CPLEX, verbose=True, cplex_filename="diet.lp")

    print("Solution status = {0}".format(prob.status))
    print("Objective value = {0}".format(prob.value))
    for idx, x_i in enumerate(x):
        print("Buy {0} = {1:17.10g}".format(idx, x_i.value))


if __name__ == "__main__":
    main()
