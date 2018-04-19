"""
Solve a capacitated facility location problem, potentially using Benders
decomposition.

The model solved here is

   minimize
       sum(j in locations) fixedCost[j]// open[j] +
       sum(j in locations) sum(i in clients) cost[i][j] * supply[i][j]
   subject to
       sum(j in locations) supply[i][j] == 1                    for each
                                                                client i
       sum(i in clients) supply[i][j] <= capacity[j] * open[j]  for each
                                                                location j
       supply[i][j] in [0,1]
       open[j] in {0, 1}
"""
from __future__ import print_function
import sys
from cvxpy import (Bool, Variable, sum_entries, mul_elemwise, Problem,
                   Minimize, CPLEX)

# capacity   -- a list/array of facility capacity
FIXED_COST = [480, 200, 320, 340, 300]
# cost       -- a matrix for the costs to serve each client by each
#               facility
COST = [[ 24, 74, 31, 51, 84], 
        [ 57, 54, 86, 61, 68],
        [ 57, 67, 29, 91, 71],
        [ 54, 54, 65, 82, 94],
        [ 98, 81, 16, 61, 27],
        [ 13, 92, 34, 94, 87],
        [ 54, 72, 41, 12, 78],
        [ 54, 64, 65, 89, 89]]
# fixedcost  -- a list/array of facility fixed cost
CAPACITY = [ 3, 1, 2, 4, 1]


def facility(use_benders=False):
    """Solve capacitated facility location problem."""
    num_locations = len(FIXED_COST)
    num_clients = len(COST)

    # Create variables. We have variables
    # open_[j]        if location j is open.
    # supply[i][j]]   how much client i is supplied from location j
    open_ = Bool(num_locations)

    constraints = [0 <= open_j for open_j in open_]
    constraints += [open_j <= 1 for open_j in open_]

    supply = Variable(num_clients, num_locations)

    constraints += [0.0 <= supply[i, j]
                    for j in range(num_locations)
                    for i in range(num_clients)]
    constraints += [supply[i, j] <= 1.0
                    for j in range(num_locations)
                    for i in range(num_clients)]

    # Constraint: Each client i must be assigned to exactly one location:
    #   sum(j in nbLocations) supply[i][j] == 1  for each i in nbClients
    constraints += [sum_entries(supply[i, :]) == 1.0
                    for i in range(num_clients)]

    # Constraint: For each location j, the capacity of the location must
    #             be respected:
    #   sum(i in nbClients) supply[i][j] <= capacity[j] * open_[j]
    constraints += [sum_entries(supply[:, j]) - CAPACITY[j] * open_[j] <= 0.0
                    for j in range(num_locations)]

    # Objective: Minimize the sum of fixed costs for using a location
    #            and the costs for serving a client from a specific
    #            location.
    obj = sum_entries(mul_elemwise(FIXED_COST, open_))
    obj += sum_entries(mul_elemwise(COST, supply.T))

    prob = Problem(Minimize(obj), constraints)

    cplex_params = {}
    if use_benders:
        # Set CPXPARAM_Benders_Strategy (1501) to FULL (3)
        cplex_params[1501] = 3

    prob.solve(solver=CPLEX, verbose=True, cplex_params=cplex_params,
               cplex_filename="facility.lp")
    print("Solution status = {0}".format(prob.status))
    print("Optimal value   = {0}".format(prob.value)) 

    for j in [x for x in range(num_locations) if open_[x].value >= 1.0 - 1e-6]:
        print("Facility {0} is open, it serves clients {1}".format(
            j, " ".join([str(i) for i in range(num_clients)
                         if supply[i, j].value >= 1.0 - 1e-6])))


def usage():
    """Prints a usage statement and exits the program."""
    print("""\
Usage: facility.py [options]
   Options are:
   -b solve problem with Benders letting CPLEX do the decomposition
   -d solve problem without using decomposition (default)
 Exiting...
""")
    sys.exit(2)


def main():
    """Handles command line argument parsing."""
    use_benders = False
    for arg in sys.argv[1:]:
        if arg == "-b":
            use_benders = True
        elif arg == "-d":
            use_benders = False
        else:
            usage()
    facility(use_benders)


if __name__ == "__main__":
    main()
