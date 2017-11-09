"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.interface as intf
import cvxpy.settings as s
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from cvxpy.problems.solvers.solver import Solver
from scipy.sparse import dok_matrix


def _select_row(tuple_list, item):
    """Select all tuples from the list where first == item."""
    return [(row, col) for row, col in tuple_list if row == item]


class CPLEX(Solver):
    """An interface for the CPLEX solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Map of CPLEX status to CVXPY status.
    # RPK: Map stati for OPTIMAL_INACCURATE, INFEASIBLE_INACCURATE,
    #      UNBOUNDED_INACCURATE
    STATUS_MAP_LP = {1: s.OPTIMAL,  # CPX_STAT_OPTIMAL
                     2: s.UNBOUNDED,  # CPX_STAT_UNBOUNDED
                     3: s.INFEASIBLE,  # CPX_STAT_INFEASIBLE
                     }

    # RPK: Map stati for OPTIMAL_INACCURATE, INFEASIBLE_INACCURATE,
    #      UNBOUNDED_INACCURATE
    STATUS_MAP_MIP = {101: s.OPTIMAL,  # CPXMIP_OPTIMAL
                      102: s.OPTIMAL,  # CPXMIP_OPTIMAL_TOL
                      103: s.INFEASIBLE,  # CPXMIP_INFEASIBLE
                      118: s.UNBOUNDED,  # CPXMIP_UNBOUNDED
                      }

    def name(self):
        """The name of the solver.
        """
        return s.CPLEX

    def import_solver(self):
        """Imports the solver.
        """
        import cplex
        # RPK: Not sure if this is necc.
        cplex  # For flake8

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTF

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

    @staticmethod
    def _param_in_constr(constraints):
        """Do any of the constraints contain parameters?
        """
        for constr in constraints:
            if len(lu.get_expr_params(constr.expr)) > 0:
                return True
        return False

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cplex

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        A = dok_matrix(data[s.A])
        # Save the dok_matrix.
        data[s.A] = A

        n = c.shape[0]

        solver_cache = cached_data[self.name()]

        # TODO warmstart with SOC constraints.
        if warm_start and solver_cache.prev_result is not None \
           and len(data[s.DIMS][s.SOC_DIM]) == 0:
            model = solver_cache.prev_result["model"]
            variables = solver_cache.prev_result["variables"]
            cpx_constrs = solver_cache.prev_result["cpx_constrs"]
            c_prev = solver_cache.prev_result["c"]
            A_prev = solver_cache.prev_result["A"]
            b_prev = solver_cache.prev_result["b"]

            # If there is a parameter in the objective, it may have changed.
            if len(lu.get_expr_params(objective)) > 0:
                c_diff = c - c_prev

                I_unique = list(set(np.where(c_diff)[0]))

                for i in I_unique:
                    model.objective.set_linear(variables[i], c[i])
            else:
                # Stay consistent with CPLEX's representation of the problem
                c = c_prev

            # Get equality and inequality constraints.
            sym_data = self.get_sym_data(objective, constraints, cached_data)
            all_constrs, _, _ = self.split_constr(sym_data.constr_map)

            # If there is a parameter in the constraints,
            # A or b may have changed.
            if self._param_in_constr(all_constrs):
                A_diff = dok_matrix(A - A_prev)
                b_diff = b - b_prev

                # Figure out which rows of A and elements of b have changed
                try:
                    I, _ = zip(*[x for x in A_diff.keys()])
                except ValueError:
                    I = []
                I_unique = list(set(I) | set(np.where(b_diff)[0]))

                # RPK: Use a different data structure?
                nonzero_locs = [x for x in A.keys()]

                # Update locations which have changed
                for i in I_unique:

                    # Remove old constraint if it exists
                    if cpx_constrs[i] is not None:
                        # Disable the old constraint by setting all
                        # coefficients and rhs to zero. This way we don't
                        # have to worry about indices needing to shift.
                        # RPK: Correct?
                        tmp = model.linear_constraints.get_rows(cpx_constrs[i])
                        model.linear_constraints.set_linear_components(
                            cpx_constrs[i],
                            cplex.SparsePair(ind=tmp.ind, val=[0.0]*len(tmp.ind)))
                        model.linear_constraints.set_rhs(cpx_constrs[i], 0.0)
                        cpx_constrs[i] = None

                    # Add new constraint
                    nonzero_loc = _select_row(nonzero_locs, i)
                    if nonzero_loc:
                        ind, val = [], []
                        for row, col in nonzero_loc:
                            ind.append(variables[col])
                            val.append(A[(row, col)])
                        if i < data[s.DIMS][s.EQ_DIM]:
                            ctype = "E"
                        else:
                            assert data[s.DIMS][s.EQ_DIM] <= i \
                                < data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]
                            ctype = "L"
                        cpx_constrs[i] = list(model.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                            senses=ctype,
                            rhs=[b[i]]))[0]

            else:
                # Stay consistent with CPLEX's representation of the problem
                A = A_prev
                b = b_prev

        else:
            model = cplex.Cplex()
            variables = []
            vtype = []
            if self.is_mip(data):
                for i in range(n):
                    # Set variable type.
                    if i in data[s.BOOL_IDX]:
                        vtype.append('B')
                    elif i in data[s.INT_IDX]:
                        vtype.append('I')
                    else:
                        vtype.append('C')
            else:
                # If we specify types (even with 'C'), then the problem will
                # be interpreted as a MIP. Leaving vtype as an empty list
                # here, will ensure that the problem type remains an LP.
                pass
            # Add the variables in a batch
            variables = list(model.variables.add(
                obj=[c[i] for i in range(n)],
                lb=[-cplex.infinity]*n,  # default LB is 0
                ub=[cplex.infinity]*n,
                types="".join(vtype),
                names=["x_%d" % i for i in range(n)]))

            nonzero_locs = [x for x in A.keys()]  # RPK: Different data structure?
            eq_constrs = self.add_model_lin_constr(model, variables,
                                                   range(data[s.DIMS][s.EQ_DIM]),
                                                   'E', nonzero_locs, A, b)
            leq_start = data[s.DIMS][s.EQ_DIM]
            leq_end = data[s.DIMS][s.EQ_DIM] + data[s.DIMS][s.LEQ_DIM]
            ineq_constrs = self.add_model_lin_constr(model, variables,
                                                     range(leq_start, leq_end),
                                                     'L', nonzero_locs, A, b)
            soc_start = leq_end
            soc_constrs = []
            new_leq_constrs = []
            for constr_len in data[s.DIMS][s.SOC_DIM]:
                soc_end = soc_start + constr_len
                soc_constr, new_leq, new_vars = self.add_model_soc_constr(
                    model, variables, range(soc_start, soc_end),
                    nonzero_locs, A, b
                )
                soc_constrs.append(soc_constr)
                new_leq_constrs += new_leq
                variables += new_vars
                soc_start += constr_len

            cpx_constrs = eq_constrs + ineq_constrs + \
                soc_constrs + new_leq_constrs

        # Set verbosity and other parameters
        # model.setParam("OutputFlag", verbose)  # RPK: ?
        # TODO user option to not compute duals.
        # model.setParam("QCPDual", True)  # RPK: ?

        for key, value in solver_opts.items():
            # model.setParam(key, value)  # RPK: ?
            pass  # RPK: ?

        results_dict = {}
        start_time = model.get_time()
        solve_time = -1
        try:
            model.solve()
            solve_time = model.get_time() - start_time
            results_dict["primal objective"] = model.solution.get_objective_value()
            results_dict["x"] = np.array(model.solution.get_values(variables))

            if self.is_mip(data):
                results_dict["status"] = self.STATUS_MAP_MIP.get(
                    model.solution.get_status(), s.SOLVER_ERROR)
            else:
                # Only add duals if not a MIP.
                vals = []
                if len(cpx_constrs) > 0:
                    vals.extend(model.solution.get_dual_values(
                        [c for c in cpx_constrs if c is not None]))
                # RPK: FIXME (use method in qcpdual.py to calculate qcp duals)
                #vals.extend(for soc_constr)
                #vals.extend(for new_leq_constr)
                results_dict["y"] = -np.array(vals)
                results_dict["status"] = self.STATUS_MAP_LP.get(
                    model.solution.get_status(), s.SOLVER_ERROR)
        except:
            if solve_time < 0.0:
                solve_time = model.get_time() - start_time
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model
        results_dict["variables"] = variables
        results_dict["cpx_constrs"] = cpx_constrs
        results_dict[s.SOLVE_TIME] = solve_time

        return self.format_results(results_dict, data, cached_data)

    def add_model_lin_constr(self, model, variables,
                             rows, ctype,
                             nonzero_locs, mat, vec):
        """Adds EQ/LEQ constraints to the model using the data from mat and vec.

        Parameters
        ----------
        model : CPLEX model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        ctype : CPLEX constraint type
            The type of constraint.
        nonzero_locs : list of tuples
            A list of all the nonzero locations.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        list
            A list of constraints.
        """
        import cplex
        constr = []
        for i in rows:
            ind, val = [], []
            for row, col in _select_row(nonzero_locs, i):
                ind.append(variables[col])
                val.append(mat[(row, col)])
            # Ignore empty constraints.
            if len(ind) > 0:
                # RPK: Would be faster if added in a batch.
                constr.extend(list(
                    model.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                        senses=ctype,
                        rhs=[vec[i]])))
            else:
                constr.append(None)
        return constr

    def add_model_soc_constr(self, model, variables,
                             rows, nonzero_locs, mat, vec):
        """Adds SOC constraint to the model using the data from mat and vec.

        Parameters
        ----------
        model : CPLEX model
            The problem model.
        variables : list
            The problem variables.
        rows : range
            The rows to be constrained.
        nonzero_locs : list of tuples
            A list of all the nonzero locations.
        mat : SciPy COO matrix
            The matrix representing the constraints.
        vec : NDArray
            The constant part of the constraints.

        Returns
        -------
        tuple
            A tuple of (QConstr, list of Constr, and list of variables).
        """
        import cplex
        # Assume first expression (i.e. t) is nonzero.
        lin_expr_list = []
        soc_vars = []
        for i in rows:
            ind, val = [], []
            for row, col in _select_row(nonzero_locs, i):
                ind.append(variables[col])
                val.append(mat[(row, col)])
            # Ignore empty constraints.
            if len(ind) > 0:
                lin_expr_list.append((ind, val))
            else:
                lin_expr_list.append(None)

        # Make a variable and equality constraint for each term.
        soc_vars, is_first = [], True
        for i in rows:
            if is_first:
                lb = [0.0]
                names = ["soc_t_%d" % i]
                is_first = False
            else:
                lb = [-cplex.infinity]
                names = ["soc_x_%d" % i]
            soc_vars.extend(list(model.variables.add(
                obj=[0],
                lb=lb,
                ub=[cplex.infinity],
                types="",
                names=names)))

        new_lin_constrs = []
        for i, expr in enumerate(lin_expr_list):
            if expr is None:
                ind = [soc_vars[i]]
                val = [1.0]
            else:
                ind, val = expr
                ind.append(soc_vars[i])
                val.append(1.0)
            new_lin_constrs.extend(list(
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=ind, val=val)],
                    senses="E",
                    rhs=[-vec[i]])))

        assert len(soc_vars) > 0
        qconstr = model.quadratic_constraints.add(
            lin_expr=cplex.SparsePair(ind=[], val=[]),
            quad_expr=cplex.SparseTriple(
                ind1=soc_vars,
                ind2=soc_vars,
                val=[-1.0] + [1.0] * (len(soc_vars) - 1)),
            sense="L",
            rhs=0.0,
            name="")
        return (qconstr, new_lin_constrs, soc_vars)

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        dims = data[s.DIMS]
        if results_dict["status"] != s.SOLVER_ERROR:
            solver_cache = cached_data[self.name()]
            solver_cache.prev_result = {
                "model": results_dict["model"],
                "variables": results_dict["variables"],
                "cpx_constrs": results_dict["cpx_constrs"],
                "c": data[s.C],
                "A": data[s.A],
                "b": data[s.B],
            }
        new_results = {}
        new_results[s.STATUS] = results_dict['status']
        new_results[s.SOLVE_TIME] = results_dict[s.SOLVE_TIME]
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict['x']
            if not self.is_mip(data):
                new_results[s.EQ_DUAL] = results_dict["y"][0:dims[s.EQ_DIM]]
                new_results[s.INEQ_DUAL] = results_dict["y"][dims[s.EQ_DIM]:]

        return new_results
