"""
Sports League Scheduling

How can a sports league schedule matches between teams in different
divisions such that the teams play each other the appropriate number of
times and maximize the objective of scheduling intradivision matches as
late as possible in the season?

A sports league with two divisions needs to schedule games such that each
team plays every team within its division a specified number of times and
plays every team in the other division a specified number of times. Each
week, a team plays exactly one game. The preference is for
intradivisional matches to be held as late as possible in the season. To
model this preference, there is an incentive for intradivisional matches;
this incentive increases in a non-linear manner by week. The problem
consists of assigning an opponent to each team each week in order to
maximize the total of the incentives.
"""
from collections import namedtuple

from cvxpy import Bool, sum_entries, Problem, Maximize, CPLEX


# ----------------------------------------------------------------------------
# Initialize the problem data
# ----------------------------------------------------------------------------
nbs = (8, 1, 1)

team_div1 = {"Baltimore Ravens", "Cincinnati Bengals", "Cleveland Browns",
             "Pittsburgh Steelers", "Houston Texans", "Indianapolis Colts",
             "Jacksonville Jaguars", "Tennessee Titans", "Buffalo Bills",
             "Miami Dolphins", "New England Patriots", "New York Jets",
             "Denver Broncos", "Kansas City Chiefs", "Oakland Raiders",
             "San Diego Chargers"}

team_div2 = {"Chicago Bears", "Detroit Lions", "Green Bay Packers",
             "Minnesota Vikings", "Atlanta Falcons", "Carolina Panthers",
             "New Orleans Saints", "Tampa Bay Buccaneers", "Dallas Cowboys",
             "New York Giants", "Philadelphia Eagles", "Washington Redskins",
             "Arizona Cardinals", "San Francisco 49ers", "Seattle Seahawks",
             "St. Louis Rams"}

Match = namedtuple("Matches", ["team1", "team2", "is_divisional"])

# a named tuple to store solution
TSolution = namedtuple("TSolution", ["week", "is_divisional", "team1", "team2"])


def build_sports():
    """Build the model."""
    print("* building sport scheduling model instance")
    nb_teams_in_division, nb_intra_divisional, nb_inter_divisional = nbs
    assert len(team_div1) == len(team_div2)
    teams = list(team_div1 | team_div2)
    # team index ranges from 1 to 2N
    team_range = range(1, 2 * nb_teams_in_division + 1)

    # Calculate the number of weeks necessary.
    nb_weeks = ((nb_teams_in_division - 1) * nb_intra_divisional +
                nb_teams_in_division * nb_inter_divisional)
    weeks = range(1, nb_weeks + 1)

    print("{0} games, {1} intradivisional, {2} interdivisional"
          .format(nb_weeks, (nb_teams_in_division - 1) * nb_intra_divisional,
                  nb_teams_in_division * nb_inter_divisional))

    # Season is split into two halves.
    first_half_weeks = range(1, nb_weeks // 2 + 1)
    nb_first_half_games = nb_weeks // 3

    # All possible matches (pairings) and whether of not each is intradivisional.
    matches = [Match(t1, t2,
                     1 if (t2 <= nb_teams_in_division or
                           t1 > nb_teams_in_division) else 0)
               for t1 in team_range for t2 in team_range if t1 < t2]
    #matches = matches
    # Number of games to play between pairs depends on
    # whether the pairing is intradivisional or not.
    nb_play = {m: nb_intra_divisional
               if m.is_divisional == 1
               else nb_inter_divisional
               for m in matches}

    plays = Bool(len(matches), len(weeks))
    constraints = []

    # Implicit bounds:
    # constraints += [0.0 <= plays[m, w]
    #                for m in range(len(matches))
    #                for w in range(len(weeks))]
    # constraints += [plays[m, w] <= 1.0
    #                 for m in range(len(matches))
    #                 for w in range(len(weeks))]

    constraints += [sum_entries(plays[m_idx, :]) == nb_play[m]
                    for m_idx, m in enumerate(matches)]

    # Each team must play exactly once in a week.
    for w_idx, w in enumerate(weeks):
        for t in team_range:
            constraints += [sum([plays[m_idx, w_idx]
                                 for m_idx, m in enumerate(matches)
                                 if m.team1 == t or m.team2 == t]) == 1]

    # Games between the same teams cannot be on successive weeks.
    constraints += [plays[m_idx, w_idx] + plays[m_idx, w_idx + 1] <= 1
                    for w_idx, w in enumerate(weeks)
                    for m_idx, m in enumerate(matches)
                    if w < nb_weeks]

    # Some intradivisional games should be in the first half.
    for t in team_range:
        constraints += [sum([plays[m_idx, w_idx]
                             for w_idx, w in enumerate(first_half_weeks)
                             for m_idx, m in enumerate(matches)
                             if m.is_divisional == 1 and (m.team1 == t or m.team2 == t)])
                        >= nb_first_half_games]

    # postpone divisional matches as much as possible
    # we weight each play variable with the square of w.
    obj = sum([sum_entries(plays[m_idx, w_idx] * w * w)
               for w_idx, w in enumerate(weeks)
               for m_idx, m in enumerate(matches)
               if m.is_divisional])

    mdl = Problem(Maximize(obj), constraints)
    mdl.teams = teams
    mdl.weeks = weeks
    mdl.matches = matches
    mdl.plays = plays
    return mdl


def print_sports_solution(mdl):
    # iterate with weeks first
    solution = [TSolution(w, m.is_divisional, mdl.teams[m.team1], mdl.teams[m.team2])
                for w_idx, w in enumerate(mdl.weeks)
                for m_idx, m in enumerate(mdl.matches)
                if mdl.plays[m_idx, w_idx].value >= 1.0 - 1e-6]

    currweek = 0
    print("Intradivisional games are marked with a *")
    for s in solution:
        # assume records are sorted by increasing week indices.
        if s.week != currweek:
            currweek = s.week
            print(" == == == == == == == == == == == == == == == == ")
            print("On week %d" % currweek)

        print("    {0:s}{1} will meet the {2}".format(
            "*" if s.is_divisional else "", s.team1, s.team2))


def print_metrics(mdl):
    metrics = mdl.size_metrics
    nb_cons = metrics.num_scalar_eq_constr + metrics.num_scalar_leq_constr
    print("""Model:
 - number of variables: {0}
 - number of non-zeros: {1}
 - number of constraints: {2}
   - eq={3}, leq={4}
* Status: {5}
* Objective: {6}
""".format(metrics.num_scalar_variables,
           metrics.num_scalar_data,
           nb_cons,
           metrics.num_scalar_eq_constr,
           metrics.num_scalar_leq_constr,
           mdl.status, mdl.value))


def main():
    """Solve the model and display the result."""
    # Build the model
    model = build_sports()
    # Solve the model.
    model.solve(solver=CPLEX, verbose=False, cplex_filename="sport_scheduling.lp")
    print_metrics(model)
    print_sports_solution(model)


if __name__ == '__main__':
    main()
    
