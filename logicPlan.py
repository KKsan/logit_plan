# coding=utf-8
# logicPlan.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game
import time

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    return logic.conjoin([(A | B), (~A % (~B | C)), logic.disjoin([~A, ~B, C])])
    # util.raiseNotDefined()


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"
    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')

    S1 = (C % (B | D))
    S2 = (A >> (~B & ~D))
    S3 = (~(B & ~C) >> A)
    S4 = (~D >> C)

    return logic.conjoin([S1, S2, S3, S4])
    # util.raiseNotDefined()


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive and time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    # ((WumpusAlive[1] <=> ((WumpusAlive[0] & ~WumpusKilled[0]) | (~WumpusAlive[0] & WumpusBorn[0]))) & ~(WumpusAlive[0] & WumpusBorn[0]) & WumpusBorn[0])
    W_alive_1 = logic.PropSymbolExpr('WumpusAlive', 1)
    W_alive_0 = logic.PropSymbolExpr('WumpusAlive', 0)
    W_born_0 = logic.PropSymbolExpr('WumpusBorn', 0)
    W_killed_0 = logic.PropSymbolExpr('WumpusKilled', 0)
    S1 = (W_alive_1 % ((W_alive_0 & ~W_killed_0) | (~W_alive_0 & W_born_0)))
    S2 = ~(W_alive_0 & W_born_0)
    S3 = W_born_0
    return logic.conjoin([S1, S2, S3])
    # util.raiseNotDefined()


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"

    return logic.pycoSAT(logic.to_cnf(sentence))
    # util.raiseNotDefined()


def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single 
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    expr = []
    for i in range(0, len(literals)):
        expr.append(literals[i])
    return logic.disjoin(expr)


def atMostOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    pairs = []
    for i in range(0, len(literals)):
        for j in range(i + 1, len(literals)):
            sub_statement = (~literals[i] | ~literals[j])
            pairs.append(sub_statement)
    return logic.conjoin(pairs)


def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    :type model: object
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    return_actions = []
    actions_d = {}
    for key in model.keys():
        parsed = logic.PropSymbolExpr.parseExpr(key)
        name = parsed[0]
        index = parsed[1]
        if (name in actions) & model[key]:
            actions_d[int(index)] = name
    for i in sorted(actions_d.keys()):
        return_actions.append(actions_d[i])
    return return_actions


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    """
    "*** YOUR CODE HERE ***"
    adjacencies = [(x + 1, y, "West"), (x - 1, y, "East"), (x, y + 1, "South"), (x, y - 1, "North")]
    states = []
    for tup in adjacencies:
        if not walls_grid[tup[0]][tup[1]]:
            pacstate = logic.PropSymbolExpr(pacman_str, tup[0], tup[1], t - 1)
            states.append((pacstate & logic.PropSymbolExpr(tup[2], t - 1)))
    if states:
        return logic.Expr('<=>', logic.PropSymbolExpr(pacman_str, x, y, t), logic.disjoin(states))
    else:
        return logic.PropSymbolExpr(pacman_str, x, y, t)


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    legal_actions = ["North", "South", "East", "West"]

    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    goal = problem.getGoalState()
    max_steps = 50

    memoized = {}

    for t in range(1, max_steps + 1):
        exprs = []
        # Initial constraints

        # has to be at goal at t
        g = logic.PropSymbolExpr(pacman_str, goal[0], goal[1], t)
        exprs.append(g)

        # Has to be at start_state at 0
        s = logic.PropSymbolExpr(pacman_str, start_state[0], start_state[1], 0)
        # Can only be at exactly one spot
        init_states = []
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                init_states.append(logic.PropSymbolExpr(pacman_str, x, y, 0))
        one_state = exactlyOne(init_states)
        exprs.append(s)
        exprs.append(one_state)
        # get all previous SSAs before t
        ssas = []
        for i in range(1, t + 1):
            # can take exactly one action at time t
            acts = []

            for a in legal_actions:
                acts.append(logic.PropSymbolExpr(a, i - 1))
            one_act = exactlyOne(acts)
            exprs.append(one_act)

            # get SSA of all location and put them into cnf
            for x in range(1, width + 1):
                for y in range(1, height + 1):
                    if memoized.has_key((x, y, i)):
                        ssa = memoized[(x, y, i)]
                    else:
                        ssa = pacmanSuccessorStateAxioms(x, y, i, walls)
                        memoized[(x, y, i)] = ssa
                    ssas.append(ssa)
                    # memoized[i] = ssas

        # check if the model is true repeatedly for each t to get optimal solution
        # exprs.append(ssas)
        ssas_dis = logic.conjoin(ssas)
        exprs_con = logic.conjoin(exprs)
        sent = logic.conjoin([ssas_dis, exprs_con])
        model = findModel(sent)
        # print model
        if model:
            action_seq = extractActionSequence(model, legal_actions)
            return action_seq
    return []


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    legal_actions = ["North", "South", "East", "West"]

    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    pac_state = start_state[0]
    food_state = start_state[1]
    max_steps = 50
    memoized = {}
    foods = food_state.asList()
    for t in range(1, max_steps + 1):
        # print t
        exprs = []
        # Initial constraints
        # Has to be at start_state at 0
        s = logic.PropSymbolExpr(pacman_str, pac_state[0], pac_state[1], 0)
        # Can only be at exactly one spot
        init_states = []
        # init_foods = []
        for x in range(1, width + 1):
            for y in range(1, height + 1):
                init_states.append(logic.PropSymbolExpr(pacman_str, x, y, 0))

        exprs.append(s)
        exprs.append(exactlyOne(init_states))
        # exprs.append(init_foods)
        # get all previous SSAs before t
        ssas = []
        goals = []
        for f in foods:
            pos = []
            for i in range(0, t + 1):
                pos.append(logic.PropSymbolExpr(pacman_str, f[0], f[1], i))
            goals.append(atLeastOne(pos))
        food_goals = logic.conjoin(goals)

        for i in range(1, t + 1):
            # can take exactly one action at time t
            acts = []

            for a in legal_actions:
                acts.append(logic.PropSymbolExpr(a, i - 1))
            one_act = exactlyOne(acts)
            exprs.append(one_act)

            # get SSA of all location and put them into cnf
            for x in range(1, width + 1):
                for y in range(1, height + 1):
                    if memoized.has_key((x, y, i)):
                        ssa = memoized[(x, y, i)]
                    else:
                        ssa = pacmanSuccessorStateAxioms(x, y, i, walls)
                        # ssaf = (~logic.PropSymbolExpr('F', x, y, i) % ssa)
                        memoized[(x, y, i)] = ssa
                    ssas.append(ssa)

                    # memoized[i] = ssas

        # check if the model is true repeatedly for each t to get optimal solution
        # exprs.append(ssas)
        exprs.append(food_goals)
        ssas_dis = logic.conjoin(ssas)
        exprs_con = logic.conjoin(exprs)
        sent = logic.conjoin([ssas_dis, exprs_con])
        model = findModel(sent)
        # print model
        if model:
            action_seq = extractActionSequence(model, legal_actions)
            return action_seq
    return []


def ghostPositionSuccessorStateAxioms(x, y, t, ghost_num, walls_grid):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    Current <==> (causes to stay) | (causes of current)
    GE is going east, ~GE is going west 
    :rtype : object
    """
    pos_str = ghost_pos_str + str(ghost_num)
    east_str = ghost_east_str + str(ghost_num)

    "*** YOUR CODE HERE ***"
    adjacencies = [(x + 1, y, "West"), (x - 1, y, "East")]
    states = []
    for tup in adjacencies:
        if not walls_grid[tup[0]][tup[1]]:
            ghoststate = logic.PropSymbolExpr(pos_str, tup[0], tup[1], t - 1)
            if tup[2] == "West":
                states.append((ghoststate & ~logic.PropSymbolExpr(east_str, t - 1)))
            if tup[2] == "East":
                states.append((ghoststate & logic.PropSymbolExpr(east_str, t - 1)))
    if states:
        prevs = logic.disjoin(states)
        ssa = logic.Expr('<=>', logic.PropSymbolExpr(pos_str, x, y, t), prevs)

        return ssa
    else:
        return logic.Expr('<=>', logic.PropSymbolExpr(pos_str, x, y, t), logic.PropSymbolExpr(pos_str, x, y, t - 1))


def ghostDirectionSuccessorStateAxioms(t, ghost_num, blocked_west_positions, blocked_east_positions):
    """
    Successor state axiom for patrolling ghost direction state (t) (from t-1).
    west or east walls.
    t时刻往东当且仅当t-1时刻往东且t时刻不在blockeast的位置集或t-1时刻向西且t时刻在blockwest的位置集
    """
    pos_str = ghost_pos_str + str(ghost_num)
    east_str = ghost_east_str + str(ghost_num)
    "*** YOUR CODE HERE ***"
    clauses = []
    for pos in blocked_west_positions:
        clause_right_1 = (logic.PropSymbolExpr(pos_str, pos[0], pos[1], t) & ~logic.PropSymbolExpr(east_str, t - 1))
        pos_east = []
        for pos2 in blocked_east_positions:
            pos_east.append(~logic.PropSymbolExpr(pos_str, pos2[0], pos2[1], t))

        clause_right_2 = logic.PropSymbolExpr(east_str, t - 1)
        if pos_east:
            clause_right_2 = (clause_right_2 & logic.conjoin(pos_east))
        clauses.append((clause_right_1 | clause_right_2))
    clause_left = logic.PropSymbolExpr(east_str, t)
    change = logic.Expr("<=>", clause_left, logic.disjoin(clauses))
    return change


def pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts):
    """
    Successor state axiom for patrolling ghost state (x,y,t) (from t-1).
    ~Pacman alives at t <==>
     (1. Pacman can move onto a ghost during his turn or
     2. a ghost can move onto Pacman during the ghost's turn.) | (Pacman dead t-1)
    """
    ghost_strs = [ghost_pos_str + str(ghost_num) for ghost_num in xrange(num_ghosts)]
    "*** YOUR CODE HERE ***"
    clause_left = ~logic.PropSymbolExpr(pacman_alive_str, t)
    clauses = [~logic.PropSymbolExpr(pacman_alive_str, t - 1)]
    for g in ghost_strs:
        p_alive = logic.PropSymbolExpr(pacman_alive_str, t - 1)
        S2 = ((p_alive & logic.PropSymbolExpr(g, x, y, t - 1) & logic.PropSymbolExpr(pacman_str, x, y, t)) | (
            p_alive & logic.PropSymbolExpr(g, x, y, t) & logic.PropSymbolExpr(pacman_str, x, y, t)))

        clauses.append(S2)
    dead = logic.Expr("<=>", clause_left, logic.disjoin(clauses))
    # print dead
    return dead


# Get the  blocked potions as two separate lists
def get_blocked_positions(walls, width, height):
    wall_list = walls.asList()
    blocked_east = []
    blocked_west = []

    for w in wall_list:
        x = w[0]
        y = w[1]
        if y == 0 or y == height + 1:
            continue
        if x == 0 and (x + 1, y) not in wall_list:
            blocked_west.append((x + 1, y))
            continue
        if x == width + 1 and (x - 1, y) not in wall_list:
            blocked_east.append((x - 1, y))
            continue
        if (x + 1, y) not in wall_list:
            blocked_west.append((x + 1, y))
        if (x - 1, y) not in wall_list:
            blocked_east.append((x - 1, y))
    return [blocked_east, blocked_west]


def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostPlanningProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    func_start_time = time.time() #DEBUG: timing


    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()

    "*** YOUR CODE HERE ***"
    # Initialization
    legal_actions = ["North", "South", "East", "West"]
    start_state = problem.getStartState()
    ghost_start_state = problem.getGhostStartStates()
    num_ghosts = len(ghost_start_state)
    pac_state = start_state[0]
    food_state = start_state[1]
    max_steps = 50

    foods = food_state.asList()
    # ghost position:

    # Get the blocked positions and pass them into ghostDirectionSSA function
    blocked = get_blocked_positions(walls, width, height)
    blocked_east = blocked[0]
    blocked_west = blocked[1]

    # Initialize the ghost starting states when t=0
    # 1. Each ghost has to be in only one position
    # 2. Each ghost starts with heading east

    ghost_init_pos = []
    ghost_init_direction = []
    ghost_num = 0


    for g in ghost_start_state:
        pos = g.getPosition()
        pos_str = ghost_pos_str + str(ghost_num)
        # Make sure that each ghost is only in its own position?
        ghost_init_pos.append(logic.PropSymbolExpr(pos_str, pos[0], pos[1], 0))
        for x in range(1, height + 1):
            for y in range(1, width + 1):
                if x == pos[0] and y == pos[1]:
                    continue
                else:
                    not_pos = ~logic.PropSymbolExpr(pos_str, x, y, 0)
                    ghost_init_pos.append(not_pos)

        east_str = ghost_east_str + str(ghost_num)
        if (pos[0], pos[1]) in blocked_east:
            init_direc = ~logic.PropSymbolExpr(east_str, 0)
        else:
            init_direc = logic.PropSymbolExpr(east_str, 0)
        ghost_init_direction.append(init_direc)
        ghost_num += 1


    time1 = time.time() - func_start_time #DEBUG
    print("--- %s sec first for loop---" % (time1))#DEBUG

    time2 = time.time()


    ghost_starting_expr = logic.Expr("&", logic.conjoin(ghost_init_pos), logic.conjoin(ghost_init_direction))

    # Pacman has to be at start_state at 0
    s = logic.PropSymbolExpr(pacman_str, pac_state[0], pac_state[1], 0)

    # And Can only be at exactly one spot
    not_states = []


    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) != pac_state:
                not_states.append(~logic.PropSymbolExpr(pacman_str, x, y, 0))

    print("--- %s sec second for loop---" % (time.time() - time2))#DEBUG


    ssas_dict = {}

    big_for_start_time = time.time() #DEBUG: timing

    for t in range(1, max_steps + 1):
        print t
        ts_start = time.time()

        # Pacman is alive initially
        exprs = [logic.PropSymbolExpr(pacman_alive_str, 0)]
        # Ghost has to be in their initial states
        exprs.append(logic.to_cnf(ghost_starting_expr))

        # Get Goals:
        # 1. Getting all the food
        # 2. Don't Die
        goals = []
        for f in foods:
            pos = []
            for i in range(0, t + 1):
                pos.append(logic.PropSymbolExpr(pacman_str, f[0], f[1], i))
            goals.append(atLeastOne(pos))
        food_goals = logic.conjoin(goals)

        alive = logic.PropSymbolExpr(pacman_alive_str, t)

        # Get all the SSAs of pacman and ghosts
        # SSAs of time = t
        ghost_ssa = []
        pac_ssa = []

        ts_1 = time.time()

        # get SSA of all location and put them into cnf
        for x in range(1, width + 1):
            for which_ghost in range(num_ghosts):
                ghost_y = ghost_start_state[which_ghost].getPosition()[1]
                pos_str = ghost_pos_str + str(which_ghost)
                pos_ssa = ghostPositionSuccessorStateAxioms(x, ghost_y, t, which_ghost, walls)
                direc_ssa = ghostDirectionSuccessorStateAxioms(t, which_ghost, blocked_west, blocked_east)
                ghost_ssa.append(logic.Expr("&", logic.to_cnf(pos_ssa), logic.to_cnf(direc_ssa)))
                for y in range(1, height + 1):
                    if ghost_y == y:
                        continue
                    else:
                        not_pos = ~logic.PropSymbolExpr(pos_str, x, y, t)
                        ghost_ssa.append(not_pos)

            #  pacman stuff
            for y in range(1, height+1):
                pac_trans = pacmanSuccessorStateAxioms(x, y, t, walls)
                pac_alive = pacmanAliveSuccessorStateAxioms(x, y, t, num_ghosts)
                pac_ssa.append(logic.Expr("&", logic.to_cnf(pac_trans), logic.to_cnf(pac_alive)))


        ts_2 = time.time()


        # can take exactly one action at time t
        acts = []
        for a in legal_actions:
            acts.append(logic.PropSymbolExpr(a, t-1))
        one_act = exactlyOne(acts)
        pac_ssa.append(one_act)

        # This is SSA of time = t
        ssas_t = logic.Expr("&", logic.conjoin(pac_ssa), logic.conjoin(ghost_ssa))

        if t == 1:
            ssas_dict[1] = ssas_t
        else:
            ssas_dict[t] = logic.Expr("&", ssas_t, ssas_dict[t-1])

        ts_3 = time.time()

        print "The time of getting SSA at t = {} : {}".format(t, ts_2-ts_1)

        if t < (len(foods)):
            print "hi"
            continue

        # This is All SSAs from 1 to time t
        all_ssa = ssas_dict[t]
        exprs.append(all_ssa)

        # Append pacman starting conditions
        exprs.append(s)
        exprs.append(logic.conjoin(not_states))

        # Append ghost starting conditions
        exprs.append(ghost_starting_expr)

        # Append goal
        exprs.append(alive)
        exprs.append(food_goals)

        # Conjoin all expressions
        exprs_con = logic.conjoin(exprs)

        # check if the model is true repeatedly for each t to get optimal solution

        ts_4 = time.time()

        model = logic.pycoSAT(exprs_con)

        ts_5 = time.time()

        # print model

        if model:
            action_seq = extractActionSequence(model, legal_actions)
            return action_seq

        print("--- %s sec ts1-ts_start---" % (ts_1- ts_start))#DEBUG
        print("--- %s sec ts2-ts_1---" % (ts_2 - ts_1))#DEBUG
        print("--- %s sec ts3-ts_2---" % (ts_3 - ts_2))#DEBUG
        print("--- %s sec ts4-ts_3---" % (ts_4 - ts_3))#DEBUG
        print("--- %s sec ts5-ts_4---" % (ts_5 - ts_4))#DEBUG


    print("--- %s sec third huge for loop---" % (time.time() - big_for_start_time))#DEBUG
    print("--- %s sec func runtime---" % (time.time() - func_start_time))#DEBUG

    return []

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(1000)
