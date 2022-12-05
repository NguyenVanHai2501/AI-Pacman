# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


'''-----------  DFS begins  -----------'''


def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""

    '''This function pushes non-visited nodes onto the stack.
    Nodes are popped one by one, and the following steps are performed:
    1. The node is marked as visited.
    2. If it is a goal node, the loop stops, and the solution is obtained by backtracking using stored parents.
    3. If it is not a goal node, it is expanded.
    4. If the successor node is not visited, then it is pushed onto the stack and its parent is stored.'''

#   from util import Stack
    
    start_state = problem.getStartState()
    stack =Stack()
    stack.push((start_state, []))
    visited = []
    while not stack.isEmpty():
        current_state, actions = stack.pop()
        if problem.isGoalState(current_state):
            return actions
        if current_state not in visited:
            visited.append(current_state)
            successors = problem.getSuccessors(current_state)
            for state, action, cost in successors:
                if state not in visited:
                    stack.push((state, actions+[action]))
    return []

'''-----------  DFS ends  -----------'''


'''-----------  BFS begins  -----------'''

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    '''This function pushes non-visited nodes onto the queue.
    Nodes are popped one by one, and the following steps are performed:
    1. The node is marked as visited.
    2. If it is a goal node, the loop stops, and the solution is obtained by backtracking using stored parents.
    3. If it is not a goal node, it is expanded.
    4. If the successor node is not visited, and has not been expanded as a child of another node,
       then it is pushed onto the queue and its parent is stored.'''

    # initializations

    # "visited" contains nodes which have been popped from the queue,
    # and the direction from which they were obtained
    visited = {}
    # "solution" contains the sequence of directions for Pacman to get to the goal state
    solution = []
    # "queue" contains triplets of: (node in the fringe list, direction, cost)
    queue = util.Queue()
    # "parents" contains nodes and their parents
    parents = {}

    # start state is obtained and added to the queue
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0))
    # the direction from which we arrived in the start state is undefined
    visited[start] = 'Undefined'

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solution

    # loop while queue is not empty and goal is not reached
    goal = False;
    while(queue.isEmpty() != True and goal != True):
        # pop from top of queue
        node = queue.pop()
        # store element and its direction
        visited[node[0]] = node[1]
        # check if element is goal
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        # expand node
        for elem in problem.getSuccessors(node[0]):
            # if successor has not already been visited or expanded as a child of another node
            if elem[0] not in visited.keys() and elem[0] not in parents.keys():
                # store successor and its parent
                parents[elem[0]] = node[0]
                # push successor onto queue
                queue.push(elem)

    # finding and storing the path
    while(node_sol in parents.keys()):
        # find parent
        node_sol_prev = parents[node_sol]
        # prepend direction to solution
        solution.insert(0, visited[node_sol])
        # go to previous node
        node_sol = node_sol_prev

    return solution

'''-----------  BFS ends  -----------'''


'''-----------  UCS begins  -----------'''

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    '''This function pushes non-visited nodes onto the priority queue.
    Nodes are popped one by one, and the following steps are performed:
    1. The node is marked as visited.
    2. If it is a goal node, the loop stops, and the solution is obtained by backtracking using stored parents.
    3. If it is not a goal node, it is expanded.
    4. If the successor node is not visited, its cost is calculated.
    5. If the cost of the successor node was calculated earlier while expanding a different node,
       and if the new calculated cost is less than old cost, then the cost and parent are updated,
       and it is pushed onto the priority queue with new cost as priority.'''

    # hàng đợi
    queue = util.PriorityQueue()

    # lưu trữ những node đã di chuyển đến
    visited = set()

    start_node = problem.getStartState()

    cost = 0
    collection = []

    queue.push((start_node, collection, cost), cost)

    while not queue.isEmpty():
        # lấy dữ liệu node đầu tiên ra khỏi hàng đợi
        cr_node, cr_collection, cr_cost = queue.pop()

        # nếu node này trùng với vị trí win thi trả về collection
        if problem.isGoalState(cr_node):
            return cr_collection

        # nếu node hiện tại chưa được di chuyển đến
        # thì thêm vào hàng đợi những node lân cận nó có thể di chuyển đến
        if not cr_node in visited:
            visited.add(cr_node)

            for next_node, next_way, next_cost in problem.getSuccessors(cr_node):
                if next_node and not next_node in visited:
                    queue.push((next_node, cr_collection + [next_way], cr_cost + next_cost), cr_cost + next_cost)

    util.raiseNotDefined()


'''-----------  UCS ends  -----------'''


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


'''-----------  A* begins  -----------'''


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    '''This function pushes non-visited nodes onto the priority queue.
    Nodes are popped one by one, and the following steps are performed:
    1. The node is marked as visited.
    2. If it is a goal node, the loop stops, and the solution is obtained by backtracking using stored parents.
    3. If it is not a goal node, it is expanded.
    4. If the successor node is not visited, its cost is calculated using the heuristic function.
    5. If the cost of the successor node was calculated earlier while expanding a different node,
       and if the new calculated cost is less than old cost, then the cost and parent are updated,
       and it is pushed onto the priority queue with new cost as priority.'''

    # # initializations

    # # "visited" contains nodes which have been popped from the queue,
    # # and the direction from which they were obtained
    # visited = {}
    # # "solution" contains the sequence of directions for Pacman to get to the goal state
    # solution = []
    # # "queue" contains triplets of: (node in the fringe list, direction, cost)
    # queue = util.PriorityQueue()
    # # "parents" contains nodes and their parents
    # parents = {}
    # # "cost" contains nodes and their corresponding costs
    # cost = {}

    # # start state is obtained and added to the queue
    # start = problem.getStartState()
    # queue.push((start, 'Undefined', 0), 0)
    # # the direction from which we arrived in the start state is undefined
    # visited[start] = 'Undefined'
    # # cost of start state is 0
    # cost[start] = 0

    # # return if start state itself is the goal
    # if problem.isGoalState(start):
    #     return solution

    # # loop while queue is not empty and goal is not reached
    # goal = False;
    # while(queue.isEmpty() != True and goal != True):
    #     current = queue.pop()
    #     # store element and its direction
    #     visited[current[0]] = current[1]
    #     # check if element is goal
    #     if problem.isGoalState(current[0]):
    #         current_sol = current[0]
    #         goal = True
    #         break
    #     # expand current
    #     for elem in problem.getSuccessors(current[0]):
    #         # if successor is not visited, calculate its new cost
    #         if elem[0] not in visited.keys():
    #             priority = current[2] + elem[2] + heuristic(elem[0], problem)
    #             # if cost of successor was calculated earlier while expanding a different current,
    #             # if new cost is more than old cost, continue
    #             if elem[0] in cost.keys():
    #                 if cost[elem[0]] <= priority:
    #                     continue
    #             # if new cost is less than old cost, push to queue and change cost and parent
    #             queue.push((elem[0], elem[1], current[2] + elem[2]), priority)
    #             cost[elem[0]] = priority
    #             # store successor and its parent
    #             parents[elem[0]] = current[0]

    # # finding and storing the path
    # while(current_sol in parents.keys()):
    #     # find parent
    #     current_sol_prev = parents[current_sol]
    #     # prepend direction to solution
    #     solution.insert(0, visited[current_sol])
    #     # go to previous current
    #     current_sol = current_sol_prev

    # return solution
    p_queue = util.PriorityQueue()  # hang doi uu tien (cost) chua vi tri hien tai, danh sach cac hanh dong va gia phai tra
    start = problem.getStartState()
    visited = {}
    cost = 0  # heuristic de tinh gia tri uu tien cost
    action = []
    p_queue.push((start, action, cost), cost)

    while not p_queue.isEmpty():
        # kiem tra neu trang thai hien tai la win thi tra lai danh sach hang dong
        current = p_queue.pop()
        if problem.isGoalState(current[0]):
            return current[1]
        if current[0] not in visited:
            visited[current[0]] = True
            for next, act, co in problem.getSuccessors(current[0]):
                if next and next not in visited:
                    next_act = current[1] + [act]
                    next_cost = current[2] + co
                    p_queue.push((next, next_act, next_cost), next_cost + heuristic(next, problem))


'''-----------  A* ends  -----------'''

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
