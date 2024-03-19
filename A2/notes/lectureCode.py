# import sys
# import util

# sys.setrecursionlimit(10000)

# ### Model (search problem)

# class TransportationProblem(object):

#     def __init__(self, N):
#         # N = number of blocks
#         self.N = N

#     def startState(self):
#         return 1
    
#     def isEnd(self, state):
#         return state == self.N
    
#     def succAndCost(self, state):
#         # return list of (action, newState, cost) triples
#         result = []
#         if state + 1 <= self.N:
#             result.append(('walk', state + 1, 1))
#         if state * 2 <= self.N:
#             result.append(('tram', state * 2, 2))
#         return result
    
# def printSolution(solution):
#     totalCost, history = solution
#     print('totalCost: {}'.format(totalCost))
#     for item in history:
#         print(item)

# ### Algorithms
    
# def backtrackingSearch(problem):
#     # Best solution found so far (dictionary because of python scoping technicality)
#     best = {
#         'cost': float('+inf'),
#         'history': None
#     }

#     def recurse(state, history, totalCost):
#         # At state, having undergone history, accumulated totalCost.
#         # Explore the rest of the subtree under state.
#         if problem.isEnd(state):
#             # print("reached end state >>>> ", state)
#             # print("history >>>> ", history)
#             if totalCost < best['cost']:
#                 best['cost'] = totalCost
#                 best['history'] = history
#             return
#         else:
#             possibleActions = problem.succAndCost(state)
#             # print("possibleActions >>>> ", possibleActions)

#             for i in range(len(possibleActions)):
#                 action = possibleActions[i][0]
#                 # print("action >>>> ", action)

#                 newState = possibleActions[i][1]
#                 # print("newState >>>> ", newState)

#                 cost = possibleActions[i][2]
#                 # print("cost >>>> ", cost)

#                 recurse(newState, history + [(action, newState, cost)], totalCost + cost)
    
#     recurse(problem.startState(), history = [], totalCost = 0)

#     return (best['cost'], best['history'])

# def dynamicProgramming(problem):
#     cache = {}

#     def futureCost(state):
#         # Base case
#         if problem.isEnd(state):
#             return 0
#         if state in cache:
#             return cache[state]
        
#         possibleActions = problem.succAndCost(state)
#         # print("possibleActions >>>> ", possibleActions)
        
#         costs = []

#         for i in range(len(possibleActions)):
#             action = possibleActions[i][0]
#             # print("action >>>> ", action)

#             newState = possibleActions[i][1]
#             # print("newState >>>> ", newState)

#             cost = possibleActions[i][2]
#             # print("cost >>>> ", cost)

#             costs.append(cost + futureCost(newState))

#         cache[state] = min(costs)
#         result = min(cost + futureCost(newState) for action, newState, cost in problem.succAndCost(state))
#         cache[state] = result
#         return result
#     return (futureCost(problem.startState()), [])

# def uniformCostSearch(problem):
#     frontier = util.PriorityQueue()
#     frontier.update(problem.startState(), 0)

#     while True:
#         state,pastCost = frontier.removeMin()
#         if problem.isEnd(state):
#             return(pastCost, [])
        
#         for action, newState, cost in problem.succAndCost(state):
#             frontier.update(newState, pastCost + cost)


# ### Main

# problem = TransportationProblem(N = 40)
# # print(problem.succAndCost(3))
# # print(problem.succAndCost(9))
# printSolution(backtrackingSearch(problem))
# printSolution(dynamicProgramming(problem))
# printSolution(uniformCostSearch(problem))