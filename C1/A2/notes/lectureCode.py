import sys
import util

sys.setrecursionlimit(10000)

### Model (search problem)

class TransportationProblem(object):

    def __init__(self, N, weights):
        # N = number of blocks
        self.weights = weights
        self.N = N

    def startState(self):
        return 1
    
    def isEnd(self, state):
        return state == self.N
    
    def succAndCost(self, state):
        # return list of (action, newState, cost) triples
        result = []
        if state + 1 <= self.N:
            result.append(('walk', state + 1, self.weights["walk"]))
        if state * 2 <= self.N:
            result.append(('tram', state * 2, self.weights["tram"]))
        return result
    
def printSolution(solution):
    totalCost, history = solution
    print('totalCost: {}'.format(totalCost))
    for item in history:
        print(item)

### Algorithms
    
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

def dynamicProgramming(problem):
    cache = {}

    def futureCost(state):
        if problem.isEnd(state):
            return 0
        if state in cache:
            return cache[state][0]

        result = min((cost + futureCost(newState), action, newState, cost) for action, newState, cost in problem.succAndCost(state))

        cache[state] = result
        return result[0]

    state = problem.startState()
    totalCost = futureCost(state)        

    history = []
    while not problem.isEnd(state):
        _, action, newState, cost = cache[state]
        history.append((action, newState, cost))
        state = newState

    return (futureCost(problem.startState()), history)

# def uniformCostSearch(problem):
#     frontier = util.PriorityQueue()
#     frontier.update(problem.startState(), 0)

#     while True:
#         state,pastCost = frontier.removeMin()
#         if problem.isEnd(state):
#             return(pastCost, [])
        
#         for action, newState, cost in problem.succAndCost(state):
#             frontier.update(newState, pastCost + cost)


### Main

# problem = TransportationProblem(N = 40)
# printSolution(dynamicProgramming(problem))
# print(problem.succAndCost(3))
# print(problem.succAndCost(9))
# printSolution(backtrackingSearch(problem))
# printSolution(uniformCostSearch(problem))

def predict(N, weights):
    problem = TransportationProblem(N, weights)
    totalCost, history = dynamicProgramming(problem)
    return [action for action, newState, cost in history]

def generateExamples():
    trueWeights = {'walk': 2, 'tram': 5}
    return [(N, predict(N, trueWeights)) for N in range(1,300)]

def structuredPerception(examples):
    weights = {'walk': 0, 'tram': 0}

    for t in range(100):
        numMistakes = 0
        for N, trueActions in examples:
            predActions = predict(N, weights)

            if predActions != trueActions:
                numMistakes += 1
            
            for action in trueActions:
                weights[action] -= 1
            for action in predActions:
                weights[action] += 1
        print('Iteration {}, numMistakes = {}, weights = {}'.format(t, numMistakes, weights))
       
        if numMistakes == 0:
            break            

examples = generateExamples()

print('Training Dataset')
for example in examples:
    print(' ', example)

structuredPerception(examples)