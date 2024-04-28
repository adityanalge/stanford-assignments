class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.
        """
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                print("Game Ended >>>> ", state.evaluationFunction(state))
                return self.evaluationFunction(state)

            choices =  [(minimax(state.generateSuccessor(agentIndex, action))[0], action) for action in state.getLegalActions(agentIndex)]

            if agentIndex == 0:  # Pac-Man's turn (maximize)
                    max(choices)
            else:  # Ghosts' turn (minimize)
                    min(choices)
        
        value, action = minimax(gameState)
        return action