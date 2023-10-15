# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore() 
        newFoodList = newFood.asList() #list of food
        newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates] #list of ghost positions

        #min distance calculations
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFoodList]
        closestFoodDistance = min(foodDistances) if foodDistances else 0
        ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 0
        
        #1. if pacman waits then subtract from score to incentivize moving
        currPos = currentGameState.getPacmanPosition()
        if newPos == currPos:
            score-=2

        #2: The closer food is, the higher your score will be  
        if closestFoodDistance > 0:
            score += 10/closestFoodDistance

        #3. chase ghost if near enough while scared, if not then avoid
        if closestGhostDistance < 2: 
            if any(newScaredTimes):
                score += 100
            else:
                score -=100

        return score
        
    
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        #main minimax function with branching logic for pacman and ghost helper functions
        def minimax(gameState, depth, agentIndex):
            #if terminal state, no new action
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:
                return maxValue(gameState, depth) #pacman
            else:
                return minValue(gameState, depth, agentIndex) #ghost

        #max agent for pacman agent trying to maximize his score
        def maxValue(gameState, depth):
            bestScore = float('-inf')
            bestAction = None
            
            #determining highest score and best action 
            for action in gameState.getLegalActions(0): 
                successor = gameState.generateSuccessor(0, action)
                score, _ = minimax(successor, depth, 1)  # Ghosts' turn

                #maximize score
                if score > bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction

        #min agent for ghost agent trying to minimize pacman's score
        def minValue(gameState, depth, agentIndex):
            bestScore = float('inf')
            bestAction = None

            #determining lowest score and best action
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

                #logic for determining if the next turn will be pacman or a ghost 
                if agentIndex % gameState.getNumAgents() == gameState.getNumAgents() - 1: 
                    score, _ = minimax(successor, depth - 1, 0)  # Back to Pacman's turn
                else:
                    score, _ = minimax(successor, depth, nextAgentIndex)  #ghost's turn

                #minimize score
                if score < bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction

        _, bestAction = minimax(gameState, self.depth, 0)  #start with Pacman's turn
        return bestAction

        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #main branching logic for if terminal state, if agent is pacman, or ghost. Very similar to minimax
        #alpha = pacman's best option on current path, beta = ghosts' best option on current path 
        def alphaBeta(gameState, depth, agentIndex, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            if agentIndex == 0:
                return maxValue(gameState, depth, alpha, beta)
            else:
                return minValue(gameState, depth, agentIndex, alpha, beta)
        
        def maxValue(gameState, depth, alpha, beta):
            bestScore = float('-inf')
            bestAction = None

            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                score, _ = alphaBeta(successor, depth, 1, alpha, beta)  #ghosts' turn

                #maximizer
                if score > bestScore:
                    bestScore = score
                    bestAction = action

                #pruning
                if bestScore > beta:
                    return bestScore, bestAction
                
                alpha = max(alpha, bestScore) #max's best option on path to route, recursive pruning

            return bestScore, bestAction
        
        def minValue(gameState, depth, agentIndex, alpha, beta):
            bestScore = float('inf')
            bestAction = None

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

                #check for if it's now pacman's turn or another ghost's turn 
                if agentIndex % gameState.getNumAgents() == gameState.getNumAgents() - 1: 
                    score, _ = alphaBeta(successor, depth - 1, 0, alpha, beta)
                else: 
                    score, _ = alphaBeta(successor, depth, nextAgentIndex, alpha, beta)

                #minimizer
                if score < bestScore:
                    bestScore = score
                    bestAction = action
            
                #prune the tree if bestScore isn't as good as max's best option 
                if bestScore < alpha:
                    return bestScore, bestAction
                
                beta = min(beta, bestScore) #min's best option on route, recursive pruning

            return bestScore, bestAction
            
        _, bestAction = alphaBeta(gameState, self.depth, 0, float('-inf'), float('inf'))  #start with Pacman's turn
        return bestAction
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        #branching logic if terminal state, pacman, or ghost
        def expectimax(gameState, depth, agentIndex):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            if agentIndex == 0:
                return maxValue(gameState, depth) #pacman
            else:
                return expectedValue(gameState, depth, agentIndex) #ghost

        #calculates the best score and action Pacman can make 
        def maxValue(gameState, depth):
            bestScore = float('-inf')
            bestAction = None

            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                score, _ = expectimax(successor, depth, 1)  #ghosts' turn

                if score > bestScore:
                    bestScore = score
                    bestAction = action

            return bestScore, bestAction

        #calculates the uniform probability each ghost agent may make, informing pacman's decision
        def expectedValue(gameState, depth, agentIndex):
            expectedValue = 0

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                numActions = len(gameState.getLegalActions(agentIndex)) #number of actions a ghost can make 

                if agentIndex % gameState.getNumAgents() == gameState.getNumAgents() - 1: 
                    score, _ = expectimax(successor, depth - 1, 0)  #back to Pacman's turn
                else:
                    score, _ = expectimax(successor, depth, nextAgentIndex)  #next ghost's turn

                expectedValue += score / numActions #uniform probability  

            return expectedValue, None

        _, bestAction = expectimax(gameState, self.depth, 0)  #start with Pacman's turn
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 3 MAIN CALCULATIONS FOR EVALUATION FUNCTION
    1. Reciprocal of nearest food distance: Designed to scale up as Pacman gets closer to food; 
       1/x function was chosen because function size exponentially decreases as Pacman gets farther away.
    2. Reciprocal of nearest ghost distance: Same as #1, because as ghost gets closer to pacman, 
       his score disproportionally decreases more the closer a ghost is.
    3. Score: add up score, #1, and #2 together to determine highest and lowest scores for 
       pacman and the ghosts respectively.  
    """
    
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [state.getPosition() for state in ghostStates]
    scaredTimes = [state.scaredTimer for state in ghostStates]

    #calculates the distance to the nearest food and ghost
    foodDistances = [manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
    closestFoodDistance = min(foodDistances) if foodDistances else 0
    ghostDistances = [manhattanDistance(pacmanPosition, ghostPos) for ghostPos, scaredTime in zip(ghostPositions, scaredTimes)]
    closestGhostDistance = min(ghostDistances) if ghostDistances else 0

    #checks if the game is in a winning state or a losing state
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    #1. calculate the reciprocal of the nearest food distance to encourage Pacman to move closer to food
    reciprocalNearestFoodDistance = 1.0 / (closestFoodDistance + 1)

    #2. calculate the reciprocal of the nearest ghost distance if ghost is not scared
    reciprocalNearestGhostDistance = 0.0  # Initialize to zero
    if not any(scaredTimes):
        reciprocalNearestGhostDistance = -1.0 / (closestGhostDistance + 1)

    #3. add up total score based on above calculations 
    score = currentGameState.getScore() + reciprocalNearestFoodDistance + reciprocalNearestGhostDistance

    return score

# Abbreviation
better = betterEvaluationFunction
