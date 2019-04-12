import numpy as np
import math
import abc
import util
from game import Agent, Action
from statistics import mean


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        return better_evaluation_function(successor_game_state)


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        return self.get_action_helper(game_state, 0)

    def get_action_helper(self, state, cur_depth):
        # TODO is it actually depth*2? or just depth?
        if cur_depth == self.depth * 2:
            return self.evaluation_function(state)
        legal_moves = state.get_legal_actions(cur_depth % 2)
        if cur_depth == 0:
            actions_scores = np.array(
                [self.get_action_helper(state.generate_successor(0, move), cur_depth + 1) for move in legal_moves])
            best_move_index = np.argmax(actions_scores)
            # print(max(actions_scores))
            return legal_moves[best_move_index]
        elif cur_depth % 2 == 0:
            if not legal_moves:
                # TODO what to do when we have no legal moves?
                return self.evaluation_function(state)
            return max(
                [self.get_action_helper(state.generate_successor(0, move), cur_depth + 1) for move in legal_moves])
        return min([self.get_action_helper(state.generate_successor(1, move), cur_depth + 1) for move in legal_moves])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.get_action_helper(game_state, 0, -math.inf, math.inf)

    def get_action_helper(self, state, cur_depth, alpha, beta):
        # TODO we don't always get the expected results, because the other player plays first
        legal_moves = state.get_legal_actions(cur_depth % 2)
        if cur_depth == self.depth * 2 or not legal_moves:
            return self.evaluation_function(state)
        if cur_depth == 0:
            max_eval = -math.inf
            best_move = None
            for move in legal_moves:
                cur_eval = self.get_action_helper(state.generate_successor(0, move), cur_depth + 1, alpha, beta)
                if cur_eval > max_eval:
                    max_eval = cur_eval
                    best_move = move
                alpha = max(alpha, cur_eval)
                if beta <= alpha:
                    break
            # print(max_eval)
            return best_move
        elif cur_depth % 2 == 0:
            max_eval = -math.inf
            for move in legal_moves:
                cur_eval = self.get_action_helper(state.generate_successor(0, move), cur_depth + 1, alpha, beta)
                max_eval = max(max_eval, cur_eval)
                alpha = max(alpha, cur_eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in legal_moves:
                cur_eval = self.get_action_helper(state.generate_successor(1, move), cur_depth + 1, alpha, beta)
                min_eval = min(min_eval, cur_eval)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        return self.get_action_helper(game_state, 0)

    def get_action_helper(self, state, cur_depth):
        # TODO is it actually depth*2? or just depth?
        if cur_depth == self.depth * 2:
            return self.evaluation_function(state)
        legal_moves = state.get_legal_actions(cur_depth % 2)
        if cur_depth == 0:
            actions_scores = np.array(
                [self.get_action_helper(state.generate_successor(0, move), cur_depth + 1) for move in legal_moves])
            best_move_index = np.argmax(actions_scores)
            # print(max(actions_scores))
            return legal_moves[best_move_index]
        elif cur_depth % 2 == 0:
            if not legal_moves:
                # TODO what to do when we have no legal moves?
                return self.evaluation_function(state)
            return max(
                [self.get_action_helper(state.generate_successor(0, move), cur_depth + 1) for move in legal_moves])
        return mean([self.get_action_helper(state.generate_successor(1, move), cur_depth + 1) for move in legal_moves])


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    return snake_heuristic(current_game_state) + empty_tiles_heuristic(current_game_state) + max_tile_in_corner(
        current_game_state) * 3


def snake_heuristic(state):
    snake_matrix = np.arange(state._num_of_rows * state._num_of_columns * 10, step=10).reshape(state._num_of_rows,
                                                                                               state._num_of_columns)
    for i in range(0, state._num_of_rows, 2):
        snake_matrix[i] = np.flip(snake_matrix[i], axis=0)
    return np.sum(np.multiply(state.board, snake_matrix))


def gradient_heuristic(state):
    matrix_range = np.arange(state._num_of_rows * state._num_of_columns).reshape(state._num_of_rows,
                                                                                 state._num_of_columns) + 1
    heatistic_matrix = np.multiply(matrix_range, matrix_range.transpose()) * 100
    return np.sum(np.multiply(heatistic_matrix, state.board))


def netta_heuristic(state):
    matrix_range = np.minimum(np.indices((state._num_of_rows, state._num_of_columns))[0],
                              np.indices((state._num_of_rows, state._num_of_columns))[1]) * 100
    return np.sum(np.multiply(matrix_range, state.board))


def max_tile_in_corner(state):
    if state.max_tile() == state.board[-1][-1]:
        return state.max_tile()
    return 0


def empty_tiles_heuristic(state):
    return np.count_nonzero(state.board == 0) * state.max_tile()


# Abbreviation
better = better_evaluation_function
