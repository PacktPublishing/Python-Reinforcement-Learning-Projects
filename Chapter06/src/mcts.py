import collections
import math

import numpy as np

import utils
from config import MCTSPARAMETERS, GOPARAMETERS


class RootNode(object):

    def __init__(self):
        self.parent_node = None
        self.child_visit_counts = collections.defaultdict(float)
        self.child_cumulative_rewards = collections.defaultdict(float)


class MCTreeSearchNode(object):

    def __init__(self, board_state, previous_move=None, parent_node=None):
        """
        A node of a MCTS tree. It is primarily responsible with keeping track of its children's scores
        and other statistics such as visit count. It also makes decisions about where to move next.

        board_state (go.BoardState): The Go board
        fmove (int): A number which represents the coordinate of the move that led to this board state. None if pass
        parent (MCTreeSearchNode): The parent node
        """
        if parent_node is None:
            parent_node = RootNode()
        self.parent_node = parent_node
        self.previous_move = previous_move
        self.board_state = board_state
        self.is_visited = False
        self.loss_counter = 0
        self.illegal_moves = 1000 * (1 - self.board_state.enumerate_possible_moves())
        self.child_visit_counts = np.zeros([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32)
        self.child_cumulative_rewards = np.zeros([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32)
        self.original_prior = np.zeros([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32)
        self.child_prior = np.zeros([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32)
        self.children_moves = {}

    def __repr__(self):
        return "<MCTreeSearchNode  Move: {} | N: {} | Player: {}>".format(self.board_state.recent[-1:],
                                                                          self.node_visit_count, self.board_state.to_play)

    @property
    def child_action_score(self):
        return self.child_mean_rewards * self.board_state.to_play + self.child_node_scores - self.illegal_moves

    @property
    def child_mean_rewards(self):
        return self.child_cumulative_rewards / (1 + self.child_visit_counts)

    @property
    def child_node_scores(self):
        # This scores each child according to the UCT scoring system
        return (MCTSPARAMETERS.c_PUCT * math.sqrt(1 + self.node_visit_count) * self.child_prior /
                (1 + self.child_visit_counts))

    @property
    def node_mean_reward(self):
        return self.node_cumulative_reward / (1 + self.node_visit_count)

    @property
    def node_visit_count(self):
        return self.parent_node.child_visit_counts[self.previous_move]

    @node_visit_count.setter
    def node_visit_count(self, value):
        self.parent_node.child_visit_counts[self.previous_move] = value

    @property
    def node_cumulative_reward(self):
        return self.parent_node.child_cumulative_rewards[self.previous_move]

    @node_cumulative_reward.setter
    def node_cumulative_reward(self, value):
        self.parent_node.child_cumulative_rewards[self.previous_move] = value

    @property
    def mean_reward_perspective(self):
        "Return value of board_state, from perspective of player to play."
        return self.node_mean_reward * self.board_state.to_play

    def choose_next_child_node(self):
        current = self
        pass_move = GOPARAMETERS.N * GOPARAMETERS.N
        while True:
            current.node_visit_count += 1
            # We stop searching when we reach a new leaf node
            if not current.is_visited:
                break
            if (current.board_state.recent
                and current.board_state.recent[-1].move is None
                    and current.child_visit_counts[pass_move] == 0):
                current = current.record_child_node(pass_move)
                continue

            best_move = np.argmax(current.child_action_score)
            current = current.record_child_node(best_move)
        return current

    def record_child_node(self, next_coordinate):
        if next_coordinate not in self.children_moves:
            new_board_state = self.board_state.play_move(
                utils.from_flat(next_coordinate))
            self.children_moves[next_coordinate] = MCTreeSearchNode(
                new_board_state, previous_move=next_coordinate, parent_node=self)
        return self.children_moves[next_coordinate]

    def propagate_loss(self, start_node):
        """
        Add a loss to each node upstream

        Args:
            start_node (MCTreeSearchNode): The node to propagate the loss to
        """
        self.loss_counter += 1
        # This is a "win" for the current node; hence a loss for its parent node
        # who will be deciding whether to investigate this node again.
        loss = self.board_state.to_play
        self.node_cumulative_reward += loss
        if self.parent_node is None or self is start_node:
            return
        self.parent_node.propagate_loss(start_node)

    def revert_loss(self, start_node):
        self.loss_counter -= 1
        revert = -1 * self.board_state.to_play
        self.node_cumulative_reward += revert
        if self.parent_node is None or self is start_node:
            return
        self.parent_node.revert_loss(start_node)

    def revert_visits(self, start_node):
        self.node_visit_count -= 1
        if self.parent_node is None or self is start_node:
            return
        self.parent_node.revert_visits(start_node)

    def incorporate_results(self, move_probabilities, result, start_node):
        if self.is_visited:
            self.revert_visits(start_node=start_node)
            return
        self.is_visited = True
        self.original_prior = self.child_prior = move_probabilities
        self.child_cumulative_rewards = np.ones([GOPARAMETERS.N * GOPARAMETERS.N + 1], dtype=np.float32) * result
        self.back_propagate_result(result, start_node=start_node)

    def back_propagate_result(self, result, start_node):
        """
        This function back propagates the result of a match all the way to where the search started from

        Args:
            result (int): the result of the search (1: black, -1: white won)
            start_node (MCTreeSearchNode): the node to back propagate until
        """
        # Keep track of the cumulative reward in this node
        self.node_cumulative_reward += result

        if self.parent_node is None or self is start_node:
            return

        self.parent_node.back_propagate_result(result, start_node)

    def is_done(self):
        '''True if the last two moves were Pass or if the board_state is at a move
        greater than the max depth.
        '''
        return self.board_state.is_game_over() or self.board_state.n >= MCTSPARAMETERS.MAX_DEPTH

    def inject_noise(self):
        dirch = np.random.dirichlet([MCTSPARAMETERS.DIRICHLET_NOISE] * ((GOPARAMETERS.N * GOPARAMETERS.N) + 1))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def get_children_as_probability_distributions(self):
        """
        Get the distribution of visiting child nodes as probabilities
        """
        probs = self.child_visit_counts ** .95
        return probs / np.sum(probs)

    def get_best_path(self):
        node = self
        out_buffer = []
        while node.children_moves:
            next_kid = np.argmax(node.child_visit_counts)
            node = node.children_moves.get(next_kid)
            if node is None:
                out_buffer.append("Game Finished Here")
                break
            out_buffer.append("{} ({}) => ".format(utils.to_kgs(utils.from_flat(node.previous_move)),
                                                node.node_visit_count))
        out_buffer.append("Mean Reward: {:.5f}\n".format(node.node_mean_reward))
        return ''.join(out_buffer)


    def describe(self):
        sort_order = list(range(GOPARAMETERS.N * GOPARAMETERS.N + 1))
        sort_order.sort(key=lambda i: (
            self.child_visit_counts[i], self.child_action_score[i]), reverse=True)
        output = []
        output.append("{mean_reward:.4f}\n".format(mean_reward=self.node_mean_reward))
        output.append(self.get_best_path())
        return ''.join(output)
