import logging
import os
import random
import time

import numpy as np

import go
import utils
from config import GLOBAL_PARAMETER_STORE, GOPARAMETERS
from mcts import MCTreeSearchNode
from utils import make_sgf

logger = logging.getLogger(__name__)

class AlphaGoZeroAgent:

    def __init__(self, network, player_v_player=False, workers=GLOBAL_PARAMETER_STORE.SIMULTANEOUS_LEAVES):
        self.network = network
        self.player_v_player = player_v_player
        self.workers = workers
        self.mean_reward_store = []
        self.game_description_store = []
        self.child_probability_store = []
        self.root = None
        self.result = 0
        self.logging_buffer = None
        self.conduct_exploration = True
        if self.player_v_player:
            self.conduct_exploration = True
        else:
            self.conduct_exploration = False

    def initialize_game(self, board_state=None):
        if board_state is None:
            board_state = go.BoardState()
        self.root = MCTreeSearchNode(board_state)
        self.result = 0
        self.logging_buffer = None
        self.game_description_store = []
        self.child_probability_store = []
        self.mean_reward_store = []

    def play_move(self, coordinates):
        if not self.player_v_player:
            self.child_probability_store.append(self.root.get_children_as_probability_distributions())
        self.mean_reward_store.append(self.root.node_mean_reward)
        self.game_description_store.append(self.root.describe())
        self.root = self.root.record_child_node(utils.to_flat(coordinates))
        self.board_state = self.root.board_state
        del self.root.parent_node.children_moves
        return True

    def select_move(self):
        # If we have conducted enough moves and this is single player mode, we turn off exploration
        if self.root.board_state.n > GLOBAL_PARAMETER_STORE.TEMPERATURE_CUTOFF and not self.player_v_player:
            self.conduct_exploration = False

        if self.conduct_exploration:
            child_visits_cum_sum = self.root.child_visit_counts.cumsum()
            child_visits_cum_sum /= child_visits_cum_sum[-1]
            coorindate = child_visits_cum_sum.searchsorted(random.random())
        else:
            coorindate = np.argmax(self.root.child_visit_counts)

        return utils.from_flat(coorindate)

    def search_tree(self):
        child_node_store = []
        iteration_count = 0
        while len(child_node_store) < self.workers and iteration_count < self.workers * 2:
            iteration_count += 1
            child_node = self.root.choose_next_child_node()
            if child_node.is_done():
                result = 1 if child_node.board_state.score() > 0 else -1
                child_node.back_propagate_result(result, start_node=self.root)
                continue
            child_node.propagate_loss(start_node=self.root)
            child_node_store.append(child_node)
        if len(child_node_store) > 0:
            move_probs, values = self.network.predict_on_multiple_board_states(
                [child_node.board_state for child_node in child_node_store])
            for child_node, move_prob, result in zip(child_node_store, move_probs, values):
                child_node.revert_loss(start_node=self.root)
                child_node.incorporate_results(move_prob, result, start_node=self.root)

    def should_resign(self):
        return self.root.mean_reward_perspective < GLOBAL_PARAMETER_STORE.RESIGN_THRESHOLD

    def set_result(self, winner, was_resign):
        self.result = winner
        if was_resign:
            string = "B+R" if winner == GOPARAMETERS.BLACK else "W+R"
        else:
            string = self.root.board_state.result_string()
        self.logging_buffer = string

    def to_sgf(self, use_comments=True):
        assert self.logging_buffer is not None
        pos = self.root.board_state
        if use_comments:
            comments = self.game_description_store or ['No comments.']
            comments[0] = ("Resign Threshold is: {}\n".format(GLOBAL_PARAMETER_STORE.RESIGN_THRESHOLD)) + comments[0]
        else:
            comments = []
        return make_sgf(pos.recent, self.logging_buffer,
                        white_name=os.path.basename(self.network.model_path) or "Unknown",
                        black_name=os.path.basename(self.network.model_path) or "Unknown",
                        comments=comments)

    def is_done(self):
        return self.result != 0 or self.root.is_done()

    def extract_data(self):
        assert len(self.child_probability_store) == self.root.board_state.n
        assert self.result != 0
        for pwc, pi in zip(go.replay_board_state(self.root.board_state, self.result),
                           self.child_probability_store):
            yield pwc.board_state, pi, pwc.result

def play_match(black_net, white_net, games, readouts, sgf_dir):

    # Create the players for the game
    black = AlphaGoZeroAgent(black_net, player_v_player=True, workers=GLOBAL_PARAMETER_STORE.SIMULTANEOUS_LEAVES)
    white = AlphaGoZeroAgent(white_net, player_v_player=True, workers=GLOBAL_PARAMETER_STORE.SIMULTANEOUS_LEAVES)

    black_name = os.path.basename(black_net.model_path)
    white_name = os.path.basename(white_net.model_path)

    for game_num in range(games):
        # Keep track of the number of moves made in the game
        num_moves = 0

        black.initialize_game()
        white.initialize_game()

        while True:
            start = time.time()
            active = white if num_moves % 2 else black
            inactive = black if num_moves % 2 else white

            current_readouts = active.root.node_visit_count
            while active.root.node_visit_count < current_readouts + readouts:
                active.search_tree()

            logger.info(active.root.board_state)

            # Check whether a player should resign
            if active.should_resign():
                active.set_result(-1 * active.root.board_state.to_play, was_resign=True)
                inactive.set_result(active.root.board_state.to_play, was_resign=True)

            if active.is_done():
                sgf_file_path = "{}-{}-vs-{}-{}.sgf".format(int(time.time()), white_name, black_name, game_num)
                with open(os.path.join(sgf_dir, sgf_file_path), 'w') as fp:
                    game_as_sgf_string = make_sgf(active.board_state.recent, active.logging_buffer,
                                      black_name=black_name,
                                      white_name=white_name)
                    fp.write(game_as_sgf_string)
                print("Game Over", game_num, active.logging_buffer)
                break

            move = active.select_move()
            active.play_move(move)
            inactive.play_move(move)

            dur = time.time() - start
            num_moves += 1

            if num_moves % 10 == 0:
                timeper = (dur / readouts) * 100.0
                print(active.root.board_state)
                logger.info("{}: {} readouts, {:.3f} s/100. ({:.2f} sec)".format(num_moves, readouts, timeper, dur))

def play_against_self(network, readouts):
    agent = AlphaGoZeroAgent(network=network,
                             workers=GLOBAL_PARAMETER_STORE.SIMULTANEOUS_LEAVES)

    agent.initialize_game()

    first_node = agent.root.choose_next_child_node()
    prob, val = network.predict_on_single_board_state(first_node.board_state)
    first_node.incorporate_results(prob, val, first_node)

    while True:
        start = time.time()
        agent.root.inject_noise()
        current_readouts = agent.root.node_visit_count

        while agent.root.node_visit_count < current_readouts + readouts:
            agent.search_tree()

        if agent.should_resign():
            agent.set_result(-1 * agent.root.board_state.to_play, was_resign=True)
            break

        move = agent.select_move()
        agent.play_move(move)

        if agent.root.is_done():
            agent.set_result(agent.root.board_state.result(), was_resign=False)
            break

        if agent.root.board_state.n % 10 == 0:
            logger.info("Mean Reward: {:.5f}".format(agent.root.node_mean_reward))
            dur = time.time() - start
            logger.info("{}: {} readouts, {:.3f} s/100. ({:.2f} sec)".format(
                agent.root.board_state.n, readouts, dur / readouts * 100.0, dur))

        print(agent.root.board_state, agent.root.board_state.score())

    return agent
