"""Utilities for working with go games and coordinates"""
import datetime
import functools
import itertools
import logging
import random
import re
import sys
import time
from contextlib import contextmanager

import gtp
import numpy as np
import petname
import sgf

import go
from config import NETWORK_HYPERPARAMETERS, GOPARAMETERS, GLOBAL_PARAMETER_STORE
from constants import PATHS, MODEL_NAME_REGEX, MODEL_NUM_REGEX
# from go import BoardState, PositionWithContext

logger = logging.getLogger(__name__)

def print_flags():
    flags = {
        'MODELS_DIR': PATHS.MODELS_DIR,
        'SELFPLAY_DIR': PATHS.SELFPLAY_DIR,
        'HOLDOUT_DIR': PATHS.HOLDOUT_DIR,
        'SGF_DIR': PATHS.SGF_DIR,
        'TRAINING_CHUNK_DIR': PATHS.TRAINING_CHUNK_DIR,
        'ESTIMATOR_WORKING_DIR': PATHS.ESTIMATOR_WORKING_DIR,
    }
    logger.info("Directories:")
    logger.info('{}'.format(flags))

def parse_parameters(**overrides):
    params = NETWORK_HYPERPARAMETERS
    params.update(**overrides)
    return params

def parse_game_result(result):
    if re.match(r'[bB]\+', result):
        return 1
    elif re.match(r'[wW]\+', result):
        return -1
    else:
        return 0

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    logger.info("%s: %.3f seconds" % (message, (tock - tick)))


@contextmanager
def logged_timer(message):
    tick = time.time()
    yield
    tock = time.time()
    logger.info("%s: %.3f seconds" % (message, (tock - tick)))


INVERSES = {
    'identity': 'identity',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90',
    'flip': 'flip',
    'fliprot90': 'fliprot90',
    'fliprot180': 'fliprot180',
    'fliprot270': 'fliprot270',
}
IMPLS = {
    'identity': lambda x: x,
    'rot90': np.rot90,
    'rot180': functools.partial(np.rot90, k=2),
    'rot270': functools.partial(np.rot90, k=3),
    'flip': lambda x: np.rot90(np.fliplr(x)),
    'fliprot90': np.flipud,
    'fliprot180': lambda x: np.rot90(np.flipud(x)),
    'fliprot270': np.fliplr,
}
SYMMETRIES = list(INVERSES.keys())

def invert_symmetry(s):
    return INVERSES[s]


def apply_symmetry_feat(s, features):
    return IMPLS[s](features)


def apply_symmetry_pi(s, pi):
    pi = np.copy(pi)
    # rotate all moves except for the pass move at end
    pi[:-1] = IMPLS[s](pi[:-1].reshape([GOPARAMETERS.N, GOPARAMETERS.N])).ravel()
    return pi


def shuffle_feature_symmetries(features):
    symmetries_used = [random.choice(SYMMETRIES) for f in features]
    return symmetries_used, [apply_symmetry_feat(s, f)
                             for s, f in zip(symmetries_used, features)]


def invert_policy_symmetries(symmetries, pis):
    return [apply_symmetry_pi(invert_symmetry(s), pi)
            for s, pi in zip(symmetries, pis)]

def generate(model_num):
    if model_num == 0:
        new_name = 'bootstrap'
    else:
        new_name = petname.generate()
    full_name = "%06d-%s" % (model_num, new_name)
    return full_name


def detect_model_version(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    else:
        return None


def detect_model_name(string):
    """Takes a string related to a model name and extract its model name.

    For example:
        '000000-bootstrap.index' => '000000-bootstrap'
    """
    match = re.match(MODEL_NAME_REGEX, string)
    if match:
        return match.group()
    else:
        return None


"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout AlphaGoZero PARAMETERS.

AlphaGoZero Coordinate: This is a tuple of the form (row, column) that is indexed
    starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
    two-letter pairs having the form (column, row) indexed from the upper-left
    where 0, 0 = 'aa'.
KGS Coordinate: Human-readable coordinate string indexed from bottom left, with
    the first character a capital letter for the column and the second a number
    from 1-19 for the row. Note that KGS chooses to skip the letter 'I' due to
    its similarity with 'l' (lowercase 'L').
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
    in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
AlphaGoZero coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
KGS             'A19'           'T19'           'pass'
pygtp           (1, 19)         (19, 19)        (0, 0)
"""




def from_flat(flat):
    """Converts from a flattened coordinate to a AlphaGoZero coordinate."""
    if flat == GOPARAMETERS.N * GOPARAMETERS.N:
        return None
    return divmod(flat, GOPARAMETERS.N)


def to_flat(coord):
    """Converts from a AlphaGoZero coordinate to a flattened coordinate."""
    if coord is None:
        return GOPARAMETERS.N * GOPARAMETERS.N
    return GOPARAMETERS.N * coord[0] + coord[1]


def from_sgf(sgfc):
    """Converts from an SGF coordinate to a AlphaGoZero coordinate."""
    if sgfc is None or sgfc == '':
        return None
    return GOPARAMETERS.SGF_COLUMNS.index(sgfc[1]), GOPARAMETERS.SGF_COLUMNS.index(sgfc[0])


def to_sgf(coord):
    """Converts from a AlphaGoZero coordinate to an SGF coordinate."""
    if coord is None:
        return ''
    return GOPARAMETERS.SGF_COLUMNS[coord[1]] + GOPARAMETERS.SGF_COLUMNS[coord[0]]


def to_kgs(coord):
    """Converts from a AlphaGoZero coordinate to a KGS coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(GOPARAMETERS.KGS_COLUMNS[x], GOPARAMETERS.N - y)


def from_pygtp(pygtpc):
    """Converts from a pygtp coordinate to a AlphaGoZero coordinate."""
    # GTP has a notion of both a Pass and a Resign, both of which are mapped to
    # None, so the conversion is not precisely bijective.
    if pygtpc in (gtp.PASS, gtp.RESIGN):
        return None
    return GOPARAMETERS.N - pygtpc[1], pygtpc[0] - 1


def to_pygtp(coord):
    """Converts from a AlphaGoZero coordinate to a pygtp coordinate."""
    if coord is None:
        return gtp.PASS
    return coord[1] + 1, GOPARAMETERS.N - coord[0]


def translate_sgf_move(player_move, comment):
    if player_move.color not in (GOPARAMETERS.BLACK, GOPARAMETERS.WHITE):
        raise ValueError("Can't translate color %s to sgf" % player_move.color)
    c = to_sgf(player_move.move)
    color = 'B' if player_move.color == GOPARAMETERS.BLACK else 'W'
    if comment is not None:
        comment = comment.replace(']', r'\]')
        comment_node = "C[{}]".format(comment)
    else:
        comment_node = ""
    return ";{color}[{coords}]{comment_node}".format(
        color=color, coords=c, comment_node=comment_node)


def make_sgf(
    move_history,
    result_string,
    ruleset="Chinese",
    komi=7.5,
    white_name=GLOBAL_PARAMETER_STORE.PROGRAM_IDENTIFIER,
    black_name=GLOBAL_PARAMETER_STORE.PROGRAM_IDENTIFIER,
    comments=[]):
    boardsize = GOPARAMETERS.N
    game_moves = ''.join(translate_sgf_move(*z)
                         for z in itertools.zip_longest(move_history, comments))
    result = result_string
    return GLOBAL_PARAMETER_STORE.SGF_TEMPLATE.format(**locals())


def sgf_prop(value_list):
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list


# def handle_node(pos, node):
#     props = node.properties
#     black_stones_added = [from_sgf(
#         c) for c in props.get('AB', [])]
#     white_stones_added = [from_sgf(
#         c) for c in props.get('AW', [])]
#     if black_stones_added or white_stones_added:
#         return add_stones(pos, black_stones_added, white_stones_added)
#     # If B/W props are not present, then there is no move. But if it is present and equal to the empty string, then the move was a pass.
#     elif 'B' in props:
#         black_move = from_sgf(props.get('B', [''])[0])
#         return pos.play_move(black_move, color=GOPARAMETERS.BLACK)
#     elif 'W' in props:
#         white_move = from_sgf(props.get('W', [''])[0])
#         return pos.play_move(white_move, color=GOPARAMETERS.WHITE)
#     else:
#         return pos


# def add_stones(pos, black_stones_added, white_stones_added):
#     working_board = np.copy(pos.board)
#     go.place_stones(working_board, GOPARAMETERS.BLACK, black_stones_added)
#     go.place_stones(working_board, GOPARAMETERS.WHITE, white_stones_added)
#     new_position = BoardState(board=working_board, n=pos.n, komi=pos.komi,
#                               caps=pos.caps, ko=pos.ko, recent=pos.recent, to_play=pos.to_play)
#     return new_position


def get_next_move(node):
    props = node.next.properties
    if 'W' in props:
        return from_sgf(props['W'][0])
    else:
        return from_sgf(props['B'][0])


def maybe_correct_next(pos, next_node):
    if (('B' in next_node.properties and not pos.to_play == GOPARAMETERS.BLACK) or
            ('W' in next_node.properties and not pos.to_play == GOPARAMETERS.WHITE)):
        pos.flip_playerturn(mutate=True)


# def replay_sgf(sgf_contents):
#     collection = sgf.parse(sgf_contents)
#     game = collection.children[0]
#     props = game.root.properties
#     assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"
#
#     komi = 0
#     if props.get('KM') != None:
#         komi = float(sgf_prop(props.get('KM')))
#     result = parse_game_result(sgf_prop(props.get('RE')))
#
#     pos = BoardState(komi=komi)
#     current_node = game.root
#     while pos is not None and current_node.next is not None:
#         pos = handle_node(pos, current_node)
#         maybe_correct_next(pos, current_node.next)
#         next_move = get_next_move(current_node)
#         yield PositionWithContext(pos, next_move, result)
#         current_node = current_node.next



def translate_gtp_colors(gtp_color):
    if gtp_color == gtp.BLACK:
        return GOPARAMETERS.BLACK
    elif gtp_color == gtp.WHITE:
        return GOPARAMETERS.WHITE
    else:
        return GOPARAMETERS.EMPTY


class GtpInterface(object):
    def __init__(self):
        self.size = 9
        self.position = None
        self.komi = 6.5

    def set_size(self, n):
        if n != GOPARAMETERS.N:
            raise ValueError(("Can't handle boardsize {n}!"
                              "Restart with env var BOARD_SIZE={n}").format(n=n))

    def set_komi(self, komi):
        self.komi = komi
        self.position.komi = komi

    def clear(self):
        if self.position and len(self.position.recent) > 1:
            try:
                sgf = self.to_sgf()
                with open(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M.sgf"), 'w') as f:
                    f.write(sgf)
            except NotImplementedError:
                pass
            except:
                print("Error saving sgf", file=sys.stderr, flush=True)
        self.position = go.BoardState(komi=self.komi)
        self.initialize_game(self.position)

    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    def make_move(self, color, vertex):
        c = from_pygtp(vertex)
        # let's assume this never happens for now.
        # self.accomodate_out_of_turn(color)
        return self.play_move(c)

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        if self.should_resign():
            return gtp.RESIGN
        return to_pygtp(move)

    def final_score(self):
        return self.position.result_string()

    def showboard(self):
        print('\n\n' + str(self.position) + '\n\n', file=sys.stderr)
        return True

    def should_resign(self):
        raise NotImplementedError

    def get_score(self):
        return self.position.result_string()

    def suggest_move(self, position):
        raise NotImplementedError

    def play_move(self, c):
        raise NotImplementedError

    def initialize_game(self):
        raise NotImplementedError

    def chat(self, msg_type, sender, text):
        raise NotImplementedError

    def to_sgf(self):
        raise NotImplementedError
