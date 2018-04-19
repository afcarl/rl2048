import numpy as np
from rl2048.game import (Game2048, board_down, board_left, board_right,
                         board_up, compress_left)

no_move_board = np.array(
    [[2, 4, 2, 4],
     [4, 2, 4, 2],
     [2, 4, 2, 4],
     [4, 2, 4, 2]]
)


def test_compress_left():

    assert compress_left([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

    assert compress_left([16]) == ([16], 0)

    assert compress_left([2, 2, 4, 16]) == ([4, 4, 16], 4)

    assert compress_left([32, 32, 32, 32]) == ([64, 64], 128)

    assert compress_left([64, 128, 128, 16]) == ([64, 256, 16], 256)


def check_board_move(before_left, after_left, score, move):

    if move == 'left':
        processed, computed_score = board_left(before_left)
        expected = after_left

    elif move == 'right':
        before = np.rot90(before_left, 2)
        processed, computed_score = board_right(before)
        expected = np.rot90(after_left, 2)

    elif move == 'up':
        before = np.rot90(before_left, -1)
        processed, computed_score = board_up(before)
        expected = np.rot90(after_left, -1)

    elif move == 'down':
        before = np.rot90(before_left, 1)
        processed, computed_score = board_down(before)
        expected = np.rot90(after_left, 1)

    np.testing.assert_array_equal(processed, expected)
    assert score == computed_score


def check_all_moves(before, after_left, score):

    check_board_move(before, after_left, score, 'left')
    check_board_move(before, after_left, score, 'right')
    check_board_move(before, after_left, score, 'up')
    check_board_move(before, after_left, score, 'down')


def test_board_moves():

    a = np.array(
        [[2,  4,  4, 16],
         [16, 16, 8,  8],
         [8,  -1, 8,  2],
         [2,   2, 2,  2]]
    )

    b = np.array(
        [[2,   8, 16, -1],
         [32, 16, -1, -1],
         [16,  2, -1, -1],
         [4,   4, -1, -1]],
    )

    check_all_moves(a, b, 8 + 32 + 16 + 16 + 4 + 4)

    a = np.array(
        [[2,   4,  8,  16],
         [2,   4,  8,  16],
         [8,  16, 32,  64],
         [16, 32,  2,   8]]
    )

    check_all_moves(a, a, 0)

    a = np.array(
        [[16, -1, -1, 16],
         [2,  -1,  2,  2],
         [4,   8,  8,  2],
         [8,   8,  8, 16]]
    )

    b = np.array(
        [[32, -1, -1, -1],
         [4,   2, -1, -1],
         [4,  16,  2, -1],
         [16,  8, 16, -1]]
    )

    check_all_moves(a, b, 32 + 4 + 16 + 16)


def test_all_collapse():
    a = np.ones((4, 4), dtype=np.int)*2

    a, score = board_left(a)
    assert score == 32

    a, score = board_right(a)
    assert score == 32

    a, score = board_up(a)
    assert score == 32

    a, score = board_down(a)
    assert score == 32

    a, _ = board_left(a)
    a, _ = board_up(a)

    assert a[0][0] == 32

    a[0, 0] = -1

    assert np.all(a == -1)


def test_no_move():

    a = no_move_board
    b, score = board_left(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_right(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_up(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_down(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0


def test_game_cannot_move():

    game = Game2048()
    game.board = no_move_board

    assert not game.can_move_left()
    assert not game.can_move_right()
    assert not game.can_move_up()
    assert not game.can_move_down()

    game.move_left()
    assert game.done


def test_game_no_move():

    a = no_move_board
    b, score = board_left(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_right(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_up(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0

    b, score = board_down(a)
    np.testing.assert_array_equal(a, b)
    assert score == 0


def test_game_can_move():

    game = Game2048()
    game.board = np.array(
        [[2, 4, 4, 8],
         [2, 2, 8, 2],
         [2, 4, 8, 8],
         [2, 2, 2, 2]]
    )

    assert game.can_move_left()
    assert game.can_move_right()
    assert game.can_move_up()
    assert game.can_move_down()


def test_random_stats():

    game = Game2048()
    rng = np.random.RandomState(13)

    for i in range(1000):
        game.board = rng.choice([2, 4, 8], (4, 4))

        assert game.can_move_left() == game.can_move_right()
        assert game.can_move_up() == game.can_move_down()

        move = [game.move_left, game.move_right,
                game.move_up, game.move_down][rng.randint(4)]

        sum_before = np.sum(game.board[game.board > 0])
        move()
        sum_after = np.sum(game.board[game.board > 0])

        assert (sum_after - sum_before) in [0, 2, 4]
