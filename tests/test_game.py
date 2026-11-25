from src.game import blockblast, Pieces
from src.scoring import next_combo


def test_pieces_are_unique():
    values = list(Pieces.values())
    assert len(values) == len(set(values))


def test_known_piece_values():
    # Check that known pieces have the expected bitboard values
    assert Pieces["single_block"] == 1
    assert Pieces["I3_H"] == 7
    assert Pieces["I3_V"] == 65793


def test_single_block_at_origin():
    game = blockblast()                                # 1. set up: an empty board (board = 0)
    game.place_piece(Pieces["single_block"], (0, 0))   # 2. do: place a piece at row 0, col 0
    assert game.board == 1                             # 3. check: bit 0 is on, so the number is 1


def test_full_row_or_column_completion():
    game = blockblast()

    # Fill the first row
    for col in range(8):
        game.place_piece(Pieces["single_block"], (0, col))
    assert game.check_completed_lines() == [0]

    # New board, fill the first column
    game = blockblast()
    for row in range(8):
        game.place_piece(Pieces["single_block"], (row, 0))
    assert game.check_completed_lines() == [8]


def test_full_column_and_row_completion():
    game = blockblast()

    # Fill the first row and the first column
    for col in range(8):
        game.place_piece(Pieces["single_block"], (0, col))
    for row in range(8):
        game.place_piece(Pieces["single_block"], (row, 0))

    completed_lines = game.check_completed_lines()
    assert 0 in completed_lines   # first row completed
    assert 8 in completed_lines   # first column completed

    # Clearing both lines leaves an empty board
    game.clear_completed_lines(completed_lines)
    assert game.board == 0


def test_clear_leaves_other_blocks():
    game = blockblast()
    for col in range(8):
        game.place_piece(Pieces["single_block"], (0, col))
    game.place_piece(Pieces["single_block"], (3, 3))   # a block outside the full row

    game.clear_completed_lines(game.check_completed_lines())
    expected = blockblast()
    expected.place_piece(Pieces["single_block"], (3, 3))
    assert game.board == expected.board


def test_place_piece_out_of_bounds():
    game = blockblast()
    assert not game.can_place_piece(Pieces["2_block_H"], (7, 7))
    assert not game.can_place_piece(Pieces["I3_H"], (0, 6))
    assert not game.can_place_piece(Pieces["I3_V"], (6, 0))


def test_place_piece_overlap():
    game = blockblast()
    game.place_piece(Pieces["single_block"], (0, 0))
    assert not game.can_place_piece(Pieces["single_block"], (0, 0))
    assert game.can_place_piece(Pieces["single_block"], (0, 1))


def test_that_play_move_returns_move_score_and_not_global_score():
    game = blockblast()
    first = game.play_move(Pieces["single_block"], (0, 0))
    second = game.play_move(Pieces["I3_H"], (3, 0))
    assert first == 1
    assert second == 3                  # this move only, not the running total
    assert game.global_score == 4       # the total of both moves


def test_play_move_rejects_impossible_move():
    game = blockblast()
    try:
        game.play_move(Pieces["I3_H"], (0, 6))   # sticks out past the right edge
        assert False, "expected a ValueError"
    except ValueError:
        pass
    assert game.board == 0              # nothing changed
    assert game.global_score == 0


def test_next_combo():
    # next_combo returns (combo, misses)
    assert next_combo(0, 1) == (1, 0)       # a clearing move starts a streak
    assert next_combo(1, 2) == (2, 0)       # +1 per clearing move, however many lines
    assert next_combo(3, 0, 0) == (3, 1)    # first miss: streak kept
    assert next_combo(3, 0, 1) == (3, 2)    # second miss: still kept
    assert next_combo(3, 0, 2) == (0, 0)    # third miss in a row resets it


def test_that_combo_updates_correctly():
    # Bits 3-7 of row 0 are filled, so I3_H at (0, 0) completes the row
    game = blockblast(board=0b11111000)
    assert game.combo == 0

    game.play_move(Pieces["I3_H"], (0, 0))
    assert game.board == 0              # the row cleared, nothing else was there
    assert game.combo == 1

    # The streak survives two moves without a clear...
    game.play_move(Pieces["single_block"], (5, 5))
    game.play_move(Pieces["single_block"], (6, 6))
    assert game.combo == 1
    assert game.misses == 2

    # ...and resets on the third
    game.play_move(Pieces["single_block"], (7, 7))
    assert game.combo == 0


def test_clearing_move_earns_bonus_and_streak_raises_it():
    game = blockblast(board=0b11111000)
    first = game.play_move(Pieces["I3_H"], (0, 0))      # 3 cells + 1 line, no streak yet
    assert first == 3 + 24 + 5 * 3                       # the whole piece was cleared: +5 per cell

    game.board = 0b11111000                              # set up another one-line clear
    second = game.play_move(Pieces["I3_H"], (0, 0))      # streak of 1 now raises the bonus
    assert second > first


def test_game_over_condition():
    game = blockblast()

    # Fill the board completely
    for row in range(8):
        for col in range(8):
            game.place_piece(Pieces["single_block"], (row, col))

    # Now the game should be over since no pieces can be placed
    assert game.game_over([Pieces["single_block"], Pieces["I3_H"], Pieces["I3_V"]]) is True

    # Clear a line to make space
    completed_lines = game.check_completed_lines()
    game.clear_completed_lines(completed_lines)

    # Now the game should not be over since at least one piece can be placed
    assert game.game_over([Pieces["single_block"], Pieces["I3_H"], Pieces["I3_V"]]) is False


def test_game_over_with_one_free_cell():
    game = blockblast(board=((1 << 64) - 1) ^ 1)         # only cell (0, 0) is free
    assert not game.game_over([Pieces["single_block"], Pieces["I3_H"], Pieces["I5_H"]])
    assert game.game_over([Pieces["I3_H"], Pieces["I5_H"], Pieces["I3_V"]])