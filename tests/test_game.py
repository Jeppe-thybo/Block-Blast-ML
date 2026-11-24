from src.game import blockblast, Pieces


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