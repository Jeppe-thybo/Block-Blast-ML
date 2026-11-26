from src.game import blockblast
from src.encode import encode_game, encode_cnn

def test_empty_game_gives_65_zeros_for_encode():
    game = blockblast()
    encoded = encode_game(game)
    assert len(encoded) == 65
    assert all(x == 0 for x in encoded)

def test_blockblast_0b101_has_correct_index_for_encode():
    game = blockblast()
    game.board = 0b101
    encoded = encode_game(game)
    assert len(encoded) == 65
    assert encoded[0] == 1
    assert encoded[2] == 1
    assert all(x == 0 for i, x in enumerate(encoded) if i not in (0, 2))

def test_blockblast_combo_20_yields_last_element_of_1():
    game = blockblast()
    game.combo = 20
    encoded = encode_game(game)
    assert len(encoded) == 65
    assert encoded[64] == 1
    assert all(x == 0 for i, x in enumerate(encoded) if i != 64)


def test_blockblast_board_0b101_yields_correct_board_for_encode_cnn():
    game = blockblast()
    game.board = 0b101
    board, extras = encode_cnn(game)
    assert board.shape == (8, 8)
    assert board[0, 0] == 1
    assert board[0, 2] == 1
    assert all(board[i, j] == 0 for i in range(8) for j in range(8) if (i, j) not in ((0, 0), (0, 2)))
    assert extras.shape == (2,)
    assert extras[0] == 0.0
    assert extras[1] == 0.0

def test_blockblast_board_1_less_than_63_yields_correct_board_for_encode_cnn():
    game = blockblast()
    game.board = 0b1 << 62
    board, extras = encode_cnn(game)
    assert board.shape == (8, 8)
    assert board[7, 6] == 1
    assert all(board[i, j] == 0 for i in range(8) for j in range(8) if (i, j) != (7, 6))
    assert extras.shape == (2,)
    assert extras[0] == 0.0
    assert extras[1] == 0.0

def test_blockblast_combo_and_misses_yields_correct_extras_for_encode_cnn():
    game = blockblast()
    game.combo = 5
    game.misses = 1
    board, extras = encode_cnn(game)
    assert extras.shape == (2,)
    assert extras[0] == 0.5
    assert extras[1] == 0.5
