from src.game import blockblast
from src.encode import encode_game

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