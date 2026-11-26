#Purpose of this file is to take a blockblast game from gaem.py and return a list of 65 numbers, one for each tile on the board and one for the combo count. If filled, the number is 1, if not its 0. 
import numpy as np

def encode_game(game):
    bits = np.unpackbits(
        np.frombuffer(int(game.board).to_bytes(8, "little"), dtype=np.uint8),
        bitorder="little",
    )
    encoded = bits.tolist()
    encoded.append(min(game.combo, 10) / 10)
    return encoded


def encode_cnn(game):
    # returns (board, extras)
    # board:  8x8 numpy array of 0.0 / 1.0, row r and column c = bit r*8 + c
    # extras: numpy array of 2 numbers: scaled combo, scaled misses
    bits = np.unpackbits(
        np.frombuffer(int(game.board).to_bytes(8, "little"), dtype=np.uint8),
        bitorder="little",
    )
    board = bits.reshape(8, 8).astype(np.float32)
    extras = np.array([min(game.combo, 10) / 10, game.misses / 2], dtype=np.float32)
    return board, extras

