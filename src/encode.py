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