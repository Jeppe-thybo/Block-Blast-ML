#Purpose of this file is to take a blockblast game from gaem.py and return a list of 65 numbers, one for each tile on the board and one for the combo count. If filled, the number is 1, if not its 0. 
def encode_game(game):
    # Create a list of 64 zeros
    encoded = [0] * 64
    # For each tile on the board, if it's filled, set the corresponding index to 1
    for r in range(8):
        for c in range(8):
            if game.board & (1 << (r * 8 + c)):
                encoded = [(game.board >> i) & 1 for i in range(64)]
    # Add the combo count as the last element
    combo_feature = min(game.combo, 10) / 10
    encoded.append(combo_feature)
    return encoded