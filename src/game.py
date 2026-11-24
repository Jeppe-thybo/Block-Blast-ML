#Game mechanics for Block Blast usign 64 bit integer bitboard to represent the game state. Each bit in the 64-bit integer represents a cell in the 8x8 grid, where a bit value of 1 indicates a block is present and 0 indicates an empty cell. The game logic includes functions for placing blocks, checking for completed lines, and updating the game state accordingly.

import numpy as np
import random

from src.scoring import move_points, next_combo

def make_piece(cells):
    mask = 0
    for i, j in cells:
        bit_number = i * 8 + j
        mask |= (1 << bit_number)
    return mask


Pieces = {
    "I3_H": make_piece([(0, 0), (0, 1), (0, 2)]),
    "I3_V": make_piece([(0, 0), (1, 0), (2, 0)]),
    "I4_H": make_piece([(0, 0), (0, 1), (0, 2), (0, 3)]),
    "I4_V": make_piece([(0, 0), (1, 0), (2, 0), (3, 0)]),
    "square_2x2": make_piece([(0, 0), (0, 1), (1, 0), (1, 1)]),
    "square_3x3": make_piece([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]),
    "L_right": make_piece([(0, 0), (1, 0), (2, 0), (2, 1)]),
    "L_left": make_piece([(0, 1), (1, 1), (2, 1), (2, 0)]),
    "T_up": make_piece([(0,1), (1, 0), (1, 1), (1, 2)]),
    "T_down": make_piece([(0, 0), (0, 1), (0, 2), (1, 1)]),
    "T_right": make_piece([(0, 0), (1, 0), (2, 0), (1, 1)]),
    "T_left": make_piece([(0, 1), (1, 1), (2, 1), (1, 0)]),
    "single_block": make_piece([(0, 0)]),
    "2_block_H": make_piece([(0, 0), (0, 1)]),
    "2_block_V": make_piece([(0, 0), (1, 0)]),

    "I5_H": make_piece([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    "I5_V": make_piece([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),

    # rectangles
    "rect_2x3": make_piece([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
    "rect_3x2": make_piece([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]),

    # small corner, 3 blocks (a 2x2 with one cell missing)
    "corner_tl": make_piece([(0, 0), (0, 1), (1, 0)]),
    "corner_tr": make_piece([(0, 0), (0, 1), (1, 1)]),
    "corner_bl": make_piece([(0, 0), (1, 0), (1, 1)]),
    "corner_br": make_piece([(0, 1), (1, 0), (1, 1)]),

    # big corner, 5 blocks (3 high and 3 wide)
    "big_corner_tl": make_piece([(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]),
    "big_corner_tr": make_piece([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]),
    "big_corner_bl": make_piece([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),
    "big_corner_br": make_piece([(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]),

    # the other 2 tall L/J orientations (you already have L_right and L_left)
    "L_hook_left": make_piece([(0, 0), (0, 1), (1, 0), (2, 0)]),
    "L_hook_right": make_piece([(0, 0), (0, 1), (1, 1), (2, 1)]),

    # flat L/J, 3 wide and 2 tall
    "L_flat_1": make_piece([(0, 0), (0, 1), (0, 2), (1, 0)]),
    "L_flat_2": make_piece([(0, 0), (0, 1), (0, 2), (1, 2)]),
    "L_flat_3": make_piece([(0, 2), (1, 0), (1, 1), (1, 2)]),
    "L_flat_4": make_piece([(0, 0), (1, 0), (1, 1), (1, 2)]),

    # lightning shapes (S and Z)
    "Z_H": make_piece([(0, 0), (0, 1), (1, 1), (1, 2)]),
    "S_H": make_piece([(0, 1), (0, 2), (1, 0), (1, 1)]),
    "Z_V": make_piece([(0, 1), (1, 0), (1, 1), (2, 0)]),
    "S_V": make_piece([(0, 0), (1, 0), (1, 1), (2, 1)]),
}


class blockblast:
    def __init__(self, board = 0, combo = 0, global_score = 0):
        self.board =  board # Initialize the game board as a 64-bit integer bitboard
        self.combo = combo # Initialize the combo count
        self.global_score = global_score # Initialize the score

    def play_move(self, piece, position):
        # Play a move by placing a piece and updating the game state
        if not self.can_place_piece(piece, position):
            raise ValueError("Cannot place piece at the given position.")

        self.place_piece(piece, position)
        completed_lines = self.check_completed_lines()
        self.clear_completed_lines(completed_lines)

        # Update score and combo based on the move
        cells_placed = bin(piece).count('1')
        lines_cleared = len(completed_lines)
        move_score = move_points(cells_placed, lines_cleared, self.combo)
        self.global_score += move_score
        self.combo = next_combo(self.combo, lines_cleared)

        return move_score

    def game_over(self, hand):
        #Given the 3 pieces and the current state of the board, check if all 3 pieces can be placed, including potential clearances.
        for piece in hand:
            for row in range(8):
                for col in range(8):
                    if self.can_place_piece(piece, (row, col)):
                        return False  # Found a valid placement
        return True  # No valid placements found, game over

    def copy_game(self):
        # Create a copy of the current game state
        return blockblast(self.board, self.combo, self.global_score)

    def place_piece(self, piece, position):
        # Place a piece on the board at the specified position
        # piece: 64-bit integer representing the piece shape
        # position: (row, col) tuple indicating where to place the piece
        row, col = position
        for i in range(8):
            for j in range(8):
                if (piece >> (i * 8 + j)) & 1:
                    self.board |= (1 << ((row + i) * 8 + (col + j)))

    def check_completed_lines(self):
        # Check for completed lines (rows) in the board
        completed_lines = []
        for i in range(8):
            if (self.board >> (i * 8)) & 0xFF == 0xFF:  # Check if the entire row is filled
                completed_lines.append(i)
        for i in range(8):
            if (self.board >> i) & 0x0101010101010101 == 0x0101010101010101:  # Check if the entire column is filled
                completed_lines.append(i + 8)  # Add 8 to distinguish columns from rows
        return completed_lines

    def clear_completed_lines(self, completed_lines):
        # Clear the completed lines from the board 
        for line in completed_lines:
            if line < 8:  # Row
                self.board &= ~(0xFF << (line * 8))  # Clear the row
            else:  # Column
                col = line - 8
                self.board &= ~(0x0101010101010101 << col)  # Clear the column

    def can_place_piece(self, piece, position):
        # Check if a piece can be placed at the specified position without overlapping existing blocks, and without going out of bounds
        row, col = position
        for i in range(8):
            for j in range(8):
                if (piece >> (i * 8 + j)) & 1:
                    if row + i >= 8 or col + j >= 8:  # Check bounds
                        return False
                    if (self.board >> ((row + i) * 8 + (col + j))) & 1:  # Check overlap
                        return False
        return True
        

    def show_board(self):
        # Convert the 64-bit integer bitboard to a 2D numpy array for display
        board_array = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                if (self.board >> (i * 8 + j)) & 1:
                    board_array[i][j] = 1
        return board_array

    def get_global_score(self):
        # Return the current global score
        return self.global_score


    def get_hand(self):
        # Generate a random hand of 3 pieces from the available pieces
        return random.choices(list(Pieces.values()), k=3)

    def legal_moves(self, hand):
        # Return a list of legal moves for the given hand of pieces
        legal_moves = []
        for piece in hand:
            for row in range(8):
                for col in range(8):
                    if self.can_place_piece(piece, (row, col)):
                        legal_moves.append((piece, (row, col)))
        return legal_moves
