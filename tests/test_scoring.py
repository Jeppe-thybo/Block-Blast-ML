
"""Scoring checks against values read from the gameplay videos."""
import pytest
from src.game import blockblast, Pieces
from src.scoring import line_bonus, move_points, next_combo

# (lines cleared in one move, combo before the move, popup seen in the video)
OBSERVED_POPUPS = [
    (1, 0, 24), (1, 1, 48), (1, 2, 71), (1, 3, 95), (1, 4, 118),
    (3, 0, 142), (2, 1, 141), (4, 1, 470), (3, 3, 564),
    (2, 6, 495), (2, 8, 636), (3, 4, 707),
]


@pytest.mark.parametrize("lines, combo, popup", OBSERVED_POPUPS)
def test_bonus_matches_video_popups(lines, combo, popup):
    assert abs(line_bonus(lines, combo) - popup) <= 1


def test_no_bonus_without_clear():
    assert move_points(4, 0, 5) == 4


def test_full_piece_clear_adds_five_per_cell():
    # video: 3x3 piece clears 3 columns at combo 4: 1909 -> 2670
    assert 1909 + move_points(9, 3, 4, piece_fully_cleared=True) == 2670


def test_streak_grows_by_one_per_clearing_move():
    assert next_combo(0, 3) == (1, 0)
    assert next_combo(1, 1) == (2, 0)


def test_streak_survives_two_misses_and_breaks_on_third():
    combo, misses = 4, 0
    combo, misses = next_combo(combo, 0, misses)
    combo, misses = next_combo(combo, 0, misses)
    assert combo == 4
    combo, misses = next_combo(combo, 0, misses)
    assert combo == 0


def test_play_move_detects_fully_cleared_piece():
    # top row has 6 cells filled; a 2-cell horizontal piece completes it and vanishes
    game = blockblast(board=0b00111111)
    piece = Pieces["2_block_H"]
    points = game.play_move(piece, (0, 6))
    assert points == 2 + 24 + 5 * 2
    assert game.combo == 1 and game.misses == 0