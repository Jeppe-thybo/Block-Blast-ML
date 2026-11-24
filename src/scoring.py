"""Scoring model for the Block Blast simulator.

Estimated from gameplay videos, not the game's real formula. Every number here
is a setting, so refine them as more recordings come in.
"""

CELL_POINTS = 1  # points per cell of the placed piece (seen in every video)

# Bonus for clearing N lines (rows + columns) in ONE move, with no combo.
# Observed first clears: 1 line ~24, 2 lines ~47-141, 3 lines ~141-564, 5 lines ~470.
# Values are multiples of 23.5, the unit the observed popups share.
LINE_BONUS = {
    0: 0,
    1: 24,
    2: 94,
    3: 282,
    4: 376,
    5: 470,
    6: 564,
    7: 658,
    8: 752,
    9: 846,
    10: 940,
    11: 1034,
    12: 1128,
    13: 1222,
    14: 1316,
    15: 1410,
    16: 1504,
}

# Each move in a row that clears something raises the bonus multiplier.
# Observed roughly x2 at combo 1, x5 at combo 3, x7-9 at combo 4-5.
COMBO_STEP = 1.5


def combo_multiplier(combo_before: int) -> float:
    """Multiplier for a clear when the streak before this move was combo_before."""
    return 1.0 + COMBO_STEP * combo_before


def line_bonus(lines_cleared: int, combo_before: int) -> int:
    """Bonus points for clearing lines_cleared lines with the given streak."""
    base = LINE_BONUS[min(lines_cleared, max(LINE_BONUS))]
    return round(base * combo_multiplier(combo_before))


def move_points(cells_placed: int, lines_cleared: int, combo_before: int) -> int:
    """Total points for one move: placement plus clear bonus."""
    return CELL_POINTS * cells_placed + line_bonus(lines_cleared, combo_before)


def next_combo(combo_before: int, lines_cleared: int) -> int:
    """Streak after the move: +1 per cleared line, back to 0 if nothing cleared."""
    return combo_before + lines_cleared if lines_cleared else 0