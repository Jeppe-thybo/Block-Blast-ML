"""Scoring model for the Block Blast simulator.

Fitted to 8 gameplay recordings (about 9 minutes of play). It reproduces every
clear popup I could read in them to within 1 point, and the full score change of
every move I could check end to end.

Points for one move:

    cells placed                                   (1 point per cell, always)
  + BONUS_UNIT * T(lines) * streak                 (only if lines were cleared)
  + FULL_CLEAR_CELL_BONUS * cells placed           (only if the whole piece was cleared away)

where T(n) = 1 + 2 + ... + n = n(n+1)/2, so 1, 3, 6, 10, 15 ... for 1, 2, 3, 4, 5 lines,
and streak is 1 for the first clearing move, 2 for the next, and so on.
"""

CELL_POINTS = 1

# One "unit" of clear bonus. Every popup in the videos is a multiple of it.
BONUS_UNIT = 23.55

# Each placed cell scores this much extra if the move clears the entire piece,
# i.e. none of its cells are left on the board afterwards (shown as a "5" on each cell).
FULL_CLEAR_CELL_BONUS = 5

# The streak survives this many placements in a row without a clear.
# One more non-clearing placement than this resets it.
MAX_MISSES = 2


def triangle(n: int) -> int:
    """1 + 2 + ... + n: how clearing several lines at once multiplies the bonus."""
    return n * (n + 1) // 2


def line_bonus(lines_cleared: int, combo_before: int) -> int:
    """Clear bonus. combo_before = clearing moves already in the current streak."""
    if lines_cleared == 0:
        return 0
    streak = combo_before + 1
    return int(BONUS_UNIT * triangle(lines_cleared) * streak + 0.5)  # round half up, like the game


def move_points(cells_placed: int, lines_cleared: int, combo_before: int,
                piece_fully_cleared: bool = False) -> int:
    """Total points for one move."""
    points = CELL_POINTS * cells_placed + line_bonus(lines_cleared, combo_before)
    if lines_cleared and piece_fully_cleared:
        points += FULL_CLEAR_CELL_BONUS * cells_placed
    return points


def next_combo(combo_before: int, lines_cleared: int, misses_before: int = 0):
    """Return (combo, misses) after the move.

    combo:  clearing moves in the current streak (0 = no streak).
            +1 per clearing move, no matter how many lines it clears.
    misses: placements in a row without a clear since the last clear.
            The streak is lost on the (MAX_MISSES + 1)th miss.
    """
    if lines_cleared:
        return combo_before + 1, 0
    misses = misses_before + 1
    if misses > MAX_MISSES:
        return 0, 0
    return combo_before, misses