from src.game import blockblast, Pieces
from src.search import best_move_sequence


def test_search_returns_legal_sequence_without_changing_game():
    game = blockblast()
    hand = [Pieces["I3_H"], Pieces["square_2x2"], Pieces["single_block"]]
    moves = best_move_sequence(game, hand)
    assert game.board == 0 and game.global_score == 0     # real game untouched

    replay = blockblast()
    for piece, position in moves:                          # every move must be legal
        replay.play_move(piece, position)
    assert len(moves) == 3


def test_search_takes_the_line_clear():
    # Row 7 has cols 0-4 filled, so I3_H at (7, 5) completes it
    row7 = sum(1 << (7 * 8 + c) for c in range(5))
    game = blockblast(board=row7)
    moves = best_move_sequence(game, [Pieces["I3_H"], Pieces["single_block"], Pieces["single_block"]])

    replay = game.copy_game()
    for piece, position in moves:
        replay.play_move(piece, position)
    assert replay.global_score >= 3 + 24                   # I3_H placed plus a one-line bonus


def test_search_places_what_it_can_when_a_piece_does_not_fit():
    # Checkerboard: no two free cells touch and no line is full,
    # so only the single block can be placed
    board = sum(1 << (r * 8 + c) for r in range(8) for c in range(8) if (r + c) % 2 == 0)
    game = blockblast(board=board)
    hand = [Pieces["square_3x3"], Pieces["single_block"], Pieces["I3_H"]]
    moves = best_move_sequence(game, hand)
    assert len(moves) == 1 and moves[0][0] == Pieces["single_block"]