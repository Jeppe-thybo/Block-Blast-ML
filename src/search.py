"""Beam search over the order and placement of the three pieces in a hand."""
import itertools


def simple_evaluate(game):
    """Placeholder board score, higher is better. A neural network replaces this later."""
    filled = bin(game.board).count("1")
    return game.global_score + 3 * (64 - filled)


def _placements(game, piece):
    return [(r, c) for r in range(8) for c in range(8)
            if game.can_place_piece(piece, (r, c))]


def best_move_sequence(game, hand, evaluate=simple_evaluate, beam_width=5):
    """Return the best list of (piece, position) moves for this hand.

    Tries every order of the pieces. For each piece it tries every legal
    placement on a copy of the game, then keeps only the beam_width best
    partial results (by evaluate) before placing the next piece. A piece with
    no legal placement is skipped. Sequences that place more pieces win,
    then the higher evaluation wins. The real game is never changed.
    """
    best_key, best_moves = None, []
    for order in set(itertools.permutations(hand)):
        states = [(game.copy_game(), [])]
        for piece in order:
            candidates = []
            for g, moves in states:
                for pos in _placements(g, piece):
                    g2 = g.copy_game()
                    g2.play_move(piece, pos)
                    candidates.append((evaluate(g2), g2, moves + [(piece, pos)]))
            if not candidates:
                continue                      # this piece fits nowhere, skip it
            candidates.sort(key=lambda c: c[0], reverse=True)
            states = [(g, m) for _, g, m in candidates[:beam_width]]
        for g, moves in states:
            key = (len(moves), evaluate(g))
            if best_key is None or key > best_key:
                best_key, best_moves = key, moves
    return best_moves


def play_search_game(evaluate=simple_evaluate, beam_width=5, max_rounds=1000):
    """Play one full game using the search and return the final score."""
    from src.game import blockblast
    game = blockblast()
    for _ in range(max_rounds):
        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        for piece, position in moves:
            game.play_move(piece, position)
        if len(moves) < len(hand):            # some piece could not be placed
            break
    return game.global_score