from src.game import blockblast
from src.encode import encode_game
from src.search import best_move_sequence, simple_evaluate


def play_and_record(evaluate=simple_evaluate, beam_width=3, max_rounds=1000):
    game = blockblast()
    states = []  # (encoded_state, score at that moment)

    for _ in range(max_rounds):
        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        for piece, position in moves:
            game.play_move(piece, position)

        # snapshot after all pieces of the round are placed
        states.append((encode_game(game), game.global_score))

        if len(moves) < len(hand):  # some piece could not be placed: game over
            break

    final_score = game.global_score
    # label = score that still came after that state
    return [(state, final_score - score) for state, score in states]