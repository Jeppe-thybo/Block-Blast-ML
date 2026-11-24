import random
from src.game import blockblast


def play_random_game():
    """Play one game with random legal moves and return the final score."""
    game = blockblast()
    while True:                                  # one pass = one round of 3 pieces
        hand = game.get_hand()
        while hand:                              # until all 3 pieces are placed
            moves = game.legal_moves(hand)
            if not moves:
                return game.global_score         # nothing fits: game over
            piece, position = random.choice(moves)
            game.play_move(piece, position)
            hand.remove(piece)


if __name__ == "__main__":
    scores = [play_random_game() for _ in range(200)]
    print("games:", len(scores), "mean:", sum(scores) / len(scores), "best:", max(scores))