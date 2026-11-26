"""Self-play training of the CNN value network.

Run from the project root:
    python -m src.train selfplay --rounds 10 --epochs 5
    python -m src.train selfplay --rounds 10 --start models/best_cnn.pt   # continue training
"""
import copy
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from src.game import blockblast
from src.encode import encode_cnn
from src.network import BoardCNN
from src.search import best_move_sequence, simple_evaluate, play_search_game

LABEL_SCALE = 5000.0  # network predicts score / LABEL_SCALE, keeps targets around 0-1


# ---------------------------------------------------------------- evaluation

def cnn_evaluator(model):
    """Turn a BoardCNN into an evaluator for the search.

    evaluate(game) scores one board; evaluate.many(games) scores a whole list in
    one network call, which the search uses to stay fast.
    """
    model.eval()

    def many(games):
        boards, extras = zip(*(encode_cnn(g) for g in games))
        b = torch.from_numpy(np.stack(boards)).unsqueeze(1)   # (N, 1, 8, 8)
        e = torch.from_numpy(np.stack(extras))                # (N, 2)
        with torch.no_grad():
            values = model(b, e).squeeze(1).numpy()
        return [g.global_score + v * LABEL_SCALE for g, v in zip(games, values)]

    def evaluate(game):
        return many([game])[0]

    evaluate.many = many
    return evaluate


# ---------------------------------------------------------------- parallel play
# Each worker process builds its evaluator once (in _init_worker), so the
# network weights are sent to each worker only once per pool, not once per game.

_EVALUATE = None


def _init_worker(state):
    global _EVALUATE
    torch.set_num_threads(1)          # one thread per worker, all cores are already busy
    if state is None:
        _EVALUATE = simple_evaluate   # round 0: the hand-written bot
    else:
        model = BoardCNN()
        model.load_state_dict(state)
        _EVALUATE = cnn_evaluator(model)


def _play_worker(beam_width):
    return play_and_record(_EVALUATE, beam_width)


def _score_worker(beam_width):
    return play_search_game(_EVALUATE, beam_width)


def _pool(state):
    return Pool(os.cpu_count(), initializer=_init_worker, initargs=(state,))


def collect_games(n_games, state=None, beam_width=3):
    """Play n_games in parallel and return all their transitions in one list."""
    with _pool(state) as pool:
        games = pool.map(_play_worker, [beam_width] * n_games)
    return [t for game in games for t in game]


def mean_score(state, n_games=50, beam_width=3):
    """Average final score of n_games played with the given network weights."""
    with _pool(state) as pool:
        scores = pool.map(_score_worker, [beam_width] * n_games)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------- data

def play_and_record(evaluate=simple_evaluate, beam_width=3, max_rounds=1000):
    """Play one game. Returns a list of (state, reward, next_state, done) per round,
    where a state is the pair (board 8x8, extras [combo, misses]) from encode_cnn."""
    game = blockblast()
    transitions = []

    for _ in range(max_rounds):
        state = encode_cnn(game)
        score_before = game.global_score

        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        for piece, position in moves:
            game.play_move(piece, position)

        done = len(moves) < len(hand)  # some piece could not be placed: game over
        transitions.append((state, game.global_score - score_before, encode_cnn(game), done))

        if done:
            break

    return transitions


# ---------------------------------------------------------------- training

def train_network(model, transitions, epochs=5, lr=3e-4, batch_size=256):
    """TD training: target = reward of the round + value of the next board (0 at game over).

    Trains on shuffled mini-batches. The target network is a frozen copy that is
    refreshed at the start of every epoch, so the target holds still while we learn.
    """
    boards = torch.from_numpy(np.stack([t[0][0] for t in transitions])).unsqueeze(1)       # (N, 1, 8, 8)
    extras = torch.from_numpy(np.stack([t[0][1] for t in transitions]))                    # (N, 2)
    boards_next = torch.from_numpy(np.stack([t[2][0] for t in transitions])).unsqueeze(1)  # (N, 1, 8, 8)
    extras_next = torch.from_numpy(np.stack([t[2][1] for t in transitions]))               # (N, 2)
    reward = torch.tensor([t[1] / LABEL_SCALE for t in transitions], dtype=torch.float32).unsqueeze(1)
    done = torch.tensor([float(t[3]) for t in transitions], dtype=torch.float32).unsqueeze(1)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    target_model = copy.deepcopy(model)
    target_model.eval()
    n = len(reward)

    for epoch in range(epochs):
        target_model.load_state_dict(model.state_dict())
        perm = torch.randperm(n)                       # new random order each epoch
        total, batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            with torch.no_grad():
                target = reward[idx] + (1 - done[idx]) * target_model(boards_next[idx], extras_next[idx])
            prediction = model(boards[idx], extras[idx])
            loss = loss_fn(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / batches:.5f}  ({n} states)")


# ---------------------------------------------------------------- self-play

def self_play_loop(rounds=10, n_games=500, epochs=5, games_per_round=100,
                   start=None, out="models/best_cnn.pt"):
    model = BoardCNN()
    if start:
        model.load_state_dict(torch.load(start))
        print(f"continuing from {start}")
        data = collect_games(n_games, state=model.state_dict())
    else:
        data = collect_games(n_games)                  # round 0: games from the simple bot

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0

    for r in range(rounds):
        print(f"Round {r + 1}/{rounds}: training")
        train_network(model, data, epochs=epochs)
        state = model.state_dict()
        data.extend(collect_games(games_per_round, state=state))
        score = mean_score(state)
        print(f"Round {r + 1}/{rounds}: network mean score {score:.0f}")
        if score > best:
            best = score
            torch.save(state, out)
            print(f"  new best, saved {out}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["selfplay", "collect"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--games", type=int, default=500, help="simple-bot games to start from")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--start", help="saved model to continue training from")
    parser.add_argument("--out", default="models/best_cnn.pt")
    args = parser.parse_args()

    if args.command == "selfplay":
        self_play_loop(rounds=args.rounds, n_games=args.games, epochs=args.epochs,
                       start=args.start, out=args.out)
    elif args.command == "collect":
        print(len(collect_games(args.games)))