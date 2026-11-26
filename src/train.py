"""Self-play training of the CNN value network.

Run from the project root:
    python -m src.train selfplay --rounds 10 --epochs 5
    python -m src.train selfplay --rounds 10 --start models/best_cnn.pt   # continue training
"""
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from src.game import blockblast
from src.encode import encode_cnn
from src.network import BoardCNN
from src.search import best_move_sequence, simple_evaluate, play_search_game

# ---------------------------------------------------------------- settings
# These are the settings of CNN run 1, the best run so far (24.5k mean after 10
# rounds, still rising). Change ONE at a time and compare against that run.
LABEL_SCALE = 5000.0   # network predicts score / LABEL_SCALE
GAMMA = 1.0            # discount per round (1.0 = no discount)
N_STEP = 1             # rounds of real points in each target before trusting the network
LOSS = "huber"           # "mse" or "huber"
LR = 3e-4              # learning rate

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
    """Play n_games in parallel.

    Returns (transitions, scores): all transitions in one list, and each game's
    final score (the sum of its round rewards), so the same games double as a
    measurement of how well this network plays.
    """
    with _pool(state) as pool:
        games = pool.map(_play_worker, [beam_width] * n_games)
    transitions = [t for game_transitions, _ in games for t in game_transitions]
    scores = [score for _, score in games]
    return transitions, scores


def mean_score(state, n_games=50, beam_width=3):
    """Average final score of n_games played with the given network weights."""
    with _pool(state) as pool:
        scores = pool.map(_score_worker, [beam_width] * n_games)
    return sum(scores) / len(scores)


# ---------------------------------------------------------------- data

def play_and_record(evaluate=simple_evaluate, beam_width=3, max_rounds=5000):
    """Play one game and turn it into n-step training examples.

    Returns (transitions, final_score). Each transition is
        (state, n_step_return, state_after_n, game_ended, discount)
    where n_step_return is the discounted points of the next N_STEP rounds
    (fewer near the end of the game), state_after_n is the board N_STEP rounds
    later, game_ended says the game died inside that window (nothing to add
    after it), and discount = GAMMA ** (rounds in the window).
    """
    game = blockblast()
    states = [encode_cnn(game)]
    rewards = []
    died = False

    for _ in range(max_rounds):
        score_before = game.global_score
        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        for piece, position in moves:
            game.play_move(piece, position)
        rewards.append(game.global_score - score_before)
        states.append(encode_cnn(game))
        if len(moves) < len(hand):  # some piece could not be placed: game over
            died = True
            break

    T = len(rewards)
    transitions = []
    for t in range(T):
        end = min(t + N_STEP, T)
        ret = sum(GAMMA ** k * rewards[t + k] for k in range(end - t))
        ended = died and end == T
        transitions.append((states[t], ret, states[end], ended, GAMMA ** (end - t)))
    return transitions, game.global_score


# ---------------------------------------------------------------- training

def train_network(model, transitions, epochs=5, lr=LR, batch_size=256):
    """n-step TD training.

    target = discounted points of the next N_STEP rounds
             + GAMMA**n * value of the board N_STEP rounds later (0 if the game died in between)

    Targets are recomputed at the start of every epoch from the network as it is
    then, and held fixed during that epoch while training on shuffled mini-batches.
    """
    boards = torch.from_numpy(np.stack([t[0][0] for t in transitions])).unsqueeze(1)       # (N, 1, 8, 8)
    extras = torch.from_numpy(np.stack([t[0][1] for t in transitions]))                    # (N, 2)
    boards_next = torch.from_numpy(np.stack([t[2][0] for t in transitions])).unsqueeze(1)  # (N, 1, 8, 8)
    extras_next = torch.from_numpy(np.stack([t[2][1] for t in transitions]))               # (N, 2)
    reward = torch.tensor([t[1] / LABEL_SCALE for t in transitions], dtype=torch.float32).unsqueeze(1)
    done = torch.tensor([float(t[3]) for t in transitions], dtype=torch.float32).unsqueeze(1)
    discount = torch.tensor([t[4] for t in transitions], dtype=torch.float32).unsqueeze(1)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss() if LOSS == "huber" else nn.MSELoss()
    n = len(reward)

    for epoch in range(epochs):
        # Targets for this epoch, from the network as it is now, then held fixed
        with torch.no_grad():
            next_value = torch.cat([model(boards_next[i:i + 1024], extras_next[i:i + 1024])
                                    for i in range(0, n, 1024)])
            targets = reward + discount * (1 - done) * next_value

        perm = torch.randperm(n)
        total, batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(boards[idx], extras[idx]), targets[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(f"  epoch {epoch + 1}/{epochs}  loss {total / batches:.5f}  ({n} states)")


# ---------------------------------------------------------------- self-play

def self_play_loop(rounds=10, n_games=500, epochs=5, games_per_round=100,
                   start=None, out="models/best_cnn.pt", buffer=60000):
    print(f"settings: LABEL_SCALE={LABEL_SCALE} GAMMA={GAMMA} N_STEP={N_STEP} LOSS={LOSS} LR={LR}")
    model = BoardCNN()
    if start:
        model.load_state_dict(torch.load(start))
        print(f"continuing from {start}: playing {n_games} starting games")
        data, scores = collect_games(n_games, state=model.state_dict())
        print(f"  starting model mean score {sum(scores) / len(scores):.0f}")
    else:
        print(f"playing {n_games} starting games with the simple bot")
        data, _ = collect_games(n_games)               # round 0: games from the simple bot

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    best = 0

    for r in range(rounds):
        data = data[-buffer:]                          # replay buffer: keep only the newest states
        print(f"Round {r + 1}/{rounds}: training")
        train_network(model, data, epochs=epochs)
        state = model.state_dict()
        new_data, scores = collect_games(games_per_round, state=state)
        data.extend(new_data)
        score = sum(scores) / len(scores)              # the new games double as the measurement
        print(f"Round {r + 1}/{rounds}: network mean score {score:.0f} "
              f"(min {min(scores)}, max {max(scores)}, {len(scores)} games)")
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
    parser.add_argument("--buffer", type=int, default=60000, help="max number of newest states to train on")
    args = parser.parse_args()

    if args.command == "selfplay":
        self_play_loop(rounds=args.rounds, n_games=args.games, epochs=args.epochs,
                       start=args.start, out=args.out, buffer=args.buffer)
    elif args.command == "collect":
        transitions, scores = collect_games(args.games)
        print(len(transitions), "states, mean score", sum(scores) / len(scores))