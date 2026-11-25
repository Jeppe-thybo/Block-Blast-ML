from src.game import blockblast
from src.encode import encode_game
from src.network import NeuralNetwork
from src.search import best_move_sequence, simple_evaluate, play_search_game
import torch
import torch.nn as nn
import numpy as np
import os
from multiprocessing import Pool


def extract_weights(model):
    layers = [m for m in model.layers if isinstance(m, nn.Linear)]
    return [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in layers]


def evaluator_from_weights(weights):
    def evaluate(game):
        x = np.array(encode_game(game), dtype=np.float32)
        for W, b in weights[:-1]:
            x = np.maximum(x @ W + b, 0)
        W, b = weights[-1]
        return game.global_score + (x @ W + b).item() * LABEL_SCALE
    return evaluate


def _play_worker(args):
    weights, beam_width = args
    evaluate = simple_evaluate if weights is None else evaluator_from_weights(weights)
    return play_and_record(evaluate, beam_width)


def _score_worker(args):
    weights, beam_width = args
    return play_search_game(evaluator_from_weights(weights), beam_width)


def collect_games(n_games, weights=None, beam_width=3):
    with Pool(os.cpu_count()) as pool:
        games = pool.map(_play_worker, [(weights, beam_width)] * n_games)
    return [t for game in games for t in game]


def mean_score(weights, n_games=50, beam_width=3):
    with Pool(os.cpu_count()) as pool:
        scores = pool.map(_score_worker, [(weights, beam_width)] * n_games)
    return sum(scores) / len(scores)


def play_and_record(evaluate=simple_evaluate, beam_width=3, max_rounds=1000):
    game = blockblast()
    transitions = []  # (state, reward, next_state, done)

    for _ in range(max_rounds):
        state = encode_game(game)
        score_before = game.global_score

        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        for piece, position in moves:
            game.play_move(piece, position)

        done = len(moves) < len(hand)  # some piece could not be placed: game over
        transitions.append((state, game.global_score - score_before, encode_game(game), done))

        if done:
            break

    return transitions


LABEL_SCALE = 1000.0  # scale labels to [0, 1] range for better training stability

def train_network(model, transitions, epochs=100, lr=0.001):
    X = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
    Reward = torch.tensor([t[1] / LABEL_SCALE for t in transitions], dtype=torch.float32).unsqueeze(1)
    X_next = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
    game_status = torch.tensor([float(t[3]) for t in transitions], dtype=torch.float32).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        with torch.no_grad():
            target = Reward + (1 - game_status) * model(X_next)
        prediction = model(X)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def self_play_loop(rounds=5, n_games=500, epochs=100):
    model = NeuralNetwork()
    data = collect_games(n_games)                      # round 0: simple bot

    for r in range(rounds):
        train_network(model, data, epochs=epochs)
        weights = extract_weights(model)
        data.extend(collect_games(100, weights=weights))
        print(f"Round {r + 1}/{rounds}: network mean score {mean_score(weights):.0f}")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["selfplay", "collect"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.command == "selfplay":
        self_play_loop(rounds=args.rounds, n_games=args.games, epochs=args.epochs)
    elif args.command == "collect":
        print(len(collect_games(args.games)))