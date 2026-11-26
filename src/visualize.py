import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from src.game import blockblast
from src.network import BoardCNN
from src.train import LABEL_SCALE, cnn_evaluator
from src.search import best_move_sequence, simple_evaluate

# Board cell values: 0 empty, 1 filled, 2 piece just placed, 3 cell cleared by this move
BOARD_CMAP = ListedColormap(["white", "#4a7fb5", "#f39c12", "#e74c3c"])
# Hand piece colors by status: 0 still to place, 1 just placed, 2 already used
PIECE_COLORS = {0: "#4a7fb5", 1: "#f39c12", 2: "#cccccc"}

# Free the arrow keys from matplotlib's built-in shortcuts
for key in ("keymap.back", "keymap.forward", "keymap.home"):
    plt.rcParams[key] = []


def mask_to_grid(mask, size=8):
    # bit r*8 + c -> grid[r][c]; works for the board (8x8) and pieces (5x5 corner)
    return np.array([[(mask >> (r * 8 + c)) & 1 for c in range(size)] for r in range(size)])


def make_frame(game, hand, status, round_no, placed=0, cleared=0):
    return {
        "board": game.board,
        "combo": game.combo,
        "score": game.global_score,
        "hand": list(hand),
        "status": list(status),
        "round": round_no,
        "placed": placed,
        "cleared": cleared,
        "game_over": False,
    }




def record_frames(evaluate=simple_evaluate, beam_width=3, max_rounds=1000):
    # One frame when a new hand is dealt, then one frame per placed piece
    game = blockblast()
    frames = []


    for round_no in range(1, max_rounds + 1):
        hand = game.get_hand()
        moves = best_move_sequence(game, hand, evaluate, beam_width)
        status = [0] * len(hand)  # 0 to place, 1 just placed, 2 used
        frames.append(make_frame(game, hand, status, round_no))

        for piece, position in moves:
            before = game.board
            temp = game.copy_game()
            temp.place_piece(piece, position)   # board with the piece, before line clears
            with_piece = temp.board
            game.play_move(piece, position)

            status = [2 if s == 1 else s for s in status]
            slot = next(i for i, p in enumerate(hand) if p == piece and status[i] == 0)
            status[slot] = 1

            frames.append(make_frame(
                game, hand, status, round_no,
                placed=with_piece & ~before,
                cleared=with_piece & ~game.board,
            ))

        if len(moves) < len(hand):  # some piece could not be placed: game over
            frames[-1]["game_over"] = True
            break

    return frames


def style_grid(ax, n):
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_piece(ax, piece, color, slot_size=5):
    # Draw only the piece's own cells, centered in its slot, with no grid around it
    cells = [(r, c) for r in range(5) for c in range(5) if (piece >> (r * 8 + c)) & 1]
    height = max(r for r, _ in cells) + 1
    width = max(c for _, c in cells) + 1
    dr = (slot_size - height) / 2
    dc = (slot_size - width) / 2
    for r, c in cells:
        ax.add_patch(Rectangle((c + dc, r + dr), 1, 1, facecolor=color, edgecolor="black", linewidth=1))
    ax.set_xlim(0, slot_size)
    ax.set_ylim(slot_size, 0)  # row 0 at the top, like the board
    ax.set_aspect("equal")
    ax.axis("off")


def draw_frame(frame, board_ax, hand_axes):
    grid = mask_to_grid(frame["board"])
    grid[mask_to_grid(frame["placed"] & frame["board"]) == 1] = 2
    grid[mask_to_grid(frame["cleared"]) == 1] = 3

    board_ax.clear()
    board_ax.imshow(grid, cmap=BOARD_CMAP, vmin=0, vmax=3)
    style_grid(board_ax, 8)
    title = f"Round {frame['round']}   Score {frame['score']}   Combo {frame['combo']}"
    if frame["game_over"]:
        title += "   GAME OVER"
    board_ax.set_title(title)

    for i, ax in enumerate(hand_axes):
        ax.clear()
        ax.axis("off")
        if i < len(frame["hand"]):
            draw_piece(ax, frame["hand"][i], PIECE_COLORS[frame["status"][i]])


def make_figure():
    # Board on top, the three hand pieces in a row below it, like in the game
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
    board_ax = fig.add_subplot(gs[0, :])
    hand_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    return fig, board_ax, hand_axes


def show_interactive(frames):
    fig, board_ax, hand_axes = make_figure()
    idx = [0]  # a list so the inner functions can change it

    def update():
        draw_frame(frames[idx[0]], board_ax, hand_axes)
        fig.suptitle(f"Frame {idx[0] + 1}/{len(frames)}   (left/right arrows, home/end)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            idx[0] = min(idx[0] + 1, len(frames) - 1)
        elif event.key == "left":
            idx[0] = max(idx[0] - 1, 0)
        elif event.key == "home":
            idx[0] = 0
        elif event.key == "end":
            idx[0] = len(frames) - 1
        else:
            return
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


def save_gif(frames, path, fps=3):
    fig, board_ax, hand_axes = make_figure()

    def update(i):
        draw_frame(frames[i], board_ax, hand_axes)
        fig.suptitle(f"Frame {i + 1}/{len(frames)}")

    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", help="save an animation to this file instead of opening the viewer")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--seed", type=int, help="fix the random hands to replay the same game")
    parser.add_argument("--model", help="path to a trained network; leave out to use the simple bot")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.model:
        model = BoardCNN()
        model.load_state_dict(torch.load(args.model))
        evaluate = cnn_evaluator(model)
    else:
        evaluate = simple_evaluate

    frames = record_frames(evaluate=evaluate, beam_width=args.beam)
    print(f"{len(frames)} frames, final score {frames[-1]['score']}")

    if args.gif:
        save_gif(frames, args.gif)
        print(f"saved {args.gif}")
    else:
        show_interactive(frames)    