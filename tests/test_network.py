import torch
from src.game import blockblast
from src.encode import encode_game
from src.network import NeuralNetwork

def test_network_outputs_one_number():
    net = NeuralNetwork()
    x = torch.tensor(encode_game(blockblast()), dtype=torch.float32)
    out = net(x)
    assert out.shape == (1,)