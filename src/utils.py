import torch
from src.variables import VOCAB_SIZE, char_to_int, int_to_char

def char_to_onehot(char):
    one_hot = torch.zeros(1, VOCAB_SIZE)
    one_hot[0, char_to_int[char]] = 1
    return one_hot

def onehot_to_char(one_hot):
    argmax = torch.argmax(one_hot).item()
    return int_to_char[argmax]