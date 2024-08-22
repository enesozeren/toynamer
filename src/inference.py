import torch
import numpy as np
import argparse
from src.utils import char_to_onehot
from src.variables import int_to_char

def generate_name(model, temperature=0.5):
    model.eval()
    with torch.no_grad():
        hidden = model.initHidden()
        name = ''
        char = '<S>'
        while char != '<E>':
            input_tensor = char_to_onehot(char)
            output, hidden = model(input_tensor, hidden)

            # Adjust the probabilities with temperature
            adj_outputs = output[0].numpy() / temperature
            # Apply softmax to get the probability distribution
            probs = np.exp(adj_outputs) / np.sum(np.exp(adj_outputs))
            
            # Sample a character index based on the probabilities
            char_idx = np.random.choice(len(probs), p=probs)
            char = int_to_char[char_idx]
            if char != '<E>': name += char
    return name

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, required=True)
    argparser.add_argument('--temperature', type=float, default=0.5)
    args = argparser.parse_args()

    toynamer_model = torch.load(args.model_path, map_location=torch.device('cpu'))
    generated_name = generate_name(model=toynamer_model, temperature=args.temperature)

    print(f'Generated Name: {generated_name}')