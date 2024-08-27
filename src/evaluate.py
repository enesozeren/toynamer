import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
from src.rnn_model import RNN
from src.variables import VOCAB_SIZE
from src.utils import char_to_onehot
import json

def prepare_val_data(val_data_path):

    with open(val_data_path, 'r') as file:
        val_name_list = file.read().split('\n')
    
    # Create validation dataset for predicting the next character
    val_dataset = []
    for name in val_name_list:
        name_char_list = list(['<S>']) + list(name) + list(['<E>'])
        for i in range(1, len(name_char_list)):
            input = [char for char in name_char_list[:i]]
            output = [name_char_list[i]]
            val_dataset.append((input, output))

    return val_dataset


def evaluate(validation_dataset, model_path,
             output_directory_path):
    '''
    Evaluate the model with the given dataset
    '''

    # Create a new folder in the output directory with current datetime to save the model and logs
    output_folder_path = f'{output_directory_path}/evaluate_run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(output_folder_path)

    # Get the model weights
    toynamer_model = torch.load(model_path, map_location=torch.device('cpu'))
    toynamer_model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    validation_loss = 0
    with torch.no_grad():
        for input_seq, target in validation_dataset:
            hidden = toynamer_model.initHidden()

            for char in input_seq:
                input_tensor = char_to_onehot(char)
                output, hidden = toynamer_model(input_tensor, hidden)
            
            output_tensor = char_to_onehot(target[0])

            loss = loss_fn(output, output_tensor)
            validation_loss += loss.item()
        
    validation_loss /= len(validation_dataset)

    print(f'Validation Loss: {validation_loss}')

    # Save the logs in a json file
    logs = {
        'model_path': model_path,
        'validation_loss': validation_loss
    }

    with open(f'{output_folder_path}/evaluation_info.json', 'w') as file:
        json.dump(logs, file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--val_data_path', type=str, default='data/val_dataset.txt')
    argparser.add_argument('--model_path', type=str, required=True)
    argparser.add_argument('--output_directory_path', type=str, default='outputs')
    args = argparser.parse_args()
    
    val_dataset = prepare_val_data(args.val_data_path)

    evaluate(validation_dataset=val_dataset, 
             model_path=args.model_path,
             output_directory_path=args.output_directory_path)