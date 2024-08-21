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

def prepare_data(data_dir_path):
    # read the train and val dataset from the txt files
    with open(f'{data_dir_path}/train_dataset.txt', 'r') as file:
        train_name_list = file.read().split('\n')

    with open(f'{data_dir_path}/val_dataset.txt', 'r') as file:
        val_name_list = file.read().split('\n')
    
    # Create train dataset for predicting the next character
    train_dataset = []
    for name in train_name_list:
        name_char_list = list(['<S>']) + list(name) + list(['<E>'])
        for i in range(1, len(name_char_list)):
            input = [char for char in name_char_list[:i]]
            output = [name_char_list[i]]
            train_dataset.append((input, output))
    
    # Create validation dataset for predicting the next character
    val_dataset = []
    for name in val_name_list:
        name_char_list = list(['<S>']) + list(name) + list(['<E>'])
        for i in range(1, len(name_char_list)):
            input = [char for char in name_char_list[:i]]
            output = [name_char_list[i]]
            val_dataset.append((input, output))

    return train_dataset, val_dataset


def train(train_dataset, validation_dataset, 
          hidden_size, epochs, lr, 
          output_directory_path):
    '''
    Train the model with the given dataset and hyperparameters
    '''

    # Create a new folder in the output directory with current datetime to save the model and logs
    output_folder_path = f'{output_directory_path}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(output_folder_path)

    # Initialize the model
    name_generator = RNN(input_size=VOCAB_SIZE, hidden_size=hidden_size, output_size=VOCAB_SIZE)
    numb_of_learned_params = sum(p.numel() for p in name_generator.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {numb_of_learned_params}')

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(name_generator.parameters(), lr=lr)

    train_loss_logs = []
    validation_loss_logs = []
    best_validation_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0
        for input_seq, target in tqdm(train_dataset):
            hidden = name_generator.initHidden()

            for char in input_seq:
                input_tensor = char_to_onehot(char)
                output, hidden = name_generator(input_tensor, hidden)
            
            output_tensor = char_to_onehot(target[0])

            optimizer.zero_grad()
            loss = loss_fn(output, output_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_dataset)
        train_loss_logs.append(train_loss)

        name_generator.eval()
        validation_loss = 0
        with torch.no_grad():
            for input_seq, target in validation_dataset:
                hidden = name_generator.initHidden()

                for char in input_seq:
                    input_tensor = char_to_onehot(char)
                    output, hidden = name_generator(input_tensor, hidden)
                
                output_tensor = char_to_onehot(target[0])

                loss = loss_fn(output, output_tensor)
                validation_loss += loss.item()
            
        validation_loss /= len(validation_dataset)
        validation_loss_logs.append(validation_loss)

        print(f'Epoch {epoch} - Train Loss: {train_loss} | Validation Loss: {validation_loss}')

        # Save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(name_generator.state_dict(), f'{output_folder_path}/best.pth')

    # Save the last model
    torch.save(name_generator.state_dict(), f'{output_folder_path}/last.pth')

    # Save the loss plots
    plt.figure()
    plt.plot(train_loss_logs, label='Train Loss')
    plt.plot(validation_loss_logs, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_folder_path}/loss_plot.png')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_directory_path', type=str, default='data')
    argparser.add_argument('--output_directory_path', type=str, default='outputs')
    argparser.add_argument('--hidden_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.0001)
    args = argparser.parse_args()
    
    train_dataset, val_dataset = prepare_data(args.data_directory_path)

    train(train_dataset=train_dataset, 
          validation_dataset=val_dataset,
          hidden_size=args.hidden_size,
          epochs=args.epochs, lr=args.lr,
          output_directory_path=args.output_directory_path)