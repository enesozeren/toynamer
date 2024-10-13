import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import json
import logging

from src.rnn_model import RNN
from src.variables import VOCAB_SIZE
from src.data_class import NameData

def train(train_data_path, val_data_path, 
          hidden_size, epochs, lr, weight_decay, batch_size,
          output_directory_path):
    '''
    Train the model with the given dataset and hyperparameters
    '''

    # Create a new folder in the output directory with current datetime to save the model and logs
    output_folder_path = f'{output_directory_path}/train_run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(output_folder_path)

    logging.basicConfig(filename=os.path.join(output_folder_path, 'logfile.txt'),
                        level=logging.INFO)
    
    # Save hyperparameters in a json file
    hyperparameters = {
        'hidden_size': hidden_size,
        'epochs': epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size
    }
    logging.info(f"Hyperparameters: {hyperparameters}")

    # Train and Validation datasets
    train_dataset = NameData(name_data_path=train_data_path)
    val_dataset = NameData(name_data_path=val_data_path)

    # Log Dataset Sizes
    logging.info(f'Train Dataset Size: {train_dataset.__len__()}')
    logging.info(f'Val Dataset Size: {val_dataset.__len__()}')
    logging.info(f"Vocab Size (# of letter tokens): {VOCAB_SIZE}")
    
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name_generator = RNN(input_size=VOCAB_SIZE, hidden_size=hidden_size, output_size=VOCAB_SIZE).to(device)
    numb_of_learned_params = sum(p.numel() for p in name_generator.parameters() if p.requires_grad)
    logging.info(f"Model Learnable Parameter Size: {numb_of_learned_params}")

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(name_generator.parameters(), lr=lr, weight_decay=weight_decay)

    # Data laoders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=False, drop_last=True)
    
    # Log variables
    train_loss_logs = []
    validation_loss_logs = []
    best_validation_loss = float('inf')

    # Create a file to log the train and validation losses for each epoch
    log_file_path = f'{output_folder_path}/losses.txt'
    with open(log_file_path, 'w') as log_file:
        log_file.write('Epoch\tTrain Loss\tValidation Loss\n')  # Header for the file

    # CHECK HERE AND FIX THE PROBLEM
    for epoch in range(epochs):
        train_loss = 0
        for input_seqs, targets in tqdm(train_loader):

            # Move data to device
            input_seqs, targets = input_seqs.to(device), targets.to(device)
            print("Input seq:", input_seqs.shape)
            print("Targets:", targets.shape)
            # Get the batch size dynamically
            batch_size_dynamic = input_seqs.size(0)
            hidden = name_generator.initHidden(batch_size=batch_size_dynamic).to(device)
            optimizer.zero_grad()

            outputs = []
            for t in range(input_seqs.size(1)):
                output, hidden = name_generator(input_seqs[:, t], hidden)
                outputs.append(output)
            
            print("outputs:", len(outputs))
            outputs = torch.stack(outputs, dim=1)  # Stack outputs for each time step
            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
            targets = targets.view(-1)  # (batch_size * seq_len)
            print("outputs and targets at the end:", outputs.shape, targets.shape)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_loss_logs.append(train_loss)

        name_generator.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_seqs, targets = batch
                # Move data to device
                input_seqs, targets = input_seqs.to(device), targets.to(device)
                
                # Get the batch size dynamically
                batch_size_dynamic = input_seqs.size(0)
                hidden = name_generator.initHidden(batch_size=batch_size_dynamic).to(device)
                
                outputs = []
                for t in range(input_seqs.size(1)):
                    output, hidden = name_generator(input_seqs[:, t], hidden)
                    outputs.append(output)

                outputs = torch.stack(outputs, dim=1)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                loss = loss_fn(outputs, targets)
                validation_loss += loss.item()
            
        validation_loss /= len(val_loader)
        validation_loss_logs.append(validation_loss)

        # Log the current epoch losses to the txt file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{epoch}\t{train_loss}\t{validation_loss}\n')
        
        print(f'Epoch {epoch} / {epochs-1} - Train Loss: {train_loss} | Validation Loss: {validation_loss}')

        # Save the best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(name_generator, f'{output_folder_path}/best.pth')
        
        # Save the loss plots in each epoch
        plt.figure()
        plt.plot(train_loss_logs, 'o-', label='Train Loss')
        plt.plot(validation_loss_logs, 'o-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{output_folder_path}/loss_plot.png')
        plt.close()

    # Save the last model
    torch.save(name_generator, f'{output_folder_path}/last.pth')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_directory_path', type=str, default='data')
    argparser.add_argument('--output_directory_path', type=str, default='outputs')
    argparser.add_argument('--hidden_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.0001)
    argparser.add_argument('--weight_decay', type=float, default=0.0)
    argparser.add_argument('--batch_size', type=int, default=1)
    args = argparser.parse_args()

    train_data_path = os.path.join(args.data_directory_path, 'train_dataset.txt')
    val_data_path = os.path.join(args.data_directory_path, 'val_dataset.txt')

    train(train_data_path=train_data_path, 
          val_data_path=val_data_path,
          hidden_size=args.hidden_size,
          epochs=args.epochs, lr=args.lr,
          weight_decay=args.weight_decay,
          batch_size=args.batch_size,
          output_directory_path=args.output_directory_path)