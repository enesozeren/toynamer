import torch
from torch.utils.data import Dataset
from src.tokenizer import CharLevTokenizer

class NameData(Dataset):
    '''
    Custom Dataset for creating input and output tensors for names
    '''
    
    def __init__(self, name_data_path):
        # create tokenizer object
        tokenizer = CharLevTokenizer()

        # read the train and val dataset from the txt files
        with open(name_data_path, 'r') as file:
            name_list = file.read().split('\n')
        
        # tokenize all the names
        name_list_tokenized = [tokenizer.encode(name) for name in name_list]
        
        # create input and target tuples
        self.input_target_list = []
        for name_tokenized in name_list_tokenized:
            for i in range(1, name_tokenized.shape[0]):
                input = name_tokenized[:i]
                output = name_tokenized[i:i+1]
                self.input_target_list.append((input, output))

    def __len__(self):
        return len(self.input_target_list)

    def __getitem__(self, idx):
        input, target = self.input_target_list[idx]
        return input, target

# train_data = NameData(name_data_path='data/train_dataset.txt')
# print("Size:", train_data.__len__())
# print("Ex:", train_data.__getitem__(5))