import torch
from src.variables import turkish_chars

class CharLevTokenizer:
    '''
    Character level tokinizer with start and end tokens
    '''
    def __init__(self):
        self.vocab = turkish_chars
        self.char_to_int = {char: index for index, char in enumerate(self.vocab)}
        self.char_to_int['<S>'] = len(self.vocab)
        self.char_to_int['<E>'] = len(self.vocab) + 1
        self.int_to_char = {index: char for char, index in self.char_to_int.items()}
    
    def encode(self, text):
        # get the char token ids
        tokens = torch.tensor([self.char_to_int[char] for char in text])
        # add start and end of sequence tokens
        tokens = torch.cat((tokens, torch.tensor([self.char_to_int['<S>']])))
        tokens = torch.cat((torch.tensor([self.char_to_int['<E>']]), tokens))
        
        return tokens

    def decode(self, tokens):
        # Decode tokens to text, ignore start and end tokens
        text = ''.join([self.int_to_char[token.item()] for token in tokens 
                        if self.int_to_char[token.item()] not in ['<S>', '<E>']])
        return text

# tokenizer = CharLevTokenizer()
# print(tokenizer.encode('Eno'))
# print(tokenizer.decode(tokenizer.encode('Eno')))
