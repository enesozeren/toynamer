import locale
import argparse
import random

# Set the locale to Turkish
locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')

def extract_names(raw_data_path: str):
    names = []
    with open(raw_data_path, 'r') as f:
        for line in f:
            # remove everythin other than characters and commas
            line = ''.join([c for c in line if c.isalpha() or c == ','])
            line = line.strip()  # remove newline character
            elements = line.split(',')  # split line by comma
            name = elements[1].strip().strip("'")  # extract the second element and remove leading/trailing spaces and quotes
            name = name.replace('â', 'a').replace('î', 'i').replace('û', 'u').replace('Â', 'A') # replace old turkish characters with new ones
            names.append(name)
    return names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, required=True)
    parser.add_argument('--train_val_split', type=float, default=0.9)
    args = parser.parse_args()

    names = extract_names(args.raw_data_path)

    # Shuffle the names
    random.shuffle(names)
    # Split the names into train and validation sets
    split_idx = int(len(names) * args.train_val_split)
    train_names = names[:split_idx]
    val_names = names[split_idx:]
    
    with open('data/train_dataset.txt', 'w') as f:
        for name in train_names[:-1]:
            f.write(name + '\n')
        f.write(train_names[-1])

    with open('data/val_dataset.txt', 'w') as f:
        for name in val_names[:-1]:
            f.write(name + '\n')      
        f.write(val_names[-1])