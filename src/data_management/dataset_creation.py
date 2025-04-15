import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformers import GPT2Tokenizer
import time
from sklearn.model_selection import train_test_split

class CrosswordDataset(Dataset):
    """
    A PyTorch Dataset for crossword clues and answers.
    Each sample consists of a formatted string containing the task, clue, and answer.
    The dataset is tokenized using a specified tokenizer and padded to a maximum length.

    Args:
        clues (list): List of crossword clues.
        answers (list): List of corresponding answers.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding the samples.
        max_length (int): Maximum length for tokenization and padding.
    """
    def __init__(self, clues, answers, tokenizer, max_length=50):
        print("[Dataset] Initializing crossword dataset...")
        self.samples = [f"Task: Solve the crossword clue:\nCrossword clue: {clue}\nAnswer: {answer}" for clue, answer in zip(clues, answers)]
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Dataset] Created {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.samples[idx],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()
        return input_ids, attention_mask, labels

def load_data(file="data/ho.csv"):
    """
    Load the crossword dataset from a CSV file.

    args:
        file (str): Path to the CSV file containing the crossword data.
    Returns:
        pd.DataFrame: DataFrame containing the crossword clues and answers.
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    print("[Data] Loading crossword dataset from CSV...")
    try:
        ho_df = pd.read_csv(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File {file} not found.")
    
    ho_df = ho_df.drop(columns=["rowid", "definition", "clue_number", "puzzle_date", "puzzle_name", "source_url"])
    print(f"[Data] Loaded {len(ho_df)} rows.")
    return ho_df

def split_and_save_data(file="data/ho.csv", train_file="data/train.csv", test_file="data/test.csv", test_size=0.5):
    """
    Split the dataset into training and testing datasets and save them to CSV files.

    Args:
        file (str): Path to the original dataset CSV file.
        train_file (str): Path to save the training dataset.
        test_file (str): Path to save the testing dataset.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    print("[Data] Splitting dataset into training and testing sets...")
    ho_df = load_data(file)
    train_df, test_df = train_test_split(ho_df, test_size=test_size, random_state=42)

    print(f"[Data] Saving training dataset to {train_file}...")
    train_df.to_csv(train_file, index=False)
    print(f"[Data] Saving testing dataset to {test_file}...")
    test_df.to_csv(test_file, index=False)
    print("[Data] Dataset split and saved successfully.")