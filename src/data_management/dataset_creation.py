import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from utils import log_statement

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
    def __init__(self, clues, answers, ans_lengths, tokenizer, model_name, max_length=50):
        log_statement(model_name, "[Dataset] Initializing crossword dataset...")
        self.samples = [f"Task: Solve the crossword clue:\nCrossword clue: {clue} ({ans_length})\nAnswer: {answer}" for clue, answer, ans_length in zip(clues, answers, ans_lengths)]
        self.tokenizer = tokenizer
        self.max_length = max_length
        log_statement(model_name, f"[Dataset] Created {len(self.samples)} samples.")

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

def load_data(file="data/ho.csv", max_elements = 1_000_000, model_name=None):
    """
    Load the crossword dataset from a CSV file.

    args:
        file (str): Path to the CSV file containing the crossword data.
        max_elements (int): Maximum number of elements to load from the dataset.
        model_name (str): Name of the model for logging purposes.
    Returns:
        pd.DataFrame: DataFrame containing the crossword clues and answers.
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    log_statement(model_name, "[Data] Loading crossword dataset from CSV...")
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File {file} not found.")
    
    df = df[["clue", "answer"]].dropna()
    log_statement(model_name, f"[Data] Loaded {len(df)} rows.")
    
    if len(df) > max_elements:
        log_statement(model_name, f"[Data] Truncating dataset to {max_elements} rows.")
        df = df.head(max_elements)
    
    df = add_answer_lengths(df, model_name=model_name)

    df.to_csv(file, index=False)

    return df

def split_and_save_data(file="data/ho.csv", train_file="data/train.csv", test_file="data/test.csv", test_size=0.8, max_elements = 10_000, model_name=None):
    """
    Split the dataset into training and testing datasets and save them to CSV files.

    Args:
        file (str): Path to the original dataset CSV file.
        train_file (str): Path to save the training dataset.
        test_file (str): Path to save the testing dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        max_elements (int): Maximum number of elements to load from the dataset.
        model_name (str): Name of the model for logging purposes.
    """
    log_statement(model_name, "[Data] Splitting dataset into training and testing sets...")
    df = load_data(file, max_elements=max_elements, model_name=model_name)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    log_statement(model_name, f"[Data] Saving training dataset to {train_file}...")
    train_df.to_csv(train_file, index=False)
    log_statement(model_name, f"[Data] Saving testing dataset to {test_file}...")
    test_df.to_csv(test_file, index=False)
    log_statement(model_name, "[Data] Dataset split and saved successfully.")



def add_answer_lengths(df, model_name=None):
    """
    Add a column 'ans_length' to the dataframe containing the length of each answer.
    The length is dash-delimited for multi-word answers.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'answer'.
        model_name (str): Name of the model for logging purposes.
    Returns:
        pd.DataFrame: DataFrame with the new 'ans_length' column.
    """

    if 'ans_length' not in df.columns:
        log_statement(model_name, "[Data] Adding 'ans_length' column to the dataframe...")
        df["ans_length"] = pd.Series(dtype="string")

    # Remove answer length information from clues
    df['clue'] = df['clue'].str.replace(r"\(\d+(,\d+)*\)", "", regex=True).str.strip()

    def calculate_length(answer):
        words = answer.split()
        if len(words) > 1:
            return ",".join(str(len(word)) for word in words)
        return str(len(answer))
    
    df['ans_length'] = df['answer'].apply(calculate_length)
    return df
