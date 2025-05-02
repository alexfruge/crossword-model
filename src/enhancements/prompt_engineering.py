from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class EnhancedCrosswordDataset(Dataset):
    """
    Enhanced dataset class for crossword clues with improved prompting techniques.
    """
    
    def __init__(self, clues: List[str], answers: List[str], 
                 ans_lengths: List[int], tokenizer: PreTrainedTokenizer, 
                 model_name: str, prompt_strategy: str = "cot"):
        """
        Initialize the dataset with clues, answers, and lengths.
        
        Args:
            clues: List of crossword clues
            answers: List of answers corresponding to the clues
            ans_lengths: List of answer lengths
            tokenizer: Tokenizer for the language model
            model_name: Name of the model (for logging)
            prompt_strategy: Strategy for prompt formatting ("basic", "detailed", "cot")
        """
        self.clues = clues
        self.answers = answers
        self.ans_lengths = ans_lengths
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.prompt_strategy = prompt_strategy
        
        # Make sure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.clues)
    
    def _format_prompt(self, clue: str, ans_length: int) -> str:
        """Format the prompt according to the selected strategy."""
        
        if self.prompt_strategy == "basic":
            # Basic format - simply provide the clue and length
            return f"Crossword clue: {clue} ({ans_length})\nAnswer:"
            
        elif self.prompt_strategy == "detailed":
            # More detailed format with explicit instructions
            return (
                f"Task: Solve this crossword puzzle clue.\n"
                f"Clue: \"{clue}\"\n"
                f"Number of letters: {ans_length}\n"
                f"The answer should be {ans_length} letters long with no spaces.\n"
                f"Answer:"
            )
            
        elif self.prompt_strategy == "cot":
            # Chain-of-thought prompting to encourage reasoning
            return (
                f"Task: Solve the following crossword clue step by step.\n"
                f"Clue: \"{clue}\"\n"
                f"Length: {ans_length} letters\n"
                f"Let's think about what this clue might be referring to, considering wordplay, "
                f"possible definitions, and similar terms. Then I'll provide a {ans_length}-letter answer.\n"
                f"Reasoning:\n"
                f"Answer:"
            )
            
        else:
            # Default to basic if an unknown strategy is specified
            return f"Crossword clue: {clue} ({ans_length})\nAnswer:"
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the tokenized prompt and target for a specific index.
        
        Returns:
            Tuple containing input_ids, attention_mask, and labels tensors
        """
        clue = self.clues[idx]
        answer = self.answers[idx]
        ans_length = self.ans_lengths[idx]
        
        # Format the prompt based on our strategy
        prompt = self._format_prompt(clue, ans_length)
        
        # Format the full sequence (prompt + answer)
        full_sequence = f"{prompt} {answer}"
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            full_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256  # Adjust as needed for your model
        )
        
        # Extract tensors
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For training, labels are the same as input_ids
        labels = input_ids.clone()
        
        return input_ids, attention_mask, labels

def create_enhanced_dataloader(
    csv_path: str, 
    tokenizer: PreTrainedTokenizer, 
    model_name: str,
    batch_size: int = 32,
    prompt_strategy: str = "detailed",
    max_samples: int = None
) -> DataLoader:
    """
    Creates an enhanced dataloader with improved prompting strategies.
    
    Args:
        csv_path: Path to the CSV file with clues and answers
        tokenizer: Tokenizer for the model
        model_name: Name of the model (for logging)
        batch_size: Batch size for the dataloader
        prompt_strategy: Strategy for prompt formatting
        max_samples: Maximum number of samples to use (None for all)
        
    Returns:
        DataLoader for training or evaluation
    """
    # Load the data
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)
        
    clues = df['clue'].tolist()
    answers = df['answer'].tolist()
    ans_lengths = df['ans_length'].tolist()
    
    # Create the dataset
    dataset = EnhancedCrosswordDataset(
        clues, answers, ans_lengths, 
        tokenizer, model_name, prompt_strategy
    )
    
    # Create and return the dataloader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)