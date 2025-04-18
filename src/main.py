import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoTokenizer

from data_management.dataset_creation import split_and_save_data, CrosswordDataset
from model.load import load_model, load_finetuned_model
from model.train import train
from utils import log_statement

def generate_answer(
    model: torch.nn.Module, 
    tokenizer: GPT2Tokenizer, 
    clue: str,
    expected_length: int = None, 
    max_new_tokens: int = 20, 
    device: str = 'cpu'
) -> str:
    """
    Generates an answer to a crossword clue using a language model.
    Args:
        model (torch.nn.Module): The pre-trained language model used for generating text.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        clue (str): The crossword clue to solve, which may include an expected answer length (e.g., "(4)").
        expected_length (int, optional): The expected length of the crossword answer. 
                        If provided, the generated answer will be truncated to this length, and non-alphabetic characters 
                        will be removed. Defaults to None.
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 20.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
    Returns:
        str: The generated answer in uppercase. If the clue specifies an expected length, the answer is truncated
             to that length and non-alphabetic characters are removed.
    """

    # Construct prompt
    prompt = f"Task: Solve the crossword clue:\nCrossword clue: {clue} ({expected_length})\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)[0, -1, :]  # get last token's logits
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)

            # Optional: break early on EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode full sequence and extract answer
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer_raw = decoded.split("Answer:")[-1].strip()

    return answer_raw.upper()


def generate_answers_from_csv(
    model: torch.nn.Module, 
    tokenizer: GPT2Tokenizer, 
    model_name: str,
    csv_path: str = "data/test.csv", 
    n: int = 10, 
    device: str = 'cpu'
) -> None:
    """
    Generates answers for a set of clues from a CSV file using a given model and tokenizer.
    Args:
        model (torch.nn.Module): The model used to generate answers.
        tokenizer (GPT2Tokenizer): The tokenizer used to preprocess the clues for the model.
        csv_path (str, optional): Path to the CSV file containing clues and correct answers. 
                                  Defaults to "data/test.csv".
        n (int, optional): Number of clues to process from the CSV file. Defaults to 10.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
    Prints:
        For each clue, prints the clue, the correct answer, and the generated answer.
    """
    
    log_statement(model_name, f"[Batch] Loading {n} clues from {csv_path}...\n")
    df = pd.read_csv(csv_path)
    clues = df['clue'].tolist()[:n]
    correct_answers = df['answer'].tolist()[:n]
    ans_lengths = df['ans_length'].tolist()[:n]

    for i, (clue, correct, expected_len) in enumerate(zip(clues, correct_answers, ans_lengths), 1):
        generated = generate_answer(model, tokenizer, clue, device=device)
        log_statement(model_name, f"{i}. Clue: {clue} ({expected_len})")
        log_statement(model_name, f"Correct Answer:   {correct}")
        log_statement(model_name, f"Generated Answer: {generated}\n")


def main(model_name: str = "gpt2-medium"):
    """
    Main function to fine-tune a crossword-solving model and generate answers.
    This script performs the following steps:
    1. Loads and splits the dataset into training and testing sets.
    2. Loads a pre-trained GPT-2 tokenizer and model.
    3. Prepares the dataset and dataloader for training.
    4. Trains the model on the crossword dataset.
    5. Saves the fine-tuned model weights.
    6. Loads the fine-tuned model and generates answers for a test dataset.
    Steps:
    - Data is loaded from a CSV file and split into training and testing sets.
    - The GPT-2 tokenizer is initialized and configured.
    - A custom dataset and dataloader are created for training.
    - The model is trained using the training data.
    - The trained model weights are saved to a file.
    - The fine-tuned model is loaded and used to generate answers for clues in the test dataset.
    Dependencies:
    - PyTorch for model training and saving.
    - Transformers library for GPT-2 tokenizer and model.
    - Custom modules for data processing, model loading, and answer generation.
    Note:
    - Ensure that the required CSV files and directories exist before running the script.
    - The script uses GPU if available; otherwise, it defaults to CPU.
    """

    log_statement(model_name, "[Main] Starting crossword fine-tuning script...")

    # Load data
    split_and_save_data(file="data/nytcrosswords_full.csv", train_file="data/train.csv", test_file="data/test.csv", test_size=0.5, max_elements=12_800, model_name=model_name)
    train_df = pd.read_csv("data/train.csv")
    clues = train_df['clue'].tolist()
    answers = train_df['answer'].tolist()
    ans_lengths = train_df['ans_length'].tolist()

    # Load tokenizer and model
    log_statement(model_name, "[Main] Loading tokenizer...")
    if "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name.split("/")[-1])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    log_statement(model_name, "[Main] Tokenizer loaded.")

    model = load_model(model_name=model_name)

    # Prepare dataset and dataloader
    dataset = CrosswordDataset(clues, answers, ans_lengths, tokenizer, model_name=model_name)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Start training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_statement(model_name, f"[Main] Using device: {device}")
    loss = train(model, model_name, dataloader, device, num_epochs=5)

    log_statement(model_name, "[Main] Training complete.")


    save_path = f"trained_models/model-{model_name.split("/")[-1]}.pt"
    log_statement(model_name, f"[Main] Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    log_statement(model_name, "[Main] Model weights saved successfully.")

    # Initialize
    # model = load_finetuned_model(model_name=model_name, weights_path=f"trained_models/model-{model_name}.pt")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Generate answers
    generate_answers_from_csv(model, tokenizer, model_name, csv_path="data/test.csv", n=100, device=device)

if __name__ == "__main__":
    models = [
        "EleutherAI/pythia-14m",
        "EleutherAI/pythia-31m",
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1.3b",
        # "EleutherAI/pythia-2.8b",
    ]
    for model_name in models:
        log_statement(model_name, f"[Main] Running with model: {model_name}")
        main(model_name=model_name)
        log_statement(model_name, f"[Main] Finished with model: {model_name}\n")


