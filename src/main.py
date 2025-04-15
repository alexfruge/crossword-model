import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import re

from data_management.dataset_creation import CrosswordDataset, split_and_save_data
from model.load import load_model, load_finetuned_model
from model.train import train

def generate_answer(
    model: torch.nn.Module, 
    tokenizer: GPT2Tokenizer, 
    clue: str, 
    max_new_tokens: int = 20, 
    device: str = 'cpu'
) -> str:
    """
    Generates an answer to a crossword clue using a language model.
    Args:
        model (torch.nn.Module): The pre-trained language model used for generating text.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        clue (str): The crossword clue to solve, which may include an expected answer length (e.g., "(4)").
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 20.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
    Returns:
        str: The generated answer in uppercase. If the clue specifies an expected length, the answer is truncated
             to that length and non-alphabetic characters are removed.
    """
    
    # Parse clue for expected answer length (e.g. "(4)")
    match = re.search(r'\((\d+)\)', clue)
    expected_length = int(match.group(1)) if match else None

    # Construct prompt
    prompt = f"Task: Solve the crossword clue:\nCrossword clue: {clue}\nAnswer:"
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

    # if expected_length:
    #     # Remove non-letters and truncate to expected length
    #     answer_clean = re.sub(r"[^A-Za-z]", "", answer_raw)
    #     return answer_clean[:expected_length].upper()

    return answer_raw.upper()


def generate_answers_from_csv(
    model: torch.nn.Module, 
    tokenizer: GPT2Tokenizer, 
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
    
    print(f"[Batch] Loading {n} clues from {csv_path}...\n")
    df = pd.read_csv(csv_path)
    clues = df['clue'].tolist()[:n]
    correct_answers = df['answer'].tolist()[:n]

    for i, (clue, correct) in enumerate(zip(clues, correct_answers), 1):
        generated = generate_answer(model, tokenizer, clue, device=device)
        print(f"{i}. Clue: {clue}")
        print(f"Correct Answer:   {correct}")
        print(f"Generated Answer: {generated}\n")


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

    print("[Main] Starting crossword fine-tuning script...")

    # Load data
    split_and_save_data(file="data/ho.csv", train_file="data/train.csv", test_file="data/test.csv", test_size=0.5)
    train_df = pd.read_csv("data/train.csv")
    clues = train_df['clue'].tolist()
    answers = train_df['answer'].tolist()

    # Load tokenizer and model
    print("[Main] Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("[Main] Tokenizer loaded.")

    model = load_model(model_name=model_name)

    # Prepare dataset and dataloader
    dataset = CrosswordDataset(clues, answers, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Start training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    train(model, dataloader, tokenizer, device)

    # print("[Main] Training complete.")

    save_path = f"trained_models/model-{model_name}.pt"
    print(f"[Main] Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("[Main] Model weights saved successfully.")

    # Initialize
    # model = load_finetuned_model(model_name=model_name, weights_path=f"trained_models/model-{model_name}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate answers
    generate_answers_from_csv(model, tokenizer, csv_path="data/test.csv", n=100, device=device)

if __name__ == "__main__":
    models = ["gpt2-medium", "gpt2-large", "gpt2-xl"]
    for model_name in models:
        print(f"[Main] Running with model: {model_name}")
        main(model_name=model_name)
        print(f"[Main] Finished with model: {model_name}\n")


