import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import re

from data_management.dataset_creation import CrosswordDataset, split_and_save_data
from model.load import load_model, load_finetuned_model
from model.train import train

def generate_answer(model, tokenizer, clue, max_new_tokens=20, device='cpu'):
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

    if expected_length:
        # Remove non-letters and truncate to expected length
        answer_clean = re.sub(r"[^A-Za-z]", "", answer_raw)
        return answer_clean[:expected_length].upper()

    return answer_raw.upper()


def generate_answers_from_csv(model, tokenizer, csv_path="data/test.csv", n=10, device='cpu'):
    """
    Generates answers for a set of clues from a CSV file using a given model and tokenizer.
    Args:
        model: The model used to generate answers.
        tokenizer: The tokenizer used to preprocess the clues for the model.
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



if __name__ == "__main__":
    print("[Main] Starting crossword fine-tuning script...")

    # Load data
    split_and_save_data(file="data/ho.csv", train_file="data/train.csv", test_file="data/test.csv", test_size=0.5)
    train_df = pd.read_csv("data/train.csv")
    clues = train_df['clue'].tolist()
    answers = train_df['answer'].tolist()

    # Load tokenizer and model
    model_name = "gpt2-medium"
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

    print("[Main] Training complete.")

    save_path = f"model-{model_name}.pt"
    print(f"[Main] Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("[Main] Model weights saved successfully.")

    # Initialize
    model = load_finetuned_model(model_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate answers
    generate_answers_from_csv(model, tokenizer, csv_path="data/test.csv", n=100, device=device)


