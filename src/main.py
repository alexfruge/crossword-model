import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AutoTokenizer

from data_management.dataset_creation import split_and_save_data
from model.load import load_model, load_finetuned_model
from utils import log_statement

# Import the enhancement modules
from enhancements.contextual_embeddings import create_enhanced_model as create_embedding_model
from enhancements.prompt_engineering import create_enhanced_dataloader
from enhancements.length_aware_model import create_length_aware_model, Constrained_Beam_Generator


def generate_answer_enhanced(
    model, 
    tokenizer, 
    clue: str,
    expected_length: int = None, 
    model_name: str = None,
    enhancement: str = None,
    device: str = 'cpu'
) -> str:
    """
    Enhanced answer generation that uses the appropriate method based on the enhancement type.
    """
    # Use the appropriate generation method based on enhancement
    if enhancement == "knowledge":
        # Knowledge augmentation has its own generation method
        if hasattr(model, 'generate_answer'):
            return model.generate_answer(tokenizer, clue, expected_length, device)
    
    elif enhancement == "length_aware":
        # Length-aware model has its own generation method
        if hasattr(model, 'generate_answer'):
            return model.generate_answer(tokenizer, clue, expected_length, device)
        
        # Alternatively, use the constrained beam generator
        generator = Constrained_Beam_Generator(model, tokenizer, device)
        return generator.generate(clue, expected_length)
        
    # Default generation method for other enhancements
    prompt = f"Task: Solve the crossword clue:\nCrossword clue: {clue} ({expected_length})\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'generate'):
            # Use the model's generate method if available
            output = model.generate(
                input_ids,
                max_new_tokens=20,
            )
        else:
            # Manual token generation if generate method not available
            generated = input_ids.clone()
            for _ in range(20):  # max_new_tokens
                logits = model(generated)[0, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
            output = generated
    
    # Decode the output
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_raw = decoded.split("Answer:")[-1].strip()
    
    # Clean up and format the answer
    answer = ''.join(char for char in answer_raw if char.isalpha()).upper()
    if expected_length and len(answer) > expected_length:
        answer = answer[:expected_length]
        
    return answer


def generate_answers_from_csv_enhanced(
    model, 
    tokenizer, 
    model_name: str,
    enhancement: str = None,
    csv_path: str = "data/test.csv", 
    n: int = 10, 
    device: str = 'cpu'
) -> None:
    """Enhanced version of generate_answers_from_csv that supports enhancements."""
    
    log_statement(model_name, f"[Batch] Loading {n} clues from {csv_path}...")
    df = pd.read_csv(csv_path)
    clues = df['clue'].tolist()[:n]
    correct_answers = df['answer'].tolist()[:n]
    ans_lengths = df['ans_length'].tolist()[:n]

    for i, (clue, correct, expected_len) in enumerate(zip(clues, correct_answers, ans_lengths), 1):
        generated = generate_answer_enhanced(
            model, tokenizer, clue, expected_len, 
            model_name=model_name, 
            enhancement=enhancement,
            device=device
        )
        log_statement(model_name, f"{i}. Clue: {clue} ({expected_len})")
        log_statement(model_name, f"Correct Answer:   {correct}")
        log_statement(model_name, f"Generated Answer: {generated}\n")


def main_enhanced(model_name: str = "gpt2-medium", enhancement: str = None):
    """
    Enhanced main function supporting various enhancements for crossword solving.
    
    Args:
        model_name: Name of the base model to use
        enhancement: Type of enhancement to apply ("embeddings", "prompt", "knowledge", 
                    "adversarial", "length_aware", None)
    """
    log_statement(model_name, f"[Main] Starting crossword fine-tuning with {enhancement if enhancement else 'no'} enhancement...")

    # Load data
    split_and_save_data(
        file="data/nytcrosswords_full.csv", 
        train_file="data/train.csv", 
        test_file="data/test.csv", 
        test_size=0.5, 
        max_elements=12_800, 
        model_name=model_name
    )
    
    # Load tokenizer
    log_statement(model_name, "[Main] Loading tokenizer...")
    if "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name.split("/")[-1])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    log_statement(model_name, "[Main] Tokenizer loaded.")

    # Create the appropriate model based on enhancement
    if enhancement == "embeddings":
        model = create_embedding_model(model_name)
        log_statement(model_name, "[Main] Created embedding-enhanced model.")
    elif enhancement == "length_aware":
        model = create_length_aware_model(model_name)
        log_statement(model_name, "[Main] Created length-aware model.")
    else:
        model = load_model(model_name)
        log_statement(model_name, "[Main] Loaded standard model.")

    # Prepare dataset and dataloader based on enhancement
    if enhancement == "prompt":
        dataloader = create_enhanced_dataloader(
            csv_path="data/train.csv",
            tokenizer=tokenizer,
            model_name=model_name,
            batch_size=32,
            prompt_strategy="detailed"
        )
        log_statement(model_name, "[Main] Created dataloader with enhanced prompts.")
    else:
        # Standard dataset creation
        train_df = pd.read_csv("data/train.csv")
        clues = train_df['clue'].tolist()
        answers = train_df['answer'].tolist()
        ans_lengths = train_df['ans_length'].tolist()
        
        from data_management.dataset_creation import CrosswordDataset
        dataset = CrosswordDataset(clues, answers, ans_lengths, tokenizer, model_name)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        log_statement(model_name, "[Main] Created standard dataloader.")

    # Start training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_statement(model_name, f"[Main] Using device: {device}")

    from model.train import train
    loss = train(model, model_name, dataloader, device, num_epochs=5)

    log_statement(model_name, "[Main] Training complete.")

    # Save model
    enhancement_tag = f"-{enhancement}" if enhancement else ""
    if enhancement:
        save_path = f"trained_models/enhanced/model-{model_name.split('/')[-1]}{enhancement_tag}.pt"
    else:
        save_path = f"trained_models/raw/model-{model_name.split('/')[-1]}{enhancement_tag}.pt"
    log_statement(model_name, f"[Main] Saving model weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    log_statement(model_name, "[Main] Model weights saved successfully.")

    # Generate answers with the trained model
    generate_answers_from_csv_enhanced(
        model, tokenizer, model_name, 
        enhancement=enhancement,
        csv_path="data/test.csv", 
        n=100, 
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a crossword solver with enhancements')
    parser.add_argument('--model', type=str, default="EleutherAI/pythia-70m", 
                        help='Base model to use')
    parser.add_argument('--enhancement', type=str, choices=[
                            'embeddings', 'prompt', 'length_aware', 'none'
                        ], default='none',
                        help='Enhancement type to apply')
    
    args = parser.parse_args()
    enhancement = None if args.enhancement == 'none' else args.enhancement
    
    main_enhanced(model_name=args.model, enhancement=enhancement)