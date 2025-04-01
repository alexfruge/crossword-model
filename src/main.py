import pandas as pd
import transformer_lens

def load_model(model_name):
    """
    Load a specified model from transformer_lens and return it.

    Args:
        model_name (str): The name of the model to load (e.g., "gpt2-small").

    Returns:
        HookedTransformer: The loaded model.
    """
    model = transformer_lens.HookedTransformer.from_pretrained(model_name)
    return model




def load_ho_csv():
    """
    Load the HO.csv file and return a DataFrame.
    """
    ho_df = pd.read_csv('data/ho.csv')
    ho_df = ho_df.drop(columns=["rowid", "definition", "clue_number", "puzzle_date", "puzzle_name", "source_url"])
    return ho_df


def generate_crossword_answer(clue, model):
    """
    Generate an answer to the given crossword clue using the provided model.

    Args:
        clue (str): The crossword clue.
        model: The pre-trained GPT-2 model.

    Returns:
        str: The generated answer to the crossword clue.
    """
    # Prepare the input prompt for the model
    prompt = f"Crossword clue: {clue}\nAnswer:"
    
    # Generate the answer using the model
    generated_text = model.generate(prompt, max_new_tokens=10, do_sample=True, temperature=0.7)
    
    # Extract and return the generated answer
    return generated_text


if __name__ == "__main__":
    # Load the HO.csv file
    ho_df = load_ho_csv()
    
    # Display the first few rows of the DataFrame
    print(ho_df.head())
    
    # Display the shape of the DataFrame
    print(f"Shape of the DataFrame: {ho_df.shape}")
    
    # Display the columns of the DataFrame
    print(f"Columns in the DataFrame: {ho_df.columns.tolist()}")

    # Load the GPT-2 medium model
    model = load_model("gpt2-medium")

    # Example clue to generate an answer for
    example_clue = ho_df.iloc[0]['clue']
    example_true_answer = ho_df.iloc[0]['answer']

    # Loop over the first 10 entries in ho_df
    for i in range(min(10, len(ho_df))):
        print("-" * 50)
        example_clue = ho_df.iloc[i]['clue']
        example_true_answer = ho_df.iloc[i]['answer']
        
        # Generate an answer for the current clue
        generated_answer = generate_crossword_answer(example_clue, model)
        print(f"Clue {i + 1}: {example_clue}")
        print(f"True Answer {i + 1}: {example_true_answer}")
        print(f"Generated Answer {i + 1}: {generated_answer}")
        print("-" * 50)

