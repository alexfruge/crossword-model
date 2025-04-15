import torch
from transformer_lens import HookedTransformer
from utils import log_statement

def load_model(model_name: str) -> HookedTransformer:
    """
    Loads a pre-trained model using the specified model name.
    Args:
        model_name (str): The name or identifier of the pre-trained model to load.
    Returns:
        HookedTransformer: An instance of the loaded pre-trained model.
    Raises:
        Any exceptions raised by HookedTransformer.from_pretrained if the model cannot be loaded.
    Example:
        model = load_model("gpt-3")
    """

    log_statement(model_name, f"[Model] Loading model: {model_name}...")
    try:
        model = HookedTransformer.from_pretrained(model_name)
    except Exception as e:
        log_statement(model_name, f"[Model] Error loading model: {e}")
        raise e
    log_statement(model_name, "[Model] Model loaded successfully.")
    return model


def load_finetuned_model(model_name: str = "gpt2-xl", weights_path: str = "model-gpt2-xl.pt") -> HookedTransformer:
    """
    Loads a fine-tuned transformer model.
    Args:
        model_name (str): The name of the base model to load. Defaults to "gpt2-xl".
        weights_path (str): The file path to the fine-tuned model weights. Defaults to "model-gpt2-xl.pt".
    Returns:
        HookedTransformer: The fine-tuned transformer model set to evaluation mode.
    Notes:
        - The function uses `HookedTransformer.from_pretrained` to load the base model.
        - The fine-tuned weights are loaded using `torch.load` with the map location set to 'cpu'.
        - The model is set to evaluation mode after loading the weights.
    """
    
    log_statement(model_name, f"[Load] Loading base model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name)
    log_statement(model_name, f"[Load] Loading fine-tuned weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    log_statement(model_name, "[Load] Model loaded and set to eval mode.")
    return model