import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from utils import log_statement


class LengthConditionedCrosswordSolver(nn.Module):
    """
    A crossword solver that explicitly conditions on the target answer length.
    """
    
    def __init__(self, base_model, vocab_size, max_answer_length=20):
        super().__init__()
        self.base_lm = base_model  # The base language model
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        
        # Length embedding - for conditioning on target length
        self.length_embedding = nn.Embedding(self.max_answer_length + 1, self.base_lm.config.hidden_size)
        
        # Projection for combining length information with LM outputs
        self.length_projection = nn.Linear(self.base_lm.config.hidden_size, self.base_lm.config.hidden_size)
        
        # Length-specific output heads (one for each possible answer length)
        self.length_specific_heads = nn.ModuleList([
            nn.Linear(self.base_lm.config.hidden_size, self.vocab_size)
            for _ in range(self.max_answer_length + 1)
        ])
        
        # Letter position embedding (helps model learn position-specific patterns)
        self.position_embedding = nn.Embedding(self.max_answer_length, self.base_lm.config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, answer_length=None, labels=None):
        """
        Forward pass with explicit length conditioning.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            answer_length: Expected length of the answer (batch)
            labels: Target labels for training
            
        Returns:
            Logits or (loss, logits) if labels are provided
        """
        # Get outputs from base model
        outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get the hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]
        
        # If no specific answer length is provided, extract it from the input
        # This assumes the length is encoded in the prompt (e.g., "Clue: X (5)")
        if answer_length is None:
            # Extract from input_ids using a heuristic - for real implementation
            # you would want a more robust extraction method
            answer_length = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device) * 5
        
        # Clamp answer length to valid range
        answer_length = torch.clamp(answer_length, 1, self.max_answer_length)
        
        # Get length embeddings
        length_emb = self.length_embedding(answer_length).unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Combine length information with hidden states
        length_info = self.length_projection(length_emb)
        enhanced_hidden = hidden_states + length_info
        
        # Use the length-specific output head for each example in the batch
        logits = torch.zeros(
            hidden_states.shape[0],  # batch size
            hidden_states.shape[1],  # sequence length
            self.vocab_size,         # vocabulary size
            device=hidden_states.device
        )
        
        # Apply the appropriate output head based on answer length
        for idx in range(input_ids.shape[0]):  # For each example in the batch
            length_idx = min(answer_length[idx].item(), self.max_answer_length)
            logits[idx] = self.length_specific_heads[length_idx](enhanced_hidden[idx])
        
        if labels is not None:
            # Compute loss if labels are provided
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, logits
            
        return logits
        
    def generate_answer(self, tokenizer, clue, expected_length, device='cpu', max_new_tokens=20):
        """Generate an answer with explicit length conditioning."""
        # Format the prompt
        prompt = f"Task: Solve the crossword clue:\nCrossword clue: {clue} ({expected_length})\nAnswer:"
        
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        answer_length = torch.tensor([expected_length], dtype=torch.long, device=device)
        
        # Generate the answer with length conditioning
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Generate the next token
                logits = self(
                    input_ids=generated, 
                    answer_length=answer_length
                )
                
                # If we've already generated the complete answer with the right length,
                # bias the model against adding more characters
                if generated.size(1) - input_ids.size(1) >= expected_length:
                    # Increase probability of EOS token
                    logits[0, -1, tokenizer.eos_token_id] = logits[0, -1, tokenizer.eos_token_id] * 10
                    
                    # Decrease probability of alphabetic tokens
                    for char_idx in range(ord('A'), ord('Z')+1):
                        token_id = tokenizer.encode(chr(char_idx))[0]
                        logits[0, -1, token_id] = logits[0, -1, token_id] * 0.1
                
                # Get the next token
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = torch.cat((generated, next_token_id.unsqueeze(0)), dim=1)
                
                # Break early on EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
        
        # Decode the generated answer
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer_raw = decoded.split("Answer:")[-1].strip()
        
        # Clean up the answer (remove non-alphabetic characters and limit to expected length)
        answer = ''.join(char for char in answer_raw if char.isalpha()).upper()
        if expected_length and len(answer) > expected_length:
            answer = answer[:expected_length]
            
        return answer


def create_length_aware_model(base_model_name):
    """
    Creates a length-aware crossword solver model.
    
    Args:
        base_model_name: Name of the base model to enhance
        
    Returns:
        A LengthConditionedCrosswordSolver model
    """
    from model.load import load_model
    from transformers import GPT2Tokenizer, AutoTokenizer
    
    # Load the base model
    base_model = load_model(base_model_name)
    
    # Get vocab size from the tokenizer
    if "gpt2" in base_model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(base_model_name.split("/")[-1])
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    vocab_size = len(tokenizer)
    
    # Create the length-aware model
    length_aware_model = LengthConditionedCrosswordSolver(base_model, vocab_size)
    
    return length_aware_model


class Constrained_Beam_Generator:
    """
    Implements constrained beam search for generating fixed-length answers.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, clue, expected_length, beam_size=5, max_steps=30):
        """
        Generate an answer using constrained beam search.
        
        Args:
            clue: The crossword clue
            expected_length: Expected length of the answer
            beam_size: Beam search width
            max_steps: Maximum number of generation steps
            
        Returns:
            The best answer candidate
        """
        # Format the prompt
        prompt = f"Task: Solve this crossword clue:\nClue: {clue} ({expected_length})\nAnswer:"
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Initialize beam with just the input prompt
        beam = [(input_ids, 0.0)]  # (token_ids, score)
        
        # Track completed sequences
        completed = []
        
        # Begin beam search
        for step in range(max_steps):
            # Collect all candidates for this step
            all_candidates = []
            
            # For each candidate in the beam
            for seq, score in beam:
                # Skip if this candidate is already completed
                if step > 0 and seq[0, -1].item() == self.tokenizer.eos_token_id:
                    completed.append((seq, score))
                    continue
                    
                # Get next token probabilities
                with torch.no_grad():
                    if hasattr(self.model, 'generate_answer'):
                        # Use model's custom generation method if available
                        logits = self.model(seq)
                    else:
                        # Default to regular forward pass
                        logits = self.model(seq)[0]
                        
                    # Get probabilities for next token
                    next_token_logits = logits[0, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Check if we've generated the expected number of tokens for the answer
                tokens_generated = seq.size(1) - input_ids.size(1)
                
                if tokens_generated >= expected_length:
                    # Strongly bias toward EOS token
                    next_token_probs = torch.zeros_like(next_token_probs)
                    next_token_probs[self.tokenizer.eos_token_id] = 1.0
                else:
                    # Apply constraints: 
                    # 1. Encourage alphabetic tokens
                    # 2. Discourage special tokens except EOS
                    for i in range(len(next_token_probs)):
                        token = self.tokenizer.decode([i])
                        if token.isalpha():
                            # Slightly boost alphabetic tokens
                            next_token_probs[i] *= 1.2
                        elif i != self.tokenizer.eos_token_id:
                            # Discourage non-alphabetic, non-EOS tokens
                            next_token_probs[i] *= 0.1
                
                # Get top-k tokens
                topk_probs, topk_indices = torch.topk(next_token_probs, beam_size)
                
                # Create new candidates
                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                    # Skip tokens with very low probability
                    if prob < 1e-5:
                        continue
                        
                    # Create new sequence with this token
                    new_seq = torch.cat([seq, torch.tensor([[idx]], device=self.device)], dim=1)
                    
                    # Calculate new score (log probability)
                    new_score = score + torch.log(torch.tensor(prob)).item()
                    
                    # Add to candidates
                    all_candidates.append((new_seq, new_score))
            
            # If no candidates were generated or all sequences complete, stop
            if not all_candidates:
                break
                
            # Select top-k candidates as the new beam
            beam = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # If all sequences in beam end with EOS, we're done
            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in beam):
                completed.extend(beam)
                break
        
        # If we have completed sequences, select the best one
        # Otherwise, select the best sequence from the beam
        candidates = completed if completed else beam
        best_seq, _ = max(candidates, key=lambda x: x[1])
        
        # Decode the best sequence
        answer_raw = self.tokenizer.decode(best_seq[0], skip_special_tokens=True)
        answer_text = answer_raw.split("Answer:")[-1].strip()
        
        # Clean up and format the answer
        answer = ''.join(char for char in answer_text if char.isalpha()).upper()
        if len(answer) > expected_length:
            answer = answer[:expected_length]
            
        return answer