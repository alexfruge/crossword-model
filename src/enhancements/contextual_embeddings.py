import torch
import torch.nn as nn
from transformers import AutoModel

class CrosswordSolverWithEmbeddings(nn.Module):
    """
    Enhanced crossword solver that incorporates contextual embeddings
    alongside the language model to improve performance.
    """
    
    def __init__(self, base_model, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.base_lm = base_model  # The base language model (e.g., Pythia)

        # Load and freeze the embedding model
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        # Dummy input for dimension inference
        dummy_input = torch.ones((1, 1), dtype=torch.long)
        with torch.no_grad():
            dummy_output = self.embedding_model(dummy_input)
            emb_dim = dummy_output.last_hidden_state.size(-1)

        # Infer hidden size from base LM
        model_dim = 768  # default
        if hasattr(self.base_lm, 'cfg') and hasattr(self.base_lm.cfg, 'd_model'):
            model_dim = self.base_lm.cfg.d_model
        elif hasattr(self.base_lm, 'config') and hasattr(self.base_lm.config, 'hidden_size'):
            model_dim = self.base_lm.config.hidden_size

        print(f"Embedding dimension: {emb_dim}, Model dimension: {model_dim}")

        # Projection and optional adapter
        self.projection = nn.Linear(emb_dim, model_dim)
        self.adapter = nn.Linear(model_dim, model_dim)

        # Gated fusion
        self.fusion_gate = nn.Linear(model_dim, model_dim)

        # Output projection fallback if lm_head is not available
        self.output_projection = None
        if not hasattr(self.base_lm, 'lm_head'):
            vocab_size = getattr(getattr(self.base_lm, 'config', None), 'vocab_size', 50304)
            self.output_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.embedding_model.config.pad_token_id).long()

        with torch.no_grad():
            raw_embeddings = self.embedding_model(
                input_ids=input_ids.clamp(min=0, max=self.embedding_model.config.vocab_size - 1),
                attention_mask=attention_mask
            ).last_hidden_state

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(raw_embeddings.size())
            sum_embeddings = torch.sum(raw_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            pooled_embeddings = sum_embeddings / sum_mask.clamp(min=1e-9)
            pooled_embeddings = pooled_embeddings.unsqueeze(1).expand(-1, input_ids.size(1), -1)

        projected_embeddings = self.projection(pooled_embeddings)

        # Use run_with_cache to get hidden states from TransformerLens
        _, cache = self.base_lm.run_with_cache(input_ids)
        lm_hidden_states = cache["resid_post", -1]  # shape: [batch, seq_len, model_dim]

        if lm_hidden_states.size(-1) != projected_embeddings.size(-1):
            lm_hidden_states = self.adapter(lm_hidden_states)

        # Gated fusion
        gate = torch.sigmoid(self.fusion_gate(lm_hidden_states))
        combined_hidden_states = gate * projected_embeddings + (1 - gate) * lm_hidden_states

        # Output projection
        if hasattr(self.base_lm, 'lm_head'):
            logits = self.base_lm.lm_head(combined_hidden_states)
        else:
            logits = self.output_projection(combined_hidden_states)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, logits

        return logits

def create_enhanced_model(base_model_name):
    from model.load import load_model
    base_model = load_model(base_model_name)
    return CrosswordSolverWithEmbeddings(base_model)
