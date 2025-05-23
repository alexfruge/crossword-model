import torch
import time

from utils import log_statement

def train(model, model_name, dataloader, device, num_epochs=3, lr=5e-5, enhancement=None):
    """
    Trains a given model using the provided dataloader and optimizer settings.
    Args:
        model (torch.nn.Module): The model to be trained.
        model_name (str): The name of the model, used for logging.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.        
        device (torch.device): The device (CPU or GPU) to perform training on.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 3.
        lr (float, optional): Learning rate for the AdamW optimizer. Defaults to 5e-5.
        enhancement (str, optional): The type of enhancement applied to the model. Defaults to None.
    Returns:
        list: A list of batch loss values recorded every 10 batches.
    Notes:
        - The function uses CrossEntropyLoss for calculating the loss.
        - The training progress, including batch loss and epoch duration, is printed to the console.
        - The model is moved to the specified device before training begins.
    """

    log_statement(model_name, "[Training] Starting training loop...", enhancement)
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch_losses = []

    for epoch in range(num_epochs):
        log_statement(model_name, f"\n[Epoch {epoch + 1}/{num_epochs}] Starting...", enhancement)
        total_loss = 0
        start_time = time.time()

        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                log_statement(model_name, f"[Epoch {epoch + 1}] Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}", enhancement)
                batch_losses.append(loss.item())

        avg_loss = total_loss / len(dataloader)
        duration = time.time() - start_time
        log_statement(model_name, f"[Epoch {epoch + 1}] Completed in {duration:.2f}s - Avg Loss: {avg_loss:.4f}", enhancement)

    return batch_losses