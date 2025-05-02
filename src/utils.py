import os
from datetime import datetime

# Dictionary to store the initial log time for each model
_MODEL_START_TIMES = {}

def log_statement(model_name: str, statement: str):
    # Get or create the initial log time for this model
    if model_name not in _MODEL_START_TIMES:
        _MODEL_START_TIMES[model_name] = datetime.now()
    
    # Create filename-safe timestamp
    start_time = _MODEL_START_TIMES[model_name]
    filename_time = start_time.strftime("%Y%m%d_%H%M%S")
    
    # Create current timestamp for log entry
    entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare log components
    model_basename = os.path.basename(model_name)
    log_dir = "logs"
    log_filename = f"{model_basename}_{filename_time}.log"
    log_path = os.path.join(log_dir, log_filename)
    log_entry = f"[{entry_time}] {statement}\n"
    
    # Print to console and write to file
    print(statement)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)