import os

def log_statement(model_name: str, statement: str):
    # Print the statement to the console
    print(statement)
    
    # Ensure the logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Construct the log file path
    log_file_path = os.path.join(logs_dir, f"{model_name.split("/")[-1]}.log")
    
    # Append the statement to the log file
    with open(log_file_path, "a") as log_file:
        log_file.write(statement + "\n")