#!/bin/bash

# List of models to run (fill this in with your model names)
models=('EleutherAI/pythia-14m' 'EleutherAI/pythia-31m' 'EleutherAI/pythia-70m' 'EleutherAI/pythia-160m')
enhancements=('embeddings' 'prompt')

# Path to the Python script
python_script="src/main.py"
test_script="src/test.py"

# Check if the list is empty
if [ ${#models[@]} -eq 0 ]; then
    echo "The models list is empty. Please add model names to the list."
    exit 1
fi

# Loop through each model and run the Python script
for model in "${models[@]}"; do
    for enhancement in $enhancements; do
        if [ "$enhancement" != "none" ]; then
            echo "Running model: $model with enhancement: $enhancement"
            python "$python_script" --model "$model" --enhancement "$enhancement"
            if [ $? -ne 0 ]; then
                echo "Error running model: $model"
                exit 1
            fi
        else
            echo "Running model: $model with no enhancement"
            python "$python_script" --model "$model"
            if [ $? -ne 0 ]; then
                echo "Error running model: $model"
                exit 1
            fi
        fi
    done
done

echo "All models have been processed successfully."
echo "Testing all models"

# python $test_script

echo "All models have been tested successfully."
echo "All done!"