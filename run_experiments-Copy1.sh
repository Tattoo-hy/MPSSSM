#!/bin/bash
# MPS-SSM Efficient Experimental Pipeline - Sequential Execution Version
# Runs dataset configurations one after another to prevent GPU conflicts.

echo "=============================================="
echo "MPS-SSM Sequential Multivariate Pipeline"
echo "=============================================="
echo "Datasets: ETT (4), Weather (1), Traffic (1)"
echo "Each dataset config will run sequentially."
echo "Using up to ${NUM_GPUS} GPUs for each run."
echo "=============================================="

# --- Configuration ---
# Configuration files for each dataset group
ETT_CONFIG="configs/ett.yaml"
WEATHER_CONFIG="configs/weather.yaml"
TRAFFIC_CONFIG="configs/traffic.yaml"

# Total number of GPUs to make available to the orchestrator
NUM_GPUS=8

# --- Sanity Checks ---
# Check if all config files exist
for config in $ETT_CONFIG $WEATHER_CONFIG $TRAFFIC_CONFIG; do
    if [ ! -f "$config" ]; then
        echo "Error: Configuration file $config not found!"
        exit 1
    fi
done

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: Data directory not found! Please ensure datasets are in the data/ directory."
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# Create results directories
echo "Setting up result directories..."
mkdir -p results/{lambda_search,lambda_search_models,final_runs/logs}
mkdir -p logs # For experiment logs

# --- Stage 1: Lambda Search (Sequential Execution) ---
echo ""
echo "=============================================="
echo "Stage 1: Lambda Search (with model saving)"
echo "=============================================="
START_TIME_LAMBDA=$(date +%s)

# Run ETT experiments and wait for them to complete
echo ">>> Starting Lambda Search for ETT datasets..."
python main.py --mode lambda_search --config $ETT_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Lambda search for ETT failed!"
    exit 1
fi
echo ">>> ETT lambda search completed."

# Run Weather experiments and wait for them to complete
echo ">>> Starting Lambda Search for Weather dataset..."
python main.py --mode lambda_search --config $WEATHER_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Lambda search for Weather failed!"
    exit 1
fi
echo ">>> Weather lambda search completed."

# Run Traffic experiments and wait for them to complete
echo ">>> Starting Lambda Search for Traffic dataset..."
python main.py --mode lambda_search --config $TRAFFIC_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Lambda search for Traffic failed!"
    exit 1
fi
echo ">>> Traffic lambda search completed."

END_TIME_LAMBDA=$(date +%s)
ELAPSED_LAMBDA=$((END_TIME_LAMBDA - START_TIME_LAMBDA))
echo ""
echo "Stage 1 (Lambda Search) completed in $((ELAPSED_LAMBDA / 60)) minutes and $((ELAPSED_LAMBDA % 60)) seconds."
echo ""

# --- Stage 2: Testing with Saved Models (Sequential Execution) ---
echo "=============================================="
echo "Stage 2: Testing with Saved Models"
echo "=============================================="
START_TIME_TEST=$(date +%s)

# Test ETT models
echo ">>> Starting Testing for ETT datasets..."
python main.py --mode test_only --config $ETT_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Testing for ETT failed!"
    exit 1
fi
echo ">>> ETT testing completed."

# Test Weather models
echo ">>> Starting Testing for Weather dataset..."
python main.py --mode test_only --config $WEATHER_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Testing for Weather failed!"
    exit 1
fi
echo ">>> Weather testing completed."

# Test Traffic models
echo ">>> Starting Testing for Traffic dataset..."
python main.py --mode test_only --config $TRAFFIC_CONFIG --num_gpus $NUM_GPUS
if [ $? -ne 0 ]; then
    echo "Error: Testing for Traffic failed!"
    exit 1
fi
echo ">>> Traffic testing completed."

END_TIME_TEST=$(date +%s)
ELAPSED_TEST=$((END_TIME_TEST - START_TIME_TEST))
echo ""
echo "Stage 2 (Testing) completed in $((ELAPSED_TEST / 60)) minutes and $((ELAPSED_TEST % 60)) seconds."
echo ""

# --- Stage 3: Results Summary ---
echo "=============================================="
echo "Stage 3: Results Summarization"
echo "=============================================="
echo "Generating comprehensive result reports..."

python ./scripts/summarize_results.py --include_all_datasets
if [ $? -ne 0 ]; then
    echo "Error: Results summarization failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo "All experiments completed successfully!"
echo "=============================================="
echo ""
echo "Results have been saved to:"
echo "  - Lambda search results: results/lambda_search/*.json"
echo "  - Saved models: results/lambda_search_models/*.pth"
echo "  - Test results: results/final_runs/logs/*.json"
echo "  - Summary report: results/results_summary.md"
echo ""
echo "To view the full summary report:"
echo "  cat results/results_summary.md"
