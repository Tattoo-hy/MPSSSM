#!/bin/bash
# Kill all running MPS-SSM experiments

echo "Killing all MPS-SSM related processes..."

# Kill run_experiment.py processes
pkill -f "run_experiment.py" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Killed run_experiment.py processes"
else
    echo "  No run_experiment.py processes found"
fi

# Kill main.py processes
pkill -f "main.py.*lambda_search" 2>/dev/null
pkill -f "main.py.*test_only" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Killed main.py processes"
else
    echo "  No main.py processes found"
fi

# Kill any python processes using the MPS-SSM models
pkill -f "models.mps_ssm" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Killed model training processes"
else
    echo "  No model training processes found"
fi

# Wait a moment
sleep 2

# Check if any processes remain
remaining=$(ps aux | grep -E "(run_experiment|main\.py|mps_ssm)" | grep -v grep | wc -l)
if [ $remaining -eq 0 ]; then
    echo ""
    echo "All MPS-SSM processes successfully terminated."
else
    echo ""
    echo "Warning: $remaining processes may still be running:"
    ps aux | grep -E "(run_experiment|main\.py|mps_ssm)" | grep -v grep
    echo ""
    echo "You may need to kill them manually with:"
    echo "  kill -9 <PID>"
fi

# Clear GPU memory
echo ""
echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv