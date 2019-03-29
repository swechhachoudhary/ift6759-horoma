#!/usr/bin/env bash

# Kill the script on error
set -e

# Generate the PBS configuration file from config.json
# Also create an experiment folder
python template/generate_script.py --config $1 --runner $2 --gpu_type $3

# Run the generated configuration
msub run.pbs
