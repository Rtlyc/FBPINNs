#!/bin/bash

# Generate Config File if Necessary
echo "***Generating Config File***"
python data_processing/gibson_config_gen.py

# Data Preprocessing (Scaling Mesh)
echo "***Scaling Mesh***"
python data_processing/scale_mesh.py

# Data Preprocessing (Generating Training Data)
echo "***Generating Training Data***"
python data_processing/mesh_sample.py

# # Training
echo "***Training the model***"
python main.py

# Evaluation
echo "***Running the evaluation script***"
python eval/eval_planning.py


