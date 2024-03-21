# Use the official Miniconda image as a base image
FROM pytorch/pytorch:latest

# Set environment variables
ENV CONDA_ENV_NAME=fbpinns

# Update conda and create a new environment
RUN conda create -n $CONDA_ENV_NAME python=3

# Activate the environment
SHELL ["conda", "run", "-n", "fbpinns", "/bin/bash", "-c"]

# Copy the current directory (assuming Dockerfile is in the root of your project) into the container
COPY . /app

# Change working directory to the copied directory
WORKDIR /app

RUN pip install --upgrade pip

# Install the project in editable mode
RUN pip install -e .

# CUDA 12 installation
# Note: wheels only available on linux.
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # CUDA 11 installation
# # Note: wheels only available on linux.
# RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install jupyter

RUN pip install torch

RUN pip install matplotlib