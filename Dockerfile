# Use NVIDIA Triton Server as the base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory (optional)
WORKDIR /workspace

# Copy the model repository into the container
COPY models /models

# Start Triton Server and expose necessary ports
CMD ["tritonserver", "--model-repository=/models"]
