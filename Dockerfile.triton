# Use NVIDIA Triton Server as the base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory (optional)
WORKDIR /workspace


# Install required dependencies for Python backend
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir tritonclient numpy opencv-python  # Add necessary dependencies

# Copy the model repository into the container
COPY models /models


# Start Triton Server and expose necessary ports
CMD ["tritonserver", "--model-repository=/models"]
