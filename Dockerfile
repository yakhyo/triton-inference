# Use NVIDIA Triton Server as the base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set working directory
WORKDIR /workspace

# Copy the model repository
COPY models /models

# Copy Python inference scripts and requirements
COPY trt_detection.py /workspace/trt_detection.py
COPY trt_recognition.py /workspace/trt_recognition.py
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies (if needed)
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Start Triton Server and keep the container running
CMD ["tritonserver", "--model-repository=/models"]
