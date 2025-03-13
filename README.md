# triton-inference

face detection: retinaface-mobilenetv2 (dynamic batch, onnx)
face recognition: mobilenetv2 (dynamix batch, onnx)

### How to Build & Run the Docker Container

**1️⃣ Build the Docker Image**

```bash
docker build -t my_triton_server .
```

**2️⃣ Run the Container**

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 my_triton_server
```
