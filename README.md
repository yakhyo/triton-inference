# 🚀 Triton Inference Server

This repository provides a **Triton Inference Server** setup for **face detection** and **face recognition** using **ONNX models** with dynamic batching.

## 📌 Models Used

- **Face Detection:** `RetinaFace-MobileNetV2` (Dynamic Batch, ONNX)
- **Face Recognition:** `MobileNetV2` (Dynamic Batch, ONNX)

---

## 🛠️ Build & Run the Triton Server

### **1️⃣ Build the Docker Image**

```bash
docker build -t tritonserver .
```

### **2️⃣ Run the Container**

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 tritonserver:latest
```

---

## 🐳 Using Docker Compose (**Recommended**)

### **Start the Server**

```bash
docker compose up --build -d
```

### **Stop the Server**

```bash
docker compose down
```

---

### Run API server

to visualize and test the API endpoints.

### **Run the API Server**

```bash
uvicorn api.face_api:app --host 0.0.0.0 --port 8000 --reload
```

### **Test the API Endpoints**

Open your browser and navigate to `http://localhost:8000/docs` to access the FastAPI Swagger UI. Here, you can test the available endpoints for face detection and recognition.

---

## 📌 TODO: TensorRT Optimization

To improve inference performance using **TensorRT**, enable GPU acceleration by adding the following configuration:

```config
execution_accelerators {
  gpu_execution_accelerator : [
    {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }  # Run inference in FP16 for better performance
    }
  ]
}
```

✅ **Upcoming Enhancements:**

- [ ] Integrate TensorRT optimizations
- [ ] Improve model inference speed

---

### 📢 **Contributions & Feedback**

Feel free to open an issue or submit a PR if you have improvements or suggestions! 🚀

---

### 📜 **License**

This project is **open-source** and available under the [MIT License](LICENSE).
