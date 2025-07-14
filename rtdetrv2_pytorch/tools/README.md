### Getting Started: Environment & Workflow

This guide provides a complete workflow from setting up the environment to training, exporting, and running inference with TensorRT.

#### **1. Environment Setup with Docker**

First, build and run the pre-configured Docker environment. This is the recommended way to ensure all dependencies and drivers are correctly aligned.

*   **Step 1.1: Clone the repository** (if you haven't already)
    ```bash
    git clone https://github.com/lyuwenyu/RT-DETR.git
    cd RT-DETR
    ```

*   **Step 1.2: Build and run the services in the background**
    From the project root directory, run:
    ```bash
    docker compose up --build -d
    ```
    *   `--build`: Forces a rebuild of the Docker image if the `Dockerfile` has changed.
    *   `-d`: Runs the containers in detached mode (in the background).

*   **Step 1.3: Access the running container**
    All subsequent commands should be run inside the container. To get a shell inside it:
    ```bash
    docker exec -it rtdetr-rtdetrv2_pytorch-1 bash
    ```
    *(Note: Your container name might be slightly different. Use `docker ps` to check.)*

---

#### **2. Training & Evaluation**

*   **To start or resume training** on 4 GPUs, use `torchrun`. Appending `&> train.log 2>&1 &` will run it in the background and save all logs.

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 \
        tools/train.py -c path/to/config.yml \
        -r path/to/checkpoint_to_resume_from.pth \
        --amp
    ```

*   **To run evaluation only**, add the `--test-only` flag to the training command.

    ```bash
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
        -c path/to/config.yml \
        -r path/to/trained_checkpoint.pth \
        --test-only
    ```

#### **3. Exporting Models for Deployment**

First, export the trained PyTorch model to the ONNX format. Then, for maximum performance on NVIDIA GPUs, convert the ONNX model to a TensorRT engine.

*   **Step 3.1: Export to ONNX**

    ```bash
    python tools/export_onnx.py \
        -c path/to/config.yml \
        -r path/to/trained_checkpoint.pth \
        --check
    ```

*   **Step 3.2: Convert ONNX to TensorRT Engine**

    Use the provided `onnx2trt.sh` script. It takes the ONNX file path and saves the resulting `.trt` engine in the same directory.

    ```bash
    bash tools/onnx2trt.sh /path/to/your/model.onnx
    ```

#### **4. Running Inference with TensorRT**

Use the `rtdetrv2_tensorrt.py` script to run inference on a single image with your generated TensorRT engine.

```bash
python references/deploy/rtdetrv2_tensorrt.py \
    --engine /path/to/your/model.trt \
    --image /path/to/your/image.jpg \
    --output /path/to/save/output.jpg \
    --threshold 0.5
```

### Utilities & Tips
- **Visualize training with TensorBoard**
  - Add `--summary-dir=/path/to/summary/dir` to your training command.
  - Launch TensorBoard:
    ``` bash
    tensorboard --host=0.0.0.0 --port=8989 --logdir=/path/to/summary/
    ```
  - **Forcefully release GPU memory** if a process gets stuck:
    - **Warning:** This is a drastic measure. use with caution.
    ``` bash
    ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9
    ```