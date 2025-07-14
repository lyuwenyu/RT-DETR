### Getting Started: A Complete Workflow

This guide provides a complete, step-by-step workflow from setting up the environment to training, exporting, and running inference with TensorRT.

#### **1. Environment Setup with Docker (Recommended)**

Using Docker is the recommended way to ensure all dependencies, drivers, and CUDA versions are perfectly aligned. This eliminates "it works on my machine" issues.

*   **Step 1.1: Build and Run the Container**

    From the project's root directory, run `docker compose`. This will build the image based on the `Dockerfile` and start the service in the background.

    ```bash
    docker compose up --build -d
    ```

*   **Step 1.2: Verify the Container is Running**

    Check that the container is up and running. Note its name for the next step.
    ```bash
    docker ps
    ```

---

#### **2. Training & Evaluation (Using `docker attach`)**

This method directly attaches your terminal to the container's main process. It's simple but requires careful handling to avoid terminating your session.

*   **Step 2.1: Attach to the Container**

    Attach your terminal to the running container. You will be dropped into a bash shell.

    ```bash
    docker attach <your_container_name>
    ```

*   **Step 2.2: Run the Training Command**

    Now, *inside the attached shell*, run your training command. `torchrun` will automatically use the GPUs assigned to the container. **Do not run it in the background (`&`)**.

    ```bash
    # Example for 4 GPUs assigned to the container
    torchrun --nproc_per_node=4 --master-port=8989 \
        tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        --amp
    ```

*   **Step 2.3: Detach from the Session (IMPORTANT!)**

    With your training running, you can safely detach and leave it running.

    **WARNING:** **DO NOT PRESS `Ctrl+C`**. This will kill the training process and potentially the entire container.

    To safely detach, press the sequence: **`Ctrl+P`**, followed immediately by **`Ctrl+Q`**.

    You will return to your local terminal, and the container will continue running the training in the background.

*   **Step 2.4: Re-attach to Your Session**

    To check on your training progress, simply run the `docker attach` command again. You will see the live output from your training command.

    ```bash
    docker attach <your_container_name>
    ```
    (Remember to detach with `Ctrl+P`, `Ctrl+Q` when you're done.)

---

#### **3. Exporting & Inference**

For tasks like exporting or running inference, which don't need to run for days, it's safer to use `docker exec` to open a new, separate shell.

*   **Step 3.1: Open a New Shell in the Container**
    ```bash
    docker exec -it <your_container_name> bash
    ```

*   **Step 3.2: Run Export or Inference Commands**
    Now, inside this new shell, run your commands.
    ```bash
    # Export to ONNX
    python tools/export_onnx.py \
        -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
        -r path/to/trained_checkpoint.pth \
        --check
    ```
    
    ```
    # Convert to TensorRT
    bash tools/onnx2trt.sh /path/to/your/model.onnx
    ```

    ```
    # RUN TRT Inference
    python references/deploy/rtdetrv2_tensorrt.py \
    --engine /path/to/your/model.trt \
    --image /path/to/your/image.jpg \
    --output /path/to/save/output.jpg \
    --threshold 0.5
    ```

### Utilities & Tips

*   **Visualize training with TensorBoard:**
    *   Use the standard port `6006` to avoid conflicts with training.
    *   Ensure the port `6006` is exposed in your `docker-compose.yml`.

    ```bash
    # Inside the container
    tensorboard --logdir=path/to/summary/ --host=0.0.0.0 --port=6006
    ```

*   **Managing the Container Lifecycle:**
    *   **To temporarily stop** the container without deleting it (e.g., to pause training and resume later):
        ```bash
        docker compose stop
        ```
        You can restart it later with `docker compose start`.

    *   **To stop and completely remove** the container, network, and volumes:
        ```bash
        docker compose down
        ```