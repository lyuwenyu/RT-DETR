# Running RT-DETR eval on Aerovect Image Dataset

## 1. Symlink datasets into their datasets folder
```
cd repos/RT-DETR/rtdetrv2_pytorch/dataset
ln -s <path to your repos folder>/data/datasets/aerovect_image_dataset aerovect_image_dataset
ln -s <path to your repos folder>/data/datasets/coco coco
```

## 2. Modify paths in config file `rtdetrv2_pytorch/configs/dataset/aerovect_detection.yml`
- Open rtdetrv2_pytorch/configs/dataset/aerovect_detection.yml
- Point to correct dataset paths, e.g., `/home/mei/repos/data/datasets/coco/annotations_trainval2017/annotations/instances_{split}2017.json`

## 3. Setup a virtual environment
```
python -m venv venv/rtdetrv2-env
source venv/rtdetrv2-env/bin/activate
```

## 4. Install requirements
```
cd repos/RT-DETR/rtdetrv2_pytorch/
pip install -r requirements.txt
```

## 5. Download checkpoint file
1. Download from their READMEs `RT-DETR/rtdetrv2_pytorch/README.md`
2. Store it somewhere (e.g., `RT-DETR/rtdetrv2_pytorch/checkpoints`)
3. Point to it later when running the eval command

## 6. Run eval

- Run eval on COCO dataset
    ```
    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml -r checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth --test-only
    ```
- Run eval on Aerovect Image dataset
    ```
    CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c configs/rtdetrv2/rtdetrv2_r101vd_6x_coco_eval_aerovect.yml -r checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth --test-only
    ```

## 7. Misc

- **Reduce batch size** to 8 (or up to how much your GPU can handle) to avoid CUDA out of memory error in `RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml`
- **Restart nvidia-driver** if it crashes in venv
    ```
    sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
    ```

# Configuring the evaluation parameters/options

- Define how Aerovect label categories map to COCO categories in `src/data/dataset/aerovect_dataset.py`
  - Define groups (e.g., `labels_map_truck_miniset`, `labels_map_truck_easyset`, etc.)
  - Toggle groups on/off in `labels_maps_list`
- Define where the tool can dump output files (e.g., final label map used, predicions, ids of images used) into `output/aerovect`
- Configure evaluation parameters in `src/solver/det_engine.py` by toggling the flags in the `evaluate()` function
  - `output_predictions`: save predictions into a file. This is useful for visualizing the results externally or feeding it into another metrics tool
  - `output_dir`: specify the output directory to dump output files
  - `additional_postprocess_methods`: can implement some additional post-processing methods (e.g., NMS, score thresholding) that RT-DETR doesn't naively support
    - Implement methods (e.g., `nms()`, `score_filter()`, etc.) and add to `additional_postprocess()`
    - The `additional_postprocess_methods` parameter should store method names and their configs (e.g., `"nms": {"nms_threshold": 0.2}`).
    - Can toggle method on/off

# Evaluation outputs

This tool currently outputs the following to `RT-DETR/rtdetrv2_pytorch/output`:
1. `labels_map.json`: This contains the taxonomy mapping used during the evaluation
1. `image_ids.txt`: This contains the ids of images used during the evaluation, these ids will be used to retrieve the image paths in the `predictions.txt` file
1. `predictions.txt`: This contains the predicted bounding boxes for each image. Predictions for each image are separated by `---` .
