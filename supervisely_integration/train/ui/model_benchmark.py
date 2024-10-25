import os
from typing import Dict, List

import torch

import supervisely as sly
import supervisely_integration.train.globals as g
from supervisely import DatasetInfo, ProjectInfo
from supervisely.app.widgets import (
    Field,
    Progress,
    ReportThumbnail,
    SlyTqdm,
    TrainValSplits,
)
from supervisely.io.fs import get_file_name, get_file_name_with_ext
from supervisely.nn.benchmark import ObjectDetectionBenchmark
from supervisely.nn.inference import SessionJSON
from supervisely_integration.train.serve import RTDETRModelMB


def get_eval_results_dir_name(api: sly.Api, task_id: int, project_info: ProjectInfo) -> str:
    task_info = api.task.get_info_by_id(task_id)
    task_dir = f"{task_id}_{task_info['meta']['app']['name']}"
    eval_res_dir = f"/model-benchmark/evaluation/{project_info.id}_{project_info.name}/{task_dir}/"
    eval_res_dir = api.storage.get_free_dir_name(sly.env.team_id(), eval_res_dir)
    return eval_res_dir


def run_model_benchmark(
    api: sly.Api,
    root_source_path: str,
    local_artifacts_dir: str,
    remote_weights_dir: str,
    remote_config_path: str,
    project_info: ProjectInfo,
    dataset_infos: List[DatasetInfo],
    ds_name_to_id: Dict[str, int],
    train_val_split: TrainValSplits,
    train_set: list,
    val_set: list,
    selected_classes: List[str],
    use_speedtest: bool,
    model_benchmark_report: ReportThumbnail,
    creating_report_f: Field,
    model_benchmark_pbar: SlyTqdm,
    model_benchmark_pbar_secondary: Progress,
) -> bool:
    model_benchmark_done = False
    try:
        best_filename = get_file_name_with_ext(g.best_checkpoint_path)
        checkpoint_path = os.path.join(remote_weights_dir, best_filename)

        sly.logger.info(f"Creating the report for the best model: {best_filename!r}")
        creating_report_f.show()
        model_benchmark_pbar.show()
        model_benchmark_pbar(message="Starting Model Benchmark evaluation...", total=1)

        repo_root_path = os.path.dirname(os.path.dirname(root_source_path))

        # 0. Serve trained model
        m = RTDETRModelMB(
            model_dir=local_artifacts_dir + "/weights",
            use_gui=False,
            custom_inference_settings=os.path.join(
                repo_root_path, "supervisely_integration", "serve", "inference_settings.yaml"
            ),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sly.logger.info(f"Using device: {device}")

        deploy_params = dict(
            device=device,
            runtime=sly.nn.inference.RuntimeType.PYTORCH,
            model_source="Custom models",
            task_type="object detection",
            checkpoint_name=best_filename,
            checkpoint_url=checkpoint_path,
            config_url=remote_config_path,
        )
        m._load_model(deploy_params)
        m.serve()
        # m.model.overrides["verbose"] = False
        session = SessionJSON(api, session_url="http://localhost:8000")
        sly.fs.remove_dir(g.data_dir + "/benchmark")

        # 1. Init benchmark (todo: auto-detect task type)
        benchmark_dataset_ids = None
        benchmark_images_ids = None
        train_dataset_ids = None
        train_images_ids = None

        split_method = train_val_split._content.get_active_tab()

        if split_method == "Based on datasets":
            if hasattr(train_val_split._val_ds_select, "get_selected_ids"):
                benchmark_dataset_ids = train_val_split._val_ds_select.get_selected_ids()
                train_dataset_ids = train_val_split._train_ds_select.get_selected_ids()
            else:
                benchmark_dataset_ids = [
                    ds_name_to_id[d] for d in train_val_split._val_ds_select.get_value()
                ]
                train_dataset_ids = [
                    ds_name_to_id[d] for d in train_val_split._train_ds_select.get_value()
                ]
        else:

            def get_image_infos_by_split(split: list):
                ds_infos_dict = {ds_info.name: ds_info for ds_info in dataset_infos}
                image_names_per_dataset = {}
                for item in split:
                    image_names_per_dataset.setdefault(item.dataset_name, []).append(item.name)
                image_infos = []
                for (
                    dataset_name,
                    image_names,
                ) in image_names_per_dataset.items():
                    if "/" in dataset_name:
                        dataset_name = dataset_name.split("/")[-1]
                    ds_info = ds_infos_dict[dataset_name]
                    image_infos.extend(
                        api.image.get_list(
                            ds_info.id,
                            filters=[
                                {
                                    "field": "name",
                                    "operator": "in",
                                    "value": image_names,
                                }
                            ],
                        )
                    )
                return image_infos

            val_image_infos = get_image_infos_by_split(val_set)
            train_image_infos = get_image_infos_by_split(train_set)
            benchmark_images_ids = [img_info.id for img_info in val_image_infos]
            train_images_ids = [img_info.id for img_info in train_image_infos]

        bm = ObjectDetectionBenchmark(
            api,
            project_info.id,
            output_dir=g.data_dir + "/benchmark",
            gt_dataset_ids=benchmark_dataset_ids,
            gt_images_ids=benchmark_images_ids,
            progress=model_benchmark_pbar,
            progress_secondary=model_benchmark_pbar_secondary,
            classes_whitelist=selected_classes,
        )

        train_info = {
            "app_session_id": g.TASK_ID,
            "train_dataset_ids": train_dataset_ids,
            "train_images_ids": train_images_ids,
            "images_count": len(train_set),
        }
        bm.train_info = train_info

        # 2. Run inference
        bm.run_inference(session)

        # 3. Pull results from the server
        gt_project_path, dt_project_path = bm.download_projects(save_images=False)

        # 4. Evaluate
        bm._evaluate(gt_project_path, dt_project_path)

        # 5. Upload evaluation results
        eval_res_dir = get_eval_results_dir_name(api, g.TASK_ID, project_info)
        bm.upload_eval_results(eval_res_dir + "/evaluation/")

        # 6. Speed test
        if use_speedtest:
            bm.run_speedtest(session, project_info.id)
            model_benchmark_pbar_secondary.hide()
            bm.upload_speedtest_results(eval_res_dir + "/speedtest/")

        # 7. Prepare visualizations, report and upload
        bm.visualize()
        remote_dir = bm.upload_visualizations(eval_res_dir + "/visualizations/")
        report = bm.upload_report_link(remote_dir)

        # 8. UI updates
        benchmark_report_template = api.file.get_info_by_path(
            sly.env.team_id(), remote_dir + "template.vue"
        )
        model_benchmark_done = True
        creating_report_f.hide()
        model_benchmark_report.set(benchmark_report_template)
        model_benchmark_report.show()
        model_benchmark_pbar.hide()
        sly.logger.info(
            f"Predictions project name: {bm.dt_project_info.name}. Workspace_id: {bm.dt_project_info.workspace_id}"
        )
        sly.logger.info(
            f"Differences project name: {bm.diff_project_info.name}. Workspace_id: {bm.diff_project_info.workspace_id}"
        )
    except Exception as e:
        sly.logger.error(f"Model benchmark failed. {repr(e)}", exc_info=True)
        creating_report_f.hide()
        model_benchmark_pbar.hide()
        model_benchmark_pbar_secondary.hide()
        try:
            if bm.dt_project_info:
                api.project.remove(bm.dt_project_info.id)
            if bm.diff_project_info:
                api.project.remove(bm.diff_project_info.id)
        except Exception as e2:
            pass
    return model_benchmark_done
