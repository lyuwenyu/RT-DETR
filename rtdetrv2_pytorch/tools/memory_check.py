import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


import time
import pickle
from collections import defaultdict
from multiprocessing import Manager


from src.data.dataloader import DataLoader, BatchImageCollateFuncion
from src.data import transforms as T
from src.data.dataset.coco_dataset import CocoDetection, CocoDetection_share_memory


import torch
import psutil
from tabulate import tabulate


"""
testing memory usage of dataloader.

requires psutil and tabulate

pip install psutil tabulate
"""


class MemoryMonitor():
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids =  Manager().list(pids)

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: self.get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
    
    @staticmethod
    def get_mem_info(pid: int) -> dict[str, int]:
        res = defaultdict(int)
        for mmap in psutil.Process(pid).memory_maps():
            res['rss'] += mmap.rss
            res['pss'] += mmap.pss
            res['uss'] += mmap.private_clean + mmap.private_dirty
            res['shared'] += mmap.shared_clean + mmap.shared_dirty
            if mmap.path.startswith('/'):
                res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
        return res


def test_dataset(
        dataset_class,
        range_num=None,
        img_folder="./dataset/coco/train2017/",
        ann_file="./dataset/coco/annotations/instances_train2017.json",
        **kwargs):

    train_dataset = dataset_class(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms = T.Compose([T.RandomPhotometricDistort(p=0.5),
                                T.RandomZoomOut(fill=0),
                                T.RandomIoUCrop(p=0.8),
                                T.SanitizeBoundingBoxes(min_size=1),
                                T.RandomHorizontalFlip(),
                                T.Resize(size=[640, 640], ),
                                T.SanitizeBoundingBoxes(min_size=1),
                                T.ConvertPILImage(dtype='float32', scale=True),
                                T.ConvertBoxes(fmt='cxcywh', normalize=True)],
                                policy={'name': 'stop_epoch',
                                        'epoch': 71 ,
                                        'ops': ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']}),
        return_masks=False,
        remap_mscoco_category=True,
        **kwargs)
    
    if range_num is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(range_num))

    return train_dataset


def test_dataloader(
        dataset,
        worker_init_fn,
        batch_size=4,
        shuffle=True, 
        num_workers=4):

    collate_fn = BatchImageCollateFuncion(scales=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800], stop_epoch=71)

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn, 
        drop_last=True, 
        worker_init_fn=worker_init_fn)


def main(**kwargs):
    def hook_pid(worker_id):
        pid = os.getpid()
        monitor.pids.append(pid)
        print(f"tracking {worker_id} PID: {pid}")

    monitor = MemoryMonitor()

    dataloader = test_dataloader(
        dataset=test_dataset(**kwargs), 
        worker_init_fn=hook_pid,
        batch_size=32, 
        num_workers=2)

    t = time.time()

    for i, (samples, targets) in enumerate(dataloader):
        # fake read the data
        samples = pickle.dumps(samples)
        targets = pickle.dumps(targets)

        if i % 10 == 0:
            print(monitor.table())
            print(f"totle pss : {sum([k[1]['pss'] / 1024 / 1024 / 1024 for k in monitor.data.items()]):.3f}GB")
            print(f"iteration : {i} / {len(dataloader)}, time : {time.time() - t:.3f}")
            t = time.time()


if __name__ == '__main__':
    # main(dataset_class=CocoDetection, range_num=10000)
    # main(dataset_class=CocoDetection_share_memory, share_memory=False, range_num=10000)
    main(dataset_class=CocoDetection_share_memory, share_memory=True, range_num=10000)