from multiprocessing import cpu_count


def get_num_workers(batch_size: int):
    num_workers = min(batch_size, 8, cpu_count())
    return num_workers