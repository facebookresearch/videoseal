
from .dist import get_rank, is_dist_avail_and_initialized
import datetime
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
fropm numpy import np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total_sum = 0.0
        self.total_count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update metric with new value.

        Args:
            value (float, tensor, or numpy array): value to update.
            n (int, optional): weight for value. Defaults to 1.
        """
        # Ensure value is a float or numpy array for safe storage
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone().item()  # Convert tensor to float
        elif isinstance(value, np.ndarray):
            value = np.asscalar(value)  # Convert numpy array to float

        # Check for NaN values
        if value != value:  # NaN check
            return

        self.deque.append(value)
        self.total_count += n
        self.total_sum += value * n

    def synchronize_between_processes(self):
        """
        Synchronize the global average across all processes using torch.distributed.
        This will ensure all processes have the same view of the metric.
        """
        if not is_dist_avail_and_initialized():
            return

        # Create tensors for synchronization
        total_sum_tensor = torch.tensor(
            self.total_sum, dtype=torch.float64, device='cuda')
        total_count_tensor = torch.tensor(
            self.total_count, dtype=torch.float64, device='cuda')

        # Reduce across all processes
        dist.reduce(total_sum_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_count_tensor, dst=0, op=dist.ReduceOp.SUM)

        # On the main process (rank 0), normalize the global sum by the global count
        if dist.get_rank() == 0:
            self.total_sum = total_sum_tensor.item()
            self.total_count = total_count_tensor.item()

        # Broadcast the result back to all processes to ensure consistency
        dist.broadcast(total_sum_tensor, src=0)
        dist.broadcast(total_count_tensor, src=0)

        # Set the total_sum and total_count for all processes
        self.total_sum = total_sum_tensor.item()
        self.total_count = total_count_tensor.item()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total_sum / self.total_count if self.total_count > 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
        # for name, meter in self.meters.items():
            # print(name)
            # meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None,):

        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        if max_iter is None:
            max_iterations = len(iterable)
        else:
            max_iterations = max_iter if max_iter < len(
                iterable) else len(iterable)
        space_fmt = ':' + str(len(str(max_iterations))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                # 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):

            if it > max_iterations:
                break

            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if it % print_freq == 0 or it == max_iterations - 1:
                eta_seconds = iter_time.global_avg * (max_iterations - it)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        it, max_iterations, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                    ))
                    # memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        it, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header,
              total_time_str, total_time / max_iterations))
