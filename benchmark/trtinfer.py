'''by lyuwenyu
'''

import time 
import contextlib
from collections import namedtuple, OrderedDict

import torch
import numpy as np
import tensorrt as trt

from utils import TimeProfiler

class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        '''build binddings
        '''
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs


    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

        elif self.backend == 'cuda':
            self.stream.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 

