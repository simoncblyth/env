#!/usr/bin/env python
"""
::

   ipython test_gpu.py -i 


"""
from chroma import gpu
import pycuda.driver as cuda

class Holder(object):
    def __init__(self, cuda_device=None):
        context = gpu.create_cuda_context(cuda_device)
        print context
        self.context = context

    def summary(self):
        d = self.context.get_device()
        return d.name(), d.compute_capability(), d.total_memory(), d.pci_bus_id(), d.count()

    def details(self):
        d = self.context.get_device()
        return "\n".join(["%-50s : %s " % (k,v) for k,v in d.get_attributes().items()])

    def __del__(self):
        self.context.pop()


if __name__ == '__main__':
    h = Holder()
    print h.summary()
    print h.details()

