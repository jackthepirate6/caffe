import os

os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('..')
os.chdir('content')
os.chdir('caffe')
os.chdir('python')
os.chdir('caffe')

#sys.path.append('/content/caffe/python/caffe')
#sys.path.append('/content/caffe/src/caffe/solvers')
#sys.path.append('/content/caffe/python/caffe/test')
print("this is sys path ",sys.path)
print("this is current working directory")
print(os.getcwd())

from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver, NCCL, Timer
from ._caffe import init_log, log, set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed, solver_count, set_solver_count, solver_rank, set_solver_rank, set_multiprocess, has_nccl
from ._caffe import __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
