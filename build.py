import glob
import os
import subprocess
import torch
from torch.utils.ffi import create_extension

cur_dir = os.path.split(os.path.abspath(__file__))[0]
sources = []
headers = []
defines = []
objects = []
with_cuda = False
inc_dir = os.path.join(cur_dir, 'sru/include')
if torch.cuda.is_available():
    nvcc_arg = ['nvcc', '-c', '-o']
    cu_lib = os.path.join(cur_dir, 'sru/src/sru_kernel.so')
    cu_src = os.path.join(cur_dir, 'sru/src/sru_kernel.cu')
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    cc_str = '{}{}'.format(major, minor)
    arch_arg = '-arch=compute_{0}'.format(cc_str)
    code_arg = '-code=compute_{0},sm_{0}'.format(cc_str)
    inc_args = ['-I%s' % inc_dir]
    compiler_arg = ['-Xcompiler', '-fPIC', '-shared']
    build_cmd = nvcc_arg + [cu_lib, cu_src, arch_arg, code_arg] + compiler_arg + inc_args
    print(subprocess.list2cmdline(build_cmd))
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        exit(-1)
    sources += ['sru/src/sru_cu.c']
    headers += ['sru/include/sru_cu.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    objects += [cu_lib]
    # objects += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    'sru._ext.sru_cu',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=objects,
    include_dirs=[inc_dir]
)

if __name__ == '__main__':
    ffi.build()
