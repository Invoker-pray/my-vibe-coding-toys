# 环境搭建

当前环境中，系统内核为：`archlinux:6.18.18-1-lts`，

相关驱动和环境依赖安装如下：

```bash
sudo pacman -S nvidia-open-dkms nvidia-utils nvidia-settings
sudo mkinitcpio -P
```

可在重启之后验证：

```bash
nvidia-smi
lsmod | grep nvidia
```

`nvidia-smi`之后应显示显卡信息和驱动版本，`lsmod | grep nvidia`之后应至少有`nvidia-drm` `nvidia` `nvidia-modeset`.

# optimus笔记本prime

我的电脑目前是RTX 3060 laptop版本，他是Intel iGPU + NVIDIA dGPU的optimus架构，PRIME render offload是NVIDIA官方支持的切换方式，通过prime-run脚本在dGPU上运行指定程序，同时保持桌面渲染在iGPU上节省电量。

因此需要配置nvidia-prime.

```bash
sudo pacman -S nvidia-prime

# 验证 prime offload
prime-run nvidia-smi
prime-run glxinfo | grep "OpenGL renderer" # 显示：string: NVIDIA Geforce RTX 3060 Laptop GPU/PCIe/SSE2
```

对于 CUDA / pynvml，不需要切换渲染到 NVIDIA，只需确保 NVIDIA 卡在 CUDA 应用启动前已上电，CUDA 应用会自动加载所需内核模块。

# CUDA tools kit

安装CUDA tools kit:

```bash
sudo pacman -S cuda cudnn
```

cuda包含nvcc, runtime等，cudnn对于deeplearning有需求可下载。

验证安装：

```bash
nvcc --version #显示NVIDIA(R) Cuda compiler driver
python -c "import torch; print(torch.cuda.is_available())"
```

# python env

直接使用GPU-monitor-venv下的rye pyproject.toml即可。

验证pynvml:

```bash
python -c "
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
print(nvmlDeviceGetName(h))
print(f'Driver: {nvmlSystemGetDriverVersion()}')
nvmlShutdown()
"
```

输出GPU和driver信息则说明链路正常。
