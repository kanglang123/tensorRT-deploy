# tensorRT-deploy
模型pth-onnx-engine格式转换。

使用TensorRT部署模型的demo.

现在这个代码中的很多函数官方已经不用了，所以直接跑不起来。我根据最新的官方文档做了一些函数的更改，可以跑起来。贴在这里，仅供参考。

## Installation
1. 安装TensorRT库：

    下载地址：[NVIDIA TensorRT 10.x Download](https://developer.nvidia.com/tensorrt/download/10x)，根据自己电脑配置选择相应的下载链接。
    我的电脑是Ubuntu20.04，所以下载的链接为：[TensorRT 10.0 EA for Linux x86_64 and CUDA 12.0 to 12.4 TAR Package](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/tars/TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-12.4.tar.gz)

    ```
    tar -zxvf TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-12.4.tar.gz
    ```
    在.bashrc中添加环境变量：
    ```
    export LD_LIBRARY_PATH=/home/wyk/TensorRT-10.0.0.6/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/home/wyk/TensorRT-10.0.0.6/lib::$LIBRARY_PATH
    ```
    安装wheel文件：
    ```
    pip install onnx_graphsurgeon-0.5.0-py2.py3-none-any.whl
    pip install tensorrt-10.0.0b6-cp310-none-linux_x86_64.whl # 根据自己的python版本安装，我的python是3.10.0版本
    ```

2. 下载仓库：
    ```
    git clone https://github.com/kanglang123/tensorRT-deploy
    ```
## Using Tutorials
1. pth2onnx.py
   
   下载开源模型和图像，进行推理，并将pth模型转换为onnx格式保存。

2. check_onnx.py

    加载保存的onnx格式模型，根据模型推理后输出的图像，检查保存onnx模型是否正确。

3. onnx2tensorrt.py

    使用pytorch定义一个模型，转换为onnx格式保存。
    加载保存的onnx模型并使用TensorRT将onnx格式模型转为engine格式模型。

4. infer_tensorrt.py

    使用保存的engine格式模型进行推理。

## 参考链接：

1. [知乎：模型部署那些事](https://zhuanlan.zhihu.com/p/547624036)

2. [NVIDIA TensorRT API Docmentation](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/pyCore.html)

