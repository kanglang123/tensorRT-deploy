from typing import Union, Optional, Sequence,Dict,Any 

import cv2
import torch 
import tensorrt as trt 
import numpy as np
 

class TRTWrapper(torch.nn.Module): 
    def __init__(self, engine: Union[str, trt.ICudaEngine], 
                 output_names: Optional[Sequence[str]] = None) -> None: 
        super().__init__() 
        self.engine = engine 
        if isinstance(self.engine, str): 
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: # 创建该模型的实例运行界面runtime
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read()     # 读取引擎文件
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) # 反序列化引擎
        self.context = self.engine.create_execution_context() # 创建执行上下文
        # 准备输入数据
        names = [_ for _ in self.engine] 
        # input_names = list(filter(self.engine.binding_is_input, names)) 
        input_names = ['input']
        self._input_names = input_names 
        self._output_names = output_names 
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 
 
    def forward(self, inputs: Dict[str, torch.Tensor]): 
        assert self._input_names is not None 
        assert self._output_names is not None 
        # 分配输入输出缓冲区。
        bindings = [None] * (len(self._input_names) + len(self._output_names)) 
        profile_id = 0 
        for input_name, input_tensor in inputs.items(): 
            # check if input shape is valid 
            profile = self.engine.get_profile_shape(profile_id, input_name) 
            assert input_tensor.dim() == len(profile[0]), 'Input dim is different from engine profile.' 
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]): 
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.' 
            # idx = self.engine.get_binding_index(input_name) 
            idx = 0
 
            # All input tensors must be gpu variables 
            assert 'cuda' in input_tensor.device.type 
            input_tensor = input_tensor.contiguous() 
            if input_tensor.dtype == torch.long: 
                input_tensor = input_tensor.int() 
            self.context.set_input_shape('input', input_tensor.shape) 
            bindings[idx] = input_tensor.contiguous().data_ptr() 
 
        # create output tensors 
        outputs = {} 
        for output_name in self._output_names: 
            # idx = self.engine.get_binding_index(output_name) 
            idx = 1
            dtype = torch.float32 
            # shape = tuple(self.context.get_tensor_shape(['output'])) 
            shape = [1, 3, 112, 112]
            device = torch.device('cuda') 
            output = torch.empty(size=shape, dtype=dtype, device=device) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        # self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream) 
        # 对批同步执行推理。此方法需要一组输入和输出缓冲区。
        self.context.execute_v2(bindings=bindings)
        return outputs 


if __name__ == '__main__': 
    # 加载优化好的TensorRT引擎文件
    model = TRTWrapper('sample.engine', ['output'])  


    # 输入
    input_img = cv2.imread('/home/wyk/bushu/face.png').astype(np.float32) 
    # HWC to NCHW 输入格式转换
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0) 
    # 转换为torch张量，放入GPU
    input_img = torch.from_numpy(input_img).cuda()
    # 裁剪张量，使其符合模型输入
    input_img = input_img[:, :, 26:250, 26:250]


    # 推理
    output = model(dict(input=input_img)) 


    # 输出
    print(output) 
    print(output['output'].shape) 
    # NCHW to HWC 输出格式转换，放入CPU
    ort_output = output['output'].cpu().numpy()
    ort_output = np.squeeze(ort_output, 0) 
    ort_output = np.clip(ort_output, 0, 255) 
    ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8) 
    cv2.imwrite("/home/wyk/bushu/output.png", ort_output)