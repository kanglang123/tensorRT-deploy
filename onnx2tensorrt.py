import torch 
import onnx 
import tensorrt as trt 
 
onnx_model = 'model.onnx' 
 
class NaiveModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.pool = torch.nn.MaxPool2d(2, 2) 
 
    def forward(self, x): 
        return self.pool(x) 
 
device = torch.device('cuda:0') 
 
# generate ONNX model 
torch.onnx.export(NaiveModel(), 
                    torch.randn(1, 3, 224, 224), 
                    onnx_model, 
                    input_names=['input'], 
                    output_names=['output'], 
                    opset_version=11) 
onnx_model = onnx.load(onnx_model) 
 
# create builder and network 
logger = trt.Logger(trt.Logger.ERROR) # 创建日志记录器
builder = trt.Builder(logger) # 创建构建器
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
network = builder.create_network(EXPLICIT_BATCH) 
 
# parse onnx 
parser = trt.OnnxParser(network, logger) 
 
if not parser.parse(onnx_model.SerializeToString()): 
    error_msgs = '' 
    for error in range(parser.num_errors): 
        error_msgs += f'{parser.get_error(error)}\n' 
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
config = builder.create_builder_config() 
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
profile = builder.create_optimization_profile() 
profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224]) 
config.add_optimization_profile(profile) 

serialized_engine = builder.build_serialized_network(network, config)
with open('sample.engine', 'wb') as f:
    f.write(serialized_engine)
    print("generating file done!") 

# # config.max_workspace_size = 1<<20 
# profile = builder.create_optimization_profile() 
 
# profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224]) 
# config.add_optimization_profile(profile) 

# # create engine 
# with torch.cuda.device(device): 
#     engine = builder.build_engine(network, config) 
 
# with open('model.engine', mode='wb') as f: 
#     f.write(bytearray(engine.serialize())) 
#     print("generating file done!") 

# # create runtime 去序列化引擎
# runtime = trt.Runtime(logger)
# engine = runtime.deserialize_cuda_engine(serialized_engine)

# # create context 
# context = engine.create_execution_context()
# # create buffer
# h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
# h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
# d_input = cuda.mem_alloc(h_input.nbytes)
# d_output = cuda.mem_alloc(h_output.nbytes)
# bindings = [int(d_input), int(d_output)]
# stream = cuda.Stream()

# # inference
# h_input[...] = np.random.random(h_input.shape)
# cuda.memcpy_htod_async(d_input, h_input, stream)
# context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# cuda.memcpy_dtoh_async(h_output, d_output, stream)
# stream.synchronize()
