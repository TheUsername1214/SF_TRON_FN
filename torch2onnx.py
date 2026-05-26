import sys
import os
# 自动获取当前文件的目录，然后找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设脚本在项目子目录中
sys.path.insert(0, project_root)
from SF_TRON_FN.SRC.Config.Config import *
from SF_TRON_FN.SRC.PPO.Actor_Critic import Actor_Critic
from SF_TRON_FN.SRC.Estimator.Estimator import *
from SF_TRON_FN.SRC.Config.TS_Config import *
import torch.onnx
import onnx
state_dim = PPOCfg.CriticParam.state_dim
# 1. 加载或定义PyTorch模型
model1 = Actor_Critic(PPOCfg, EnvCfg, index = 1)
model1.actor.load_state_dict(torch.load("Model/NN_Model/actor1.pth"))
model2 = Estimator(PPOCfg, EnvCfg, index = 1)
model2.Estimator.load_state_dict(torch.load("Model/NN_Model/estimator1_f.pth"))
# 2. 创建示例输入（dummy input）
# 注意：需要与实际推理时的输入形状一致
dummy_input1 = torch.randn(1,33 + PPOCfg.EstimatorParam.output_dim).to("cuda") 
dummy_input2 = torch.randn(1,33 * 10).to("cuda") 
# 3. 导出为ONNX
onnx_path1 = "model1.onnx"
onnx_path2 = "model2.onnx"
torch.onnx.export(
    model1.actor,                   # PyTorch模型
    dummy_input1,             # 模型输入
    onnx_path1,               # 保存路径
    export_params=True,      # 导出训练好的参数
    opset_version=11,        # ONNX算子集版本（常用11或13）
    do_constant_folding=True,# 优化常量
    input_names=['input'],   # 输入节点名称
    output_names=['output'], # 输出节点名称
    dynamic_axes={           # 动态轴（支持可变batch_size）
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

torch.onnx.export(
    model2.Estimator,                   # PyTorch模型
    dummy_input2,             # 模型输入
    onnx_path2,               # 保存路径
    export_params=True,      # 导出训练好的参数
    opset_version=11,        # ONNX算子集版本（常用11或13）
    do_constant_folding=True,# 优化常量
    input_names=['input'],   # 输入节点名称
    output_names=['output'], # 输出节点名称
    dynamic_axes={           # 动态轴（支持可变batch_size）
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"模型已导出")