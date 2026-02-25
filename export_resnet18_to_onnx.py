import torch
import torchvision.models as models
import os

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载预训练的ResNet18模型
print("正在加载预训练的ResNet18模型...")
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()  # 设置为评估模式
print("模型加载完成!")

# 创建一个示例输入张量（ResNet18期望的输入是224x224的RGB图像）
batch_size = 1
input_shape = (3, 224, 224)  # 通道数，高度，宽度
dummy_input = torch.randn(batch_size, *input_shape, device=device)

# 定义输出ONNX文件路径
onnx_file_path = "resnet18.onnx"

print(f"正在将模型导出为ONNX格式: {onnx_file_path}")

# 导出模型为ONNX格式
torch.onnx.export(
    model,                          # 要导出的模型
    dummy_input,                    # 示例输入
    onnx_file_path,                 # 输出文件路径
    export_params=True,             # 存储训练好的权重
    opset_version=11,               # ONNX版本
    do_constant_folding=True,       # 是否执行常量折叠优化
    input_names=['input'],          # 输入名称
    output_names=['output'],        # 输出名称
    dynamic_axes={                  # 动态维度
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"模型已成功导出为ONNX格式: {onnx_file_path}")

# 验证导出的ONNX文件是否存在
if os.path.exists(onnx_file_path):
    file_size = os.path.getsize(onnx_file_path) / (1024 * 1024)  # 转换为MB
    print(f"ONNX文件大小: {file_size:.2f} MB")
else:
    print("警告: 导出的ONNX文件不存在!")

# 可选：使用ONNX Runtime验证导出的模型
print("\n验证导出的ONNX模型...")
try:
    import onnxruntime as ort
    import numpy as np
    
    # 创建ONNX Runtime推理会话
    ort_session = ort.InferenceSession(onnx_file_path)
    
    # 将PyTorch张量转换为NumPy数组
    dummy_input_np = dummy_input.cpu().numpy()
    
    # 使用ONNX Runtime进行推理
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 使用PyTorch进行推理以比较结果
    with torch.no_grad():
        torch_outputs = model(dummy_input).cpu().numpy()
    
    # 计算结果差异
    np.testing.assert_allclose(torch_outputs, ort_outputs[0], rtol=1e-03, atol=1e-05)
    print("✓ ONNX模型验证成功! 推理结果与PyTorch一致。")
    
    # 显示一些输出信息
    print(f"\nPyTorch输出形状: {torch_outputs.shape}")
    print(f"ONNX Runtime输出形状: {ort_outputs[0].shape}")
    print(f"\nPyTorch Top-5预测索引: {np.argsort(torch_outputs[0])[-5:][::-1]}")
    print(f"ONNX Top-5预测索引: {np.argsort(ort_outputs[0][0])[-5:][::-1]}")
    
    # 计算并显示平均绝对误差
    mae = np.mean(np.abs(torch_outputs - ort_outputs[0]))
    print(f"\n平均绝对误差: {mae:.10f}")
    
    # 计算并显示最大绝对误差
    max_abs_error = np.max(np.abs(torch_outputs - ort_outputs[0]))
    print(f"最大绝对误差: {max_abs_error:.10f}")
    
    # 计算并显示误差的标准差
    std_error = np.std(np.abs(torch_outputs - ort_outputs[0]))
    print(f"误差标准差: {std_error:.10f}")
    
except ImportError:
    print("⚠️  onnxruntime 未安装，跳过验证步骤。可以使用 'pip install onnxruntime' 安装。")
except Exception as e:
    print(f"❌ 验证过程中出现错误: {str(e)}")

print("\n导出和验证过程完成!")