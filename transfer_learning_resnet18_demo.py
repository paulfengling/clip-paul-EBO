import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image

# 定义一个简单的自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 假设数据集在两个子目录中：'class1' 和 'class2'
        for label, class_name in enumerate(['ants', 'bees']):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                self.images.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载自定义数据集
root_dir = './hymenoptera_data/train'
train_dataset = CustomDataset(root_dir=root_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features


# 修改最后的全连接层以适应二分类任务
model.fc = nn.Linear(num_features, 2)

# 使用GPU（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

print("训练完成")

torch.save(model, 'resnet18_ant_beel_model.pth')
print("Model saved to resnet18_ant_beel_model.pth")

model.eval()  # 设置为评估模式
print("模型加载完成!")

# 创建一个示例输入张量（ResNet18期望的输入是224x224的RGB图像）
batch_size = 1
input_shape = (3, 224, 224)  # 通道数，高度，宽度
dummy_input = torch.randn(batch_size, *input_shape, device=device)

# 定义输出ONNX文件路径
onnx_file_path = "resnet18_ant_bee_model.onnx"

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
    # dynamic_axes={                  # 动态维度
    #     'input': {0: 'batch_size'},
    #     'output': {0: 'batch_size'}
    # }
)
