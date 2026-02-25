import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化，使用MNIST数据集的均值和标准差
])

# 加载MNIST数据集
def load_data(batch_size=64):
    print("正在加载MNIST数据集...")
    # 训练数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 测试数据集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"数据集加载完成: 训练集{len(train_dataset)}张图像, 测试集{len(test_dataset)}张图像")
    return train_loader, test_loader

# 定义卷积神经网络模型
class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        # 第一个卷积层：输入通道1，输出通道16，卷积核大小3x3，步长1， padding=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # 第二个卷积层：输入通道16，输出通道32，卷积核大小3x3，步长1， padding=1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 最大池化层：池化核大小2x2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 三个全连接层
        # 卷积后的特征图大小: (28 -> 14 -> 7) 经过两次池化
        # 输入特征数: 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # 第一个卷积层 + ReLU激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积层 + ReLU激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 32 * 7 * 7)
        # 第一个全连接层 + ReLU激活
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 第二个全连接层 + ReLU激活
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 第三个全连接层（输出层）
        x = self.fc3(x)
        return x

# 训练模型
def train_model(model, train_loader, epochs=5, learning_rate=0.001):
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 确保模型在训练模式
    model.train()
    
    train_losses = []
    train_accuracies = []
    
    print(f"开始训练模型，共{epochs}个epoch")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计损失
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 每100个batch打印一次训练状态
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '\
                      f'Loss: {running_loss / (i + 1):.4f}, Accuracy: {100 * correct / total:.2f}%')
        
        # 记录每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch + 1}/{epochs}] 完成 - 平均损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
    
    print("训练完成！")
    return train_losses, train_accuracies

# 测试模型
def test_model(model, test_loader):
    # 设置模型为评估模式
    model.eval()
    
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for images, labels in test_loader:
            # 前向传播
            outputs = model(images)
            
            # 预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算准确率
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy

# 保存模型
def save_model(model, filepath='mnist_cnn_model.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存到 {filepath}")

# 可视化训练结果
def visualize_training(train_losses, train_accuracies, test_accuracy):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'r-', marker='o')
    plt.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}%')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 可视化预测结果
def visualize_predictions(model, test_loader):
    model.eval()
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # 显示一些测试图像及其预测结果
    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i + 1)
        # 转换为numpy数组并恢复原始形状
        img = images[i].numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {predictions[i]}\nActual: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 设置训练参数
    batch_size = 64
    epochs = 5
    learning_rate = 0.001
    model_save_path = 'mnist_cnn_model.pth'
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size)
    
    # 创建模型实例
    print("创建卷积神经网络模型...")
    model = CNNMNIST()
    print(model)
    
    # 训练模型
    train_losses, train_accuracies = train_model(model, train_loader, epochs, learning_rate)
    
    # 测试模型
    test_accuracy = test_model(model, test_loader)
    
    # 保存模型
    save_model(model, model_save_path)
    
    # 可视化训练结果
    print("显示训练结果图表...")
    visualize_training(train_losses, train_accuracies, test_accuracy)
    
    # 可视化预测结果
    print("显示预测结果示例...")
    visualize_predictions(model, test_loader)

if __name__ == "__main__":
    main()