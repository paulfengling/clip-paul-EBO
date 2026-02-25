import cv2
import numpy as np
import openvino as ov

# 加载ONNX格式的SWIN Tiny模型
core = ov.Core()
compiled_model = core.compile_model("swin_t_ant_bee.onnx", "CPU")

# 获取输入和输出层
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 定义图像预处理函数
def preprocess_image(image_path):
    # 打开图像
    image = cv2.imread(image_path)
    cv2.imshow("input", image)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # 调整图像大小为224x224
    input_size = (224, 224)
    image = cv2.resize(image, input_size)

    # 转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 归一化图像
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = np.array(image) / 255.0
    image -= mean
    image /= std

    # 调整维度为[1, C, H, W]
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image


# 进行推理
def run_model(input_data):
    results = compiled_model(input_data)[output_layer]
    class_index = np.argmax(results) # top-1
    return class_index

# 主函数
if __name__ == '__main__':
    # # 加载预训练的Swin Tiny模型
    from torchvision import  models
    model = models.swin_t(pretrained=True)
    print(model)

    image_path = './ant_bees/ant2.jpg'  # 替换为你的图像路径
    image = cv2.imread(image_path)
    # 预处理图像
    input_data = preprocess_image(image_path)

    # 进行推理
    class_index = run_model(input_data)
    lines = ['ant', 'bee']

    print("Predicted class: ", lines[class_index])
    cv2.putText(image, lines[class_index], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Swin@gloomyfish+OpenVINO2025", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

