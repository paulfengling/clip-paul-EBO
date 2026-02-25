import os
import json
from PIL import Image

# 设置数据集路径
yolo_data_dir = "D:/camel_elephant_training/data/"  # 修改为YOLO格式的数据集路径
coco_data_dir = "D:/barray_det_ds/"  # 修改你想输出的COCO格式数据集路径

images_path = os.path.join(yolo_data_dir, "images")
labels_path = os.path.join(yolo_data_dir, "labels")

# 支持的图片格式
SUPPORTED_IMG_FORMATS = ('.png', '.jpg', '.jpeg')

# 类别映射
categories = [
    {"id": 0, "name": "elephant", "supercategory": "none"},
    {"id": 1, "name": "camel", "supercategory": "none"}
]


def convert_yolo_to_coco(x_center, y_center, box_width, box_height, img_width, img_height):
    """将YOLO格式的边界框转换为COCO格式"""
    x_min = max(0, (x_center - box_width / 2) * img_width)
    y_min = max(0, (y_center - box_height / 2) * img_height)
    width = min(box_width * img_width, img_width - x_min)
    height = min(box_height * img_height, img_height - y_min)

    return [round(x_min, 2), round(y_min, 2), round(width, 2), round(height, 2)]


def init_coco_format():
    """初始化COCO数据结构"""
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }



def get_label_path(img_name, labels_dir, split):
    """获取标签文件路径，支持多种图片格式"""
    base_name = os.path.splitext(img_name)[0]
    possible_extensions = ['.txt']

    for ext in possible_extensions:
        label_path = os.path.join(labels_dir, split, base_name + ext)
        if os.path.exists(label_path):
            return label_path

    return None


# 处理每个数据集分区
for split in ['train', 'valid', 'test']:
    coco_format = init_coco_format()
    annotation_id = 0

    split_output_dir = os.path.join(coco_data_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    print(f"开始处理 {split} 文件夹")

    split_images_dir = os.path.join(images_path, split)

    # 检查目录是否存在
    if not os.path.exists(split_images_dir):
        print(f"警告: 目录 {split_images_dir} 不存在，跳过")
        continue

    for img_name in os.listdir(split_images_dir):
        if img_name.lower().endswith(SUPPORTED_IMG_FORMATS):
            img_path = os.path.join(split_images_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                image_info = {
                    "file_name": img_name,
                    "id": len(coco_format["images"]),
                    "width": img_width,
                    "height": img_height
                }

                coco_format["images"].append(image_info)

                # 查找对应的标签文件
                label_path = get_label_path(img_name, labels_path, split)

                if label_path and os.path.exists(label_path):
                    with open(label_path, "r") as file:
                        for line_num, line in enumerate(file, 1):
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                parts = line.split()
                                if len(parts) != 5:
                                    print(f"警告: {label_path} 第{line_num}行格式错误，跳过")
                                    continue

                                category_id, x_center, y_center, width, height = map(float, parts)

                                # 验证类别ID
                                if not (0 <= category_id < len(categories)):
                                    print(f"警告: {label_path} 第{line_num}行类别ID {category_id} 超出范围，跳过")
                                    continue

                                bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)

                                # 验证边界框
                                if bbox[2] <= 0 or bbox[3] <= 0:
                                    print(f"警告: {label_path} 第{line_num}行边界框尺寸无效，跳过")
                                    continue

                                annotation = {
                                    "id": annotation_id,
                                    "image_id": image_info["id"],
                                    "category_id": int(category_id),
                                    "bbox": bbox,
                                    "area": bbox[2] * bbox[3],
                                    "iscrowd": 0
                                }

                                coco_format["annotations"].append(annotation)
                                annotation_id += 1

                            except ValueError as e:
                                print(f"错误: 解析 {label_path} 第{line_num}行时出错: {e}，跳过该行")
                                continue

            except Exception as e:
                print(f"错误: 处理图片 {img_path} 时出错: {e}，跳过该图片")
                continue

    # 为每个分区保存JSON文件
    output_json_path = os.path.join(split_output_dir, "_annotations.coco.json")
    if 'info' not in coco_format:
        coco_format['info'] = {
            "description": "COCO Dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2023/01/01"
        }

    if 'licenses' not in coco_format:
        coco_format['licenses'] = []
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"{split} 分区完成: 共 {len(coco_format['images'])} 张图片, {len(coco_format['annotations'])} 个标注")

print(f"转换完成！COCO格式数据集已保存到: {coco_data_dir}")