import cv2 as cv
import numpy as np
import openvino as ov
import time

# from rfdetr import RFDETRNano
# model = RFDETRNano(pretrain_weights="rf-detr-nano.pth")
# model.export()

def format_rfdetr(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def read_all_lines(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"文件 '{filename}' 未找到。")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []
class_list = read_all_lines("D:/python/yolov5-7.0/classes.txt")

def redetr_infer_demo():
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    core = ov.Core()
    for device in core.available_devices:
        print(device)

    # Read IR
    # model = core.read_model()
    onnxpath="./rf-detr-nano.onnx"
    compiled_model = core.compile_model(model=onnxpath, device_name="GPU")
    output_layer1 = compiled_model.output(0)
    output_layer2 = compiled_model.output(1)
    print("out1 name", output_layer1.names)
    print("out2 name", output_layer2.names)
    frame = cv.imread("D:/images/frame_deeppose.jpg")
    bgr = format_rfdetr(frame)
    img_h, img_w, img_c = bgr.shape

    start = time.time()
    image = cv.dnn.blobFromImage(bgr, 1 / 255.0, (384, 384), swapRB=True, crop=False)

    outs = compiled_model([image])
    res = outs[output_layer1] # 1xNx4
    out2 = outs[output_layer2]  # 1xNx4
    rows = np.squeeze(res, 0)
    labels = np.squeeze(out2, 0)
    print(rows.shape, labels.shape)
    x_factor = img_w / 384
    y_factor = img_h / 384

    for r in range(rows.shape[0]):
        row = rows[r]
        classes_scores = labels[r]
        class_id = np.argmax(classes_scores)
        conf = classes_scores[class_id]
        if conf>0.5:
            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * 384) * x_factor
            top = int((y - 0.5 * h) * 384) * x_factor
            width = int(w * 384) * x_factor
            height = int(h * 384) * x_factor
            box = [int(left), int(top), int(width), int(height)]
            # color = colors[class_id % len(colors)]
            cv.rectangle(frame, box, (0, 0, 255), 2)
            cv.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 0, 255), -1)
            # cv.putText(frame, class_list[class_id] + (" %.2f"%conf), (box[0], box[1] - 7), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    end = time.time()
    inf_end = end - start
    fps = 1 / inf_end
    fps_label = "FPS: %.2f" % fps
    cv.putText(frame, fps_label, (20, 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("RFDETR Object Detection + OpenVINNO2025.1", frame)
    cc = cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    redetr_infer_demo()