import os
import time
import base64
import random
import logging
import numpy as np
from io import BytesIO
import onnxruntime as ort
from PIL import Image, ImageDraw
from crop_image import crop_image, convert_png_to_jpg,draw_points_on_image,bytes_to_pil,validate_path,save_path
logger = logging.getLogger(__name__)
def safe_load_img(image):
    im_pil = None
    try:
        if isinstance(image, Image.Image):
            im_pil = image
        elif isinstance(image, str):
            try:
                im_pil = Image.open(image)
            except (IOError, FileNotFoundError):
                if ',' in image:
                    image = image.split(',')[-1]
                padding = len(image) % 4
                if padding > 0:
                    image += '=' * (4 - padding)
                img_bytes = base64.b64decode(image)
                im_pil = Image.open(io.BytesIO(img_bytes))
        elif isinstance(image, bytes):
            im_pil = bytes_to_pil(image)
        elif isinstance(image, np.ndarray):
            im_pil = Image.fromarray(image)
        else:
            raise ValueError(f"不支持的输入类型: {type(image)}")
        return im_pil.convert("RGB")
    except Exception as e:
        raise ValueError(f"无法加载或解析图像，错误: {e}")
    
def predict(icon_image, bg_image):
    import torch
    from train import MyResNet18, data_transform
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', 'resnet18_38_0.021147585306924.pth')
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    target_images = []
    target_images.append(data_transform(Image.open(BytesIO(icon_image))))

    bg_images = crop_image(bg_image, coordinates)
    for bg_image in bg_images:
        target_images.append(data_transform(bg_image))

    start = time.time()
    model = MyResNet18(num_classes=91)  # 这里的类别数要与训练时一致
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("加载模型，耗时:", time.time() - start)
    start = time.time()

    target_images = torch.stack(target_images, dim=0)
    target_outputs = model(target_images)

    scores = []

    for i, out_put in enumerate(target_outputs):
        if i == 0:
            # 增加维度，以便于计算
            target_output = out_put.unsqueeze(0)
        else:
            similarity = torch.nn.functional.cosine_similarity(
                target_output, out_put.unsqueeze(0)
            )
            scores.append(similarity.cpu().item())
    # 从左到右，从上到下，依次为每张图片的置信度
    logger.info(scores)
    # 对数组进行排序，保持下标
    indexed_arr = list(enumerate(scores))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # 提取最大三个数及其下标
    largest_three = sorted_arr[:3]
    logger.info(largest_three)
    logger.info("识别完成，耗时:", time.time() - start)

def load_model(name='PP-HGNetV2-B4.onnx'):
    # 加载onnx模型
    global session,input_name
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    logger.info(f"加载{name}模型，耗时:{time.time() - start}")

def load_dfine_model(name='d-fine-n.onnx'):
    # 加载onnx模型
    global session_dfine
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session_dfine = ort.InferenceSession(model_path)
    logger.info(f"加载{name}模型，耗时:{time.time() - start}")

def load_yolo11n(name='yolo11n.onnx'):
    global session_yolo11n
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session_yolo11n = ort.InferenceSession(model_path)
    logger.info(f"加载{name}模型，耗时:{time.time() - start}")

def load_dinov3(name='dinov3-small.onnx'):
    global session_dino3
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session_dino3 = ort.InferenceSession(model_path)
    logger.info(f"加载{name}模型，耗时:{time.time() - start}")

def load_dino_classify(name='atten.onnx'):
    global session_dino_cf
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    session_dino_cf = ort.InferenceSession(model_path)
    logger.info(f"加载{name}模型，耗时:{time.time() - start}")

def predict_onnx(icon_image, bg_image, point = None):
    import cv2
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]

    def cosine_similarity(vec1, vec2):
        # 将输入转换为 NumPy 数组
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        # 计算向量的范数
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        # 计算余弦相似度
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity

    def data_transforms(image):
        image = image.resize((224, 224))
        image = Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB))
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)
        return image_array

    target_images = []
    target_images.append(data_transforms(Image.open(BytesIO(icon_image))))
    bg_images = crop_image(bg_image, coordinates)

    for one in bg_images:
        target_images.append(data_transforms(one))

    start = time.time()
    outputs = session.run(None, {input_name: target_images})[0]

    scores = []
    for i, out_put in enumerate(outputs):
        if i == 0:
            target_output = out_put
        else:
            similarity = cosine_similarity(target_output, out_put)
            scores.append(similarity)
    logger.debug(f"从左到右，从上到下，依次为每张图片的置信度:\n{scores}")
    # 对数组进行排序，保持下标
    indexed_arr = list(enumerate(scores))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    # 提取最大三个数及其下标
    if point == None:
        largest_three = sorted_arr[:3]
        answer = [coordinates[i[0]] for i in largest_three]
    # 基于分数判断
    else:
        answer = [one[0] for one in sorted_arr if one[1] > point]
    logger.info(f"识别完成{answer}，耗时: {time.time() - start}")
    #draw_points_on_image(bg_image, answer)
    return answer

def predict_onnx_pdl(images_path):
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    def data_transforms(path):
        # 打开图片
        img = Image.open(path)
        # 调整图片大小为232x224（假设最短边长度调整为232像素）
        if img.width < img.height:
            new_size = (232, int(232 * img.height / img.width))
        else:
            new_size = (int(232 * img.width / img.height), 232)
        resized_img = img.resize(new_size, Image.BICUBIC)
        # 裁剪图片为224x224
        cropped_img = resized_img.crop((0, 0, 224, 224))
        # 将图像转换为NumPy数组并进行归一化处理
        img_array = np.array(cropped_img).astype(np.float32)
        img_array /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_array -= np.array(mean)
        img_array /= np.array(std)
        # 将通道维度移到前面
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array
    images = []
    for pic in sorted(os.listdir(images_path)):
        if "cropped" not in pic:
            continue
        image_path = os.path.join(images_path,pic)
        images.append(data_transforms(image_path))
    if len(images) == 0:
        raise FileNotFoundError(f"先使用切图代码切图至{image_path}再推理,图片命名如cropped_9.jpg,从0到9共十个,最后一个是检测目标")
    start = time.time()
    outputs = session.run(None, {input_name: images})[0]
    result = [np.argmax(one) for one in outputs]
    target = result[-1]
    answer = [coordinates[index] for index in range(9) if result[index] == target]
    if len(answer) == 0:
        all_sort =[np.argsort(one) for one in outputs]
        answer = [coordinates[index] for index in range(9) if all_sort[index][1] == target]
    logger.info(f"识别完成{answer}，耗时: {time.time() - start}")
    with open(os.path.join(images_path,"nine.jpg"),'rb') as f:
        bg_image = f.read()
    #draw_points_on_image(bg_image, answer)
    return answer

# d-fine的推理代码及函数
def calculate_iou(boxA, boxB):
    """
    使用 NumPy 计算两个边界框的交并比 (IoU)。
    """
    # 确定相交矩形的坐标
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    # 计算相交区域的面积
    intersection_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # 计算两个边界框的面积
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 计算并集面积
    union_area = float(boxA_area + boxB_area - intersection_area)

    # 计算 IoU
    if union_area == 0:
        return 0.0
        
    iou = intersection_area / union_area
    return iou

def non_maximum_suppression(detections, iou_threshold=0.35):
    """
    对检测结果执行非极大值抑制 (NMS)。

    参数:
    detections -- 一个列表，其中每个元素是包含 'box', 'score' 的字典。
                  例如: [{'box': [x1, y1, x2, y2], 'score': 0.9, ...}, ...]
    iou_threshold -- 一个浮点数，用于判断框是否重叠的 IoU 阈值。

    返回:
    final_detections -- 经过 NMS 处理后保留下来的检测结果列表。
    """
    # 1. 检查检测结果是否为空
    if not detections:
        return []

    # 2. 按置信度（score）从高到低对边界框进行排序
    #    我们使用 lambda 函数来指定排序的键
    detections.sort(key=lambda x: x['score'], reverse=True)

    final_detections = []
    
    # 3. 循环处理，直到没有检测结果为止
    while detections:
        # 4. 将当前得分最高的检测结果（第一个）添加到最终列表中
        #    并将其从原始列表中移除
        best_detection = detections.pop(0)
        final_detections.append(best_detection)

        # 5. 计算刚刚取出的最佳框与剩余所有框的 IoU
        #    并只保留那些 IoU 小于阈值的框
        detections_to_keep = []
        for det in detections:
            # 假设相同类别的才进行NMS
            iou = calculate_iou(best_detection['box'], det['box'])
            if iou < iou_threshold:
                detections_to_keep.append(det)
        
        # 用筛选后的列表替换原始列表，进行下一轮迭代
        detections = detections_to_keep

    return final_detections

def predict_onnx_dfine(image,draw_result=False):
    input_nodes = session_dfine.get_inputs()
    output_nodes = session_dfine.get_outputs()
    image_input_name = input_nodes[0].name
    size_input_name = input_nodes[1].name
    output_names = [node.name for node in output_nodes]
    im_pil = safe_load_img(image)
    w, h = im_pil.size
    orig_size_np = np.array([[w, h]], dtype=np.int64)
    im_resized = im_pil.resize((320, 320), Image.Resampling.BILINEAR)
    im_data = np.array(im_resized, dtype=np.float32) / 255.0
    im_data = im_data.transpose(2, 0, 1)
    im_data = np.expand_dims(im_data, axis=0)
    inputs = {
        image_input_name: im_data,
        size_input_name: orig_size_np
    }
    outputs = session_dfine.run(output_names, inputs)
    output_map = {name: data for name, data in zip(output_names, outputs)}
    labels = output_map['labels'][0]
    boxes = output_map['boxes'][0]
    scores = output_map['scores'][0]

    colors = ["red", "blue", "green", "yellow", "white", "purple", "orange"]
    mask = scores > 0.4
    filtered_labels = labels[mask]
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    rebuild_color = {}
    unique_labels = list(set(filtered_labels))
    for i, l_val in enumerate(unique_labels):
        class_id = int(l_val)
        if class_id not in rebuild_color:
            rebuild_color[class_id] = colors[i % len(colors)]
    result = {k: [] for k in unique_labels}
    for i, box in enumerate(filtered_boxes):
        if box[2]>160 and box[3] < 45:
            continue
        label_val = filtered_labels[i]
        class_id = int(label_val)
        color = rebuild_color[class_id]
        score = filtered_scores[i]
        
        result[class_id].append({
            'box': box,
            'label_val': label_val,
            'score': score
        })
    keep_result = {}
    result_points = []
    for class_id in result:
        tp = non_maximum_suppression(result[class_id],0.01)
        if len(tp) < 2:
            continue
        point = tp[0]["score"]+tp[1]["score"]
        if point < 0.85:
            continue
        keep_result.update({class_id:tp[0:2]})
        result_points.append({"id":class_id,"point":point})
    result_points.sort(key=lambda item: item['point'], reverse=True)
    if len(keep_result) > 3:
        tp = {}
        for one in result_points[0:3]:
            tp.update({one['id']:keep_result[one['id']]})
        keep_result = tp
    for class_id in keep_result:
        keep_result[class_id].sort(key=lambda item: item['box'][3], reverse=True)
    sorted_result = {}
    sorted_class_ids = sorted(keep_result.keys(), key=lambda cid: keep_result[cid][0]['box'][0])
    for class_id in sorted_class_ids:
        sorted_result[class_id] = keep_result[class_id]
    points = []

    if draw_result:
        draw = ImageDraw.Draw(im_pil)
    for c1,class_id in enumerate(sorted_result):
        items = sorted_result[class_id]
        last_item = items[-1]
        center_x = (last_item['box'][0] + last_item['box'][2]) / 2
        center_y = (last_item['box'][1] + last_item['box'][3]) / 2
        text_position_center = (center_x , center_y)
        points.append(text_position_center)
        if draw_result:
            color = rebuild_color[class_id]
            draw.point((center_x, center_y), fill=color)
            text_center = f"{c1}"
            draw.text(text_position_center, text_center, fill=color)
            for c2,item in enumerate(items):
                box = item['box']
                score = item['score']
                
                draw.rectangle(list(box), outline=color, width=1)
                text = f"{class_id}_{c1}-{c2}: {score:.2f}"
                text_position = (box[0] + 2, box[1] - 12 if box[1] > 12 else box[1] + 2)
                draw.text(text_position, text, fill=color)
    if draw_result:
       save_path_temp = os.path.join(validate_path,"icon_result.jpg")
       im_pil.save(save_path_temp)
       logger.info(f"图片可视化结果暂时保存在{save_path_temp},运行完成后移至{save_path}")
    logger.info(f"图片顺序的中心点{points}")
    return points

# yolo的推理代码及函数
def predict_onnx_yolo(image):
    def filter_Detections(results, thresh = 0.5):
        results = results[0]
        results = results.transpose()
        # if model is trained on 1 class only
        if len(results[0]) == 5:
            # filter out the detections with confidence > thresh
            considerable_detections = [detection for detection in results if detection[4] > thresh]
            considerable_detections = np.array(considerable_detections)
            return considerable_detections

        # if model is trained on multiple classes
        else:
            A = []
            for detection in results:

                class_id = detection[4:].argmax()
                confidence_score = detection[4:].max()

                new_detection = np.append(detection[:4],[class_id,confidence_score])

                A.append(new_detection)

            A = np.array(A)

            # filter out the detections with confidence > thresh
            considerable_detections = [detection for detection in A if detection[-1] > thresh]
            considerable_detections = np.array(considerable_detections)

            return considerable_detections
    def NMS(boxes, conf_scores, iou_thresh = 0.55):

        #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        areas = (x2-x1)*(y2-y1)

        order = conf_scores.argsort()

        keep = []
        keep_confidences = []

        while len(order) > 0:
            idx = order[-1]
            A = boxes[idx]
            conf = conf_scores[idx]

            order = order[:-1]

            xx1 = np.take(x1, indices= order)
            yy1 = np.take(y1, indices= order)
            xx2 = np.take(x2, indices= order)
            yy2 = np.take(y2, indices= order)

            keep.append(A)
            keep_confidences.append(conf)

            # iou = inter/union

            xx1 = np.maximum(x1[idx], xx1)
            yy1 = np.maximum(y1[idx], yy1)
            xx2 = np.minimum(x2[idx], xx2)
            yy2 = np.minimum(y2[idx], yy2)

            w = np.maximum(xx2-xx1, 0)
            h = np.maximum(yy2-yy1, 0)

            intersection = w*h

            # union = areaA + other_areas - intesection
            other_areas = np.take(areas, indices= order)
            union = areas[idx] + other_areas - intersection

            iou = intersection/union

            boleans = iou < iou_thresh

            order = order[boleans]

            # order = [2,0,1]  boleans = [True, False, True]
            # order = [2,1]

        return keep, keep_confidences
    def rescale_back(results,img_w,img_h,imgsz=384):
        cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:, 4], results[:,-1]
        cx = cx/imgsz * img_w
        cy = cy/imgsz * img_h
        w = w/imgsz * img_w
        h = h/imgsz * img_h
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        boxes = np.column_stack((x1, y1, x2, y2, class_id))
        keep, keep_confidences = NMS(boxes,confidence)
        return keep, keep_confidences
    im_pil = safe_load_img(image)
    im_resized = im_pil.resize((384, 384), Image.Resampling.BILINEAR)
    im_data = np.array(im_resized, dtype=np.float32) / 255.0
    im_data = im_data.transpose(2, 0, 1)
    im_data = np.expand_dims(im_data, axis=0)
    res = session_yolo11n.run(None,{"images":im_data})
    results = filter_Detections(res)
    rescaled_results, confidences = rescale_back(results,im_pil.size[0],im_pil.size[1])
    images = {"top":[],"bottom":[]}
    for r, conf in zip(rescaled_results, confidences):
        x1,y1,x2,y2, cls_id = r
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = im_pil.crop((x1, y1, x2, y2))
        if cls_id == 0:
            images['top'].append({"image":cropped_image,"bbox":[x1, y1, x2, y2]})
        else:
            images['bottom'].append({"image":cropped_image,"bbox":[x1, y1, x2, y2]})
    return images

# dinov3的推理代码及函数
def make_lvd_transform(resize_size: int = 224):
    """
    返回一个图像预处理函数，功能与PyTorch版本相同
    """
    # 定义标准化参数
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def transform(image) -> np.ndarray:
        """
        图像预处理转换
        
        Args:
            image: PIL Image 或 numpy array (H,W,C) 范围[0,255]
            
        Returns:
            numpy array (C,H,W) 标准化后的float32数组
        """
        # 确保输入是PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # 1. 调整大小 (使用LANCZOS抗锯齿，对应antialias=True)
        image = image.resize((resize_size, resize_size), Image.LANCZOS)
        
        # 2. 转换为numpy数组并调整数据类型和范围
        # PIL图像转换为numpy数组 (H,W,C) 范围[0,255]
        image_array = np.array(image, dtype=np.float32)
        
        # 如果图像是RGBA，只取RGB通道
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        # 缩放到[0,1]范围 (对应scale=True)
        image_array /= 255.0
        
        # 3. 标准化
        # 注意：PyTorch的Normalize是逐通道进行的
        image_array = (image_array - mean) / std
        
        # 4. 转换维度从 (H,W,C) 到 (C,H,W) - 与PyTorch张量布局一致
        image_array = np.transpose(image_array, (2, 0, 1))
        
        return image_array
    
    return transform
transform = make_lvd_transform(224)

def predict_onnx_dino(image):
    im_pil = safe_load_img(image)
    input_name_model = session_dino3.get_inputs()[0].name
    output_name_model = session_dino3.get_outputs()[0].name
    return session_dino3.run([output_name_model], 
                                        {input_name_model: 
                                         np.expand_dims(transform(im_pil), axis=0).astype(np.float32)
                                        }
                                     )[0]

# dinov3结果分类的推理代码及函数
def predict_dino_classify(tokens1,tokens2):
    patch_tokens1 = tokens1[:, 5:, :]
    patch_tokens2 = tokens2[:, 5:, :]
    input_name_model =session_dino_cf.get_inputs()[0].name
    output_name_model =session_dino_cf.get_outputs()[0].name
    emb1 = session_dino_cf.run([output_name_model], {input_name_model: patch_tokens1})[0]
    emb2 = session_dino_cf.run([output_name_model], {input_name_model: patch_tokens2})[0]
    emb1_flat = emb1.flatten()
    emb2_flat = emb2.flatten()
    return float(np.dot(emb1_flat, emb2_flat) / (np.linalg.norm(emb1_flat) * np.linalg.norm(emb2_flat)))

def predict_dino_classify_pipeline(image,draw_result=False):
    im_pil = safe_load_img(image)
    if draw_result:
        draw = ImageDraw.Draw(im_pil)
    crops = predict_onnx_yolo(im_pil)
    features = {}
    for k in crops:
        features.update({k:[]})
        for v in crops[k]:
            features[k].append({"feature":predict_onnx_dino(v['image']),"bbox":v['bbox']})
    features["bottom"] = sorted(features["bottom"], key=lambda x: x["bbox"][0])
    used_indices = set()
    sequence = []

    for target in features['bottom']:
        available = [(idx, opt) for idx, opt in enumerate(features['top']) if idx not in used_indices]

        if not available:
            break

        if len(available) == 1:
            best_idx, best_opt = available[0]
        else:
            best_idx, best_opt = max(
                available,
                key=lambda item: predict_dino_classify(target['feature'], item[1]['feature'])
            )

        sequence.append(best_opt['bbox'])
        used_indices.add(best_idx)
    colors = ["red", "blue", "green", "yellow", "white", "purple", "orange"]
    points = []
    for id,one in enumerate(sequence):
        center_x = (one[0] + one[2]) / 2
        center_y = (one[1] + one[3]) / 2
        w = abs(one[0] - one[2])
        y = abs(one[1] - one[3])
        points.append((center_x+random.randint(int(-w/5),int(w/5)),
                       center_y+random.randint(int(-y/5),int(y/5))
                       ))
        if draw_result:
            draw.rectangle(one, outline=colors[id], width=1)
            text = f"{id+1}"
            text_position = (center_x, center_y)
            draw.text(text_position, text, fill='white')
    if draw_result:
        save_path_temp = os.path.join(validate_path,"icon_result.jpg")
        im_pil.save(save_path_temp)
        logger.info(f"图片可视化结果暂时保存在{save_path_temp},运行完成后移至{save_path}")
    return points


logger.info(f"使用推理设备: {ort.get_device()}")
def use_pdl():
    load_model()

def use_dfine():
    load_dfine_model()

def use_multi():
    load_yolo11n()
    load_dinov3()
    load_dino_classify()

model_for = [
    {"loader":use_pdl,
     "include":["session"],
     "support":['paddle','pdl','nine','原神','genshin']
     },
    {"loader":use_dfine,
     "include":["session_dfine"],
     "support":['dfine','click','memo','note','‌便笺‌']
     },
    {"loader":use_multi,
     "include":['session_yolo11n', 'session_dino3', 'session_dino_cf'],
     "support":['multi','dino','click2','星穹铁道','崩铁','绝区零','zzz','hkrpg']
     }
]

def get_models():
    res = ["以下是当前加载的模型及其对应关键字"]
    for key,value in globals().items():
        if key.startswith("session") and value is not None:
            for one in model_for:
                if key in one['include']:
                    res.append(f" -{key},关键词:{one['support']}")
    return res
def get_available_models():
    res = ["以下是所有可用模型及其对应关键字"]
    for one in model_for:
        res.append(f" -{one['include']}关键词:{one['support']}")
    return res

def load_by(name):
    for one in model_for:
        if name in one['support'] or name in one['include']:
            one['loader']()
            return get_models()
    logger.error(f"不支持的名称，可以使用‌便笺‌、原神、崩铁、绝区零表示")
    
def unload(*names, safe_mode=True):
    import gc
    protected_vars = {'__name__', '__file__', '__builtins__', 
                     'unload'}
    for name in names:
        if name in globals():
            if safe_mode and name in protected_vars:
                logger.error(f"警告: 跳过保护变量 '{name}'")
                continue
            if not name.startswith('session'):
                logger.info("删除的不是模型！")
            var = globals()[name]
            if hasattr(var, 'close'):
                try:
                    var.close()
                except:
                    pass
            globals()[name] = None
            logger.info(f"已释放变量: {name}")
    collected = gc.collect()
    logger.info(f"垃圾回收器清理了 {collected} 个对象")
    return get_models()

if int(os.environ.get("use_pdl",1)):
    use_pdl()
if int(os.environ.get("use_dfine",1)):
    use_dfine()
if int(os.environ.get("use_multi",1)):
    use_multi()

if __name__ == "__main__":
    # 使用resnet18.onnx
    # load_model("resnet18.onnx")
    # icon_path = "img_2_val/cropped_9.jpg"
    # bg_path = "img_2_val/nine.jpg"
    # with open(icon_path, "rb") as rb:
    #     if icon_path.endswith('.png'):
    #         icon_image = convert_png_to_jpg(rb.read())
    #     else:
    #         icon_image = rb.read()
    # with open(bg_path, "rb") as rb:
    #     bg_image = rb.read()
    # predict_onnx(icon_image, bg_image)
    
    # 使用PP-HGNetV2-B4.onnx
    #predict_onnx_pdl(r'img_saved\img_fail\7fe559a85bac4c03bc6ea7b2e85325bf')
    print(predict_onnx_dfine(r"f:\项目留档\JPEGImages\8bdee494b00d401aae3f496e76d886fc.jpg",True))
    # use_multi()
    # print(predict_dino_classify_pipeline("0a92e85f89b345279e74deaa9afa9e1c.jpg",True))
    