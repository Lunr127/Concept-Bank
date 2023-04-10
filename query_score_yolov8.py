import os
import csv
import torch
import numpy as np
from keyphrase_vectorizers import KeyphraseCountVectorizer
from collections import defaultdict
from operator import mul
from functools import reduce
from lavis.models import load_model_and_preprocess

# 加载 blip 模型，用以提取特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
                                                                  is_eval=True, device=device)


# 计算两个向量余弦相似度
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 提取文本的特征
def getTextVector(text):
    caption = text
    text_input = txt_processors["eval"](caption)
    sample = {"text_input": [text_input]}
    features_text = model.extract_features(sample, mode="text")
    return features_text.text_embeds_proj[:, 0, :]


# 从 coco_classes.txt 中提取 classes
def extract_classes(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    classes = []
    for line in lines:
        if line.startswith(' '):  # 类名行以空格开头
            class_name = line.split(':')[1].strip()  # 提取类名
            classes.append(class_name)

    return classes


# 从查询语句中提取关键词
def get_keywords_from_query(query):
    query_list = [query]
    vectorizer = KeyphraseCountVectorizer()
    vectorizer.fit(query_list)
    keyphrase = vectorizer.get_feature_names_out()
    return keyphrase


# 初始化csv_data字符串
csv_data = ""

datacsv_path = "/mnt/disk6/fsw/VBS/AVS/concept_bank/yolov8_datacsv_prob"

# 遍历datacsv文件夹下的所有文件
for file_name in os.listdir(datacsv_path):
    # 检查文件是否是CSV文件
    if file_name.endswith('.csv'):
        # 读取CSV文件内容
        with open(os.path.join(datacsv_path, file_name), 'r') as file:
            file_content = file.read()

        # 将文件内容添加到csv_data字符串中
        csv_data += file_content + "/n"

# 将csv_data字符串转换为逐行的数据
csv_lines = csv_data.splitlines()
# 使用csv.reader处理CSV数据
reader = csv.reader(csv_lines)

# 按shotId分组数据
shots = defaultdict(list)
for row in reader:
    if len(row) > 1:
        shot_id, target, score = row
        shots[shot_id].append((target, float(score)))

# 打开查询 topic 文件并读取每一行内容
with open('tv22.avs.topics.txt', 'r') as file:
    lines = file.readlines()

# 创建一个空字典用于存储 queryId-topic 键值对
query_dictionary = {}

# 遍历每一行并将内容分割成 queryId-topic 键值对
for line in lines:
    key, value = line.strip().split('  ')
    query_dictionary[key] = value

result_path = "/mnt/disk6/fsw/VBS/AVS/concept_bank/result/yolov8_result/"
result_prob_path = "/mnt/disk6/fsw/VBS/AVS/concept_bank/result/yolov8_result_prob/"

# 检查 result 文件夹是否存在，如果不存在则创建
if not os.path.exists(result_path):
    os.makedirs(result_path)

# 检查 result_prob 文件夹是否存在，如果不存在则创建
if not os.path.exists(result_prob_path):
    os.makedirs(result_prob_path)

# 遍历字典query_dictionary并获取键值对
for queryId, query_topic in query_dictionary.items():
    # 打印键和对应的值
    print(queryId, ":", query_topic)

    # 查询语句转化为关键词
    keywords = get_keywords_from_query(query_topic)

    # 获取 关键词-向量 键值对
    keyword_vector_map = {}
    for keyword in keywords:
        keyword_vector = getTextVector(keyword).cpu().numpy().tolist()
        keyword_vector_map[keyword] = keyword_vector[0]

    # 从 coco_classes.txt 中提取 classes
    coco_classes_path = 'coco_classes.txt'
    coco_classes = extract_classes(coco_classes_path)

    # 获取 coco_class-向量 键值对
    coco_classes_vector_map = {}
    for coco_class in coco_classes:
        coco_class_vector = getTextVector(coco_class).cpu().numpy().tolist()
        coco_classes_vector_map[coco_class] = coco_class_vector[0]

    # 映射 keyword_vector_map 的 key 到 coco_classes_vector_map 的 key
    keyword_coco_class_map = []
    for keyword, vector in keyword_vector_map.items():
        similarities = []
        # 计算余弦相似度
        for coco_class, coco_vector in coco_classes_vector_map.items():
            similarity = cosine_similarity(vector, coco_vector)
            similarities.append((similarity, coco_class))
        # 取前三个最高相似度
        similarities.sort(reverse=True)
        top_similarities = similarities[:3]
        # 添加对应的 coco_class
        for sim, coco_class in top_similarities:
            keyword_coco_class_map.append(coco_class)

    # 计算每个shotId的分数乘积之和
    query_scores = {}
    for shot_id, target_scores in shots.items():
        score_products = []
        for target in keyword_coco_class_map:
            target_score_sum = sum([score for t, score in target_scores if t == target])
            target_score_sum += 1  # 将每个score加1
            if target_score_sum > 2.5:
                target_score_sum = 2.5
            score_products.append(target_score_sum)
        query_score = reduce(mul, score_products)
        query_scores[shot_id] = query_score

    # 按query_score降序排列结果
    sorted_results = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)

    # 找到最大和最小的query_score
    max_score = sorted_results[0][1]
    min_score = sorted_results[-1][1]

    # 如果最大值和最小值相等，不进行归一化处理
    if max_score != min_score:
        # 对每个query_score进行归一化处理
        normalized_results = [(shot_id, (score - min_score) / (max_score - min_score)) for shot_id, score in
                              sorted_results]
    else:
        normalized_results = [(shot_id, 1) for shot_id, score in sorted_results]

    # 创建文件名为 key.txt 的文件并打开
    queryId = "1" + queryId

    with open(result_path + queryId + '.txt', 'w', newline='') as file:
        # 输出归一化后的结果
        for shot_id, score in normalized_results:
            # 写入文件中
            if score > 0:
                file.write(queryId + " 0 " + shot_id + "\n")

    with open(result_prob_path + queryId + '.txt', 'w', newline='') as file:
        # 输出归一化后的结果
        for shot_id, score in normalized_results:
            # 写入文件中
            if score > 0:
                file.write(queryId + " 0 " + shot_id + " " + str(score) + "\n")
