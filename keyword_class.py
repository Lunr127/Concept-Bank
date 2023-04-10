import torch
import numpy as np
from keyphrase_vectorizers import KeyphraseCountVectorizer

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

# 查询语句转化为关键词
query = "A person is mixing ingredients in a bowl, cup, or similar type of containers"
keywords = get_keywords_from_query(query)

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
keyword_coco_class_map = {}
for keyword, vector in keyword_vector_map.items():
    max_similarity = -1
    max_class = None
    # 计算余弦相似度
    for coco_class, coco_vector in coco_classes_vector_map.items():
        similarity = cosine_similarity(vector, coco_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            max_class = coco_class
    keyword_coco_class_map[keyword] = max_class

print(keyword_coco_class_map)
