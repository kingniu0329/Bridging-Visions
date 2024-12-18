from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np

from transformers import CLIPProcessor, CLIPModel

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并使用 DataParallel
model = CLIPModel.from_pretrained("CLIP").to(device)
model = torch.nn.DataParallel(model)

# 加载数据处理器
processor = CLIPProcessor.from_pretrained("CLIP")


def get_all_folders(folder_path):
    all_files = os.listdir(folder_path)
    folder_files = [file for file in all_files if os.path.isdir(os.path.join(folder_path, file))]
    folder_paths = [os.path.join(folder_path, folder_file) for folder_file in folder_files]
    return folder_paths


def get_all_images(folder_path):
    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.endswith((".jpg", ".png", ".jpeg"))]
    image_paths = [os.path.join(folder_path, image_file) for image_file in image_files]
    return image_paths


def get_clip_score(images_path, text, batch_size=4):
    clip_scores = []
    for i in range(0, len(images_path), batch_size):
        images_batch = [Image.open(img) for img in images_path[i:i + batch_size]]
        inputs = processor(text=text, images=images_batch, return_tensors="pt", padding=True)

        # 将输入数据移动到GPU
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        outputs = model(**inputs)
        clip_scores.append(outputs.logits_per_image)

        # 清理GPU内存
        del inputs, outputs
        torch.cuda.empty_cache()

    return torch.cat(clip_scores)


def calculate_clip_scores_for_all_categories(images_folder_path, text_prompts):
    category_folders = get_all_folders(images_folder_path)
    category_clip_scores = {}
    category_mean_scores = {}

    for category_folder, text_prompt in tqdm(zip(category_folders, text_prompts), total=len(category_folders),
                                             desc="Processing categories"):
        category_name = os.path.basename(category_folder)
        images_path = get_all_images(category_folder)
        clip_score = get_clip_score(images_path, text_prompt)
        category_clip_scores[category_name] = clip_score
        mean_score = torch.mean(clip_score, dim=0)
        category_mean_scores[category_name] = mean_score.item()

    return category_clip_scores, category_mean_scores


def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        prompts = file.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    return prompts


text_prompts = read_prompts_from_file('prompts.txt')
big_cate = ['sd','sd+ae+od','sd+ir','sd+ir+od','sd+od']
iname = 'temp'
category_clip_scores, category_mean_scores = calculate_clip_scores_for_all_categories("../samples_images/"+iname, text_prompts)
specific_category = "Category2"  # 替换为你想要输出的类别名称

# 输出 CLIP 分数
if specific_category in category_clip_scores:
    clip_score_list = category_clip_scores[specific_category].tolist()
    for score in clip_score_list:
        print(f"Category: {specific_category}, Clip Score: {score[0]:.4f}")

# 输出均值分数
if specific_category in category_mean_scores:
    mean_score = category_mean_scores[specific_category]
    print(f"Category: {specific_category}, Mean Score: {mean_score:.4f}")
# for category, clip_score in category_clip_scores.items():
#     clip_score_list = clip_score.tolist()
#     for score in clip_score_list:
#         print(f"Category: {category}, Clip Score: {score[0]:.4f}")

# for category, mean_score in category_mean_scores.items():
#     print(f"Category: {category}, Mean Score: {mean_score:.4f}")
overall_mean_score = sum(category_mean_scores.values()) / len(category_mean_scores)
print(iname,':',overall_mean_score)
