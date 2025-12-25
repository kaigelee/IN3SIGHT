import json
import os

import torch.cuda

from utils import Qwen_util
from tqdm import tqdm

class MemoryState:

    initial_query:str = ''
    inspection_result: str = ''
    inspection_answer: str = ''
    same_judge_result: str = ''
    same_judge_answer: str = ''
    investigation_result: str = ''
    investigation_answer: str = ''

    def __init__(self, initial_query=''):
        self.initial_query = initial_query

import re
import json

def fix_unescaped_quotes(json_string):
    """
    修复未转义的双引号问题（如 "The image shows "Hello World"" 变为合法 JSON）。
    """
    # 寻找不在键值对中的双引号
    return re.sub(r'(?<!\\)"(.*?)"(?![:,}\]])', r'"\1"', json_string)

def fix_extra_commas(json_string):
    """
    移除多余的逗号（如 {"key": "value",} 变为 {"key": "value"}）。
    """
    # 删除逗号后面紧跟 } 或 ]
    return re.sub(r",\s*([}\]])", r"\1", json_string)

def fix_missing_commas(json_string):
    """
    修复缺少逗号的问题（如 {"key1": "value1" "key2": "value2"}）。
    """
    # 在 } 或 " 后面没有逗号的地方插入逗号
    return re.sub(r'(?<=[}\]"])(")', r',\1', json_string)

def fix_non_json_format(json_string):
    """
    尝试将非标准 JSON 转为标准 JSON（如 key: value 格式转为 JSON 格式）。
    """
    # 添加双引号包裹键和值（如果缺失）
    json_string = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', json_string)
    return json_string

def fix_json(json_string):
    """
    综合修复函数，依次修复常见问题。
    """
    json_string = fix_unescaped_quotes(json_string)
    json_string = fix_extra_commas(json_string)
    json_string = fix_missing_commas(json_string)
    json_string = fix_non_json_format(json_string)
    return json_string

def inspection_reasoning(train_json, batch_idx, initial_prompt):

    message_list = []
    memory_list = []


    for idx in batch_idx:
        img = "./datasets/" + train_json[idx]['image_path']

        caption = train_json[idx]['caption']
        memory = MemoryState(
            f'''{initial_prompt.format(caption=caption)}''',)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{img}", "max_pixels": 512*512},
                    {"type": "text", "text": f'''{memory.initial_query}'''}
              ],
            }
        ]
        memory_list.append(memory)
        message_list.append(messages)

    result_list = Qwen_util.img2text_batch(message_list)


    for idx in range(0,len(batch_idx)):
        memory_list[idx].inspection_result = result_list[idx]


    final_message_list = []
    for idx in range(0,len(batch_idx)):

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'''result: {memory_list[idx].inspection_result} \n
                   If the result states that the image does not match the caption or no, output "No". If the result states that the image matches the caption or yes, output "Yes".DO NOT OUTPUT ANY OTHER WORDS.'''}
                ]
            }
        ]
        final_message_list.append(messages)

    answer_list = Qwen_util.text2text_batch(final_message_list)

    for idx in range(0,len(batch_idx)):
        memory_list[idx].inspection_answer = answer_list[idx]

    return memory_list

filename = "./datasets/test_external_info_llama_final.json"
# filename = "./datasets/VERITE_test_external_info_final1.json"
with open(filename, 'r', encoding='utf-8') as file:
    # 加载JSON文件内容到data变量中
    test_json = json.load(file)

output_filename = "./output/2/test.json"

# with open(output_filename, 'r', encoding='utf-8') as file:
#     # 加载JSON文件内容到data变量中
#     result_list = json.load(file)

result_list = []
batch_size = 8


#25-27
# best_prompt = "Evaluate the relevance and accuracy of the following caption in relation to the associated image content: '{caption}'"

#1-1
# best_prompt = "Evaluate whether the following image caption accurately describes the image: '{caption}'. Do they match?"
#1-26
# best_prompt = "Does the caption '{caption}' accurately match the image? Please provide a brief explanation for your evaluation."

#2-29
best_prompt = "Assess the following caption for its accuracy and how well it relates to the provided image: {caption}"
#2-12
# best_prompt = "Assess the accuracy and relevance of the following caption in relation to the image provided: {caption}"
#2-1
# best_prompt = "Does the following caption accurately describe the image? {caption}"

#8-1
# best_prompt = "Given the following caption: '{caption}', does it accurately describe the content of the image? Please provide your judgment."
# F2F:3184 T2T:2230  0.745
#8-25
# best_prompt = "Evaluate the caption: '{caption}'. Does it accurately match the content of the provided image? Please provide a clear judgment on their correspondence."
# F2F:2820 T2T:2786  0.772

#4-22
# best_prompt = "Assess the caption '{caption}' to determine if it correctly matches the paired image."

# c_3_b_8-19
# best_prompt = "How accurately does this caption align with the visual content of the provided image? {caption}"
# F2F:2944 T2T:2594  0.762

# c_5_b_8-23
# best_prompt = "Does the following caption perfectly align with the content displayed in the image? {caption}"
# F2F:3076 T2T:2569  0.777

# c_9_b_8_2
# best_prompt = "Evaluate whether the caption effectively represents the essence of the image: {caption}"
# F2F:2803 T2T:2637 0.749

# c_7_b_6-8
# best_prompt = "How well does this caption align with the image? {caption}"
# F2F:2960 T2T:2589 0.764

# c_7_b_4-12
# best_prompt = "Assess if the provided caption correctly summarizes the content of the given image: {caption}"
# F2F:2600 T2T:2868 0.753

# c_7_b_10
# best_prompt = "Does the following caption match the content of the image provided? {caption}"
# F2F:3264 T2T:2274 0.762


length = len(result_list)
for start in tqdm(
            range(length, len(test_json), batch_size),
            desc=f"Processing",
            unit="batch",
        ):
    end = min(len(test_json), start + batch_size)
    # torch.cuda.empty_cache()
    batch_idx = range(start, end)

    inspection_memory_list = inspection_reasoning(test_json, batch_idx, best_prompt)

    for idx in range(0, len(batch_idx)):
        res = {}
        img = "./datasets/" + test_json[batch_idx[0] + idx]['image_path']
        caption = test_json[batch_idx[0] + idx]['caption']
        res['img'] = img
        res['caption'] = caption
        res['gt'] = test_json[batch_idx[0] + idx]['falsified']
        res['inspection_result'] = inspection_memory_list[idx].inspection_result
        res['inspection_answer'] = inspection_memory_list[idx].inspection_answer
        result_list.append(res)


    if not os.path.exists(output_filename):
        with open(output_filename, 'w') as file:
            pass  # 创建空文件

    # 使用with语句打开文件，'w'表示写入模式，encoding='utf-8'确保文件以UTF-8编码保存
    with open(output_filename, 'w', encoding='utf-8') as file:
        # 使用json.dump()函数将字典转换为JSON格式并写入文件
        # indent参数用于美化输出，使其更易于阅读
        json.dump(result_list, file, ensure_ascii=False, indent=4)