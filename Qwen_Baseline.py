import base64
import json
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

from utils import Qwen_util

os.environ["OPENAI_API_KEY"] = "sk-itzvOjtKAeuWdEojgC2cO74vbVwiSgRMhVj9iZrU6magTBib"

filename = "./datasets/output_data_test_400.json"
with open(filename, 'r', encoding='utf-8') as file:
    # 加载JSON文件内容到data变量中
    train_json = json.load(file)

result_list = []
for data in train_json:
    caption = data['caption']
    img = "./datasets/" + data['image_path']
    gt = data['falsified']

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{img}", "max_pixels": 512 * 512},
                # {"type": "text", "text": f'''Evaluate the following caption and determine if it accurately describes the image: '{caption}' '''}
                # {"type": "text",
                #  "text": f'''Evaluate the relevance and accuracy of the following caption in relation to the associated image content: '{caption}' '''}
                # {"type": "text",
                #  "text": f'''Does the following caption accurately describe the image? {caption} '''}
                {"type": "text",
                 "text": f'''Evaluate the following caption and determine if it accurately describes the image: '{caption}'
      To ensure a clear and accurate understanding of this image, please identify the following aspects:
      Time Context: Based on any visible details in the image, estimate the approximate time period or season. If any clues such as weather, fashion, or setting can indicate a time frame, please include those.
      Location: Describe the location, including any recognizable landmarks, architectural styles, or environmental features that may provide insight into where this image was taken. If the location cannot be determined, note that too.
      People: Identify and describe any people in the image, including approximate ages, attire, possible relationships, or roles they may have, based on context. If specific individuals cannot be determined, focus on general characteristics.  
      Events or Actions: Outline the main actions or events depicted. Describe what appears to be happening, including any relevant details from expressions, gestures, or objects in the image.
      Answer:'''}

            ],
        }
    ]
    result = Qwen_util.img2text_batch([messages])[0]


    res = {}
    res['img'] = img
    res['caption'] = caption
    res['gt'] = gt
    res['result'] = result
    result_list.append(res)
    print(res)

    filename = 'output/result_qwen_test_handcraft.json'

    # 使用with语句打开文件，'w'表示写入模式，encoding='utf-8'确保文件以UTF-8编码保存
    with open(filename, 'w', encoding='utf-8') as file:
        # 使用json.dump()函数将字典转换为JSON格式并写入文件
        # indent参数用于美化输出，使其更易于阅读
        json.dump(result_list, file, ensure_ascii=False, indent=4)

filename = "output/result_qwen_test_handcraft.json"
with open(filename, 'r', encoding='utf-8') as file:
    # 加载JSON文件内容到data变量中
    result_list = json.load(file)

for data in result_list:
    result = data['result']

    message = HumanMessage(
        content=[
            {"type": "text", "text": f'''result: {result} \n
                                If the result states that the image does not match the caption or no, output "No". If the result states that the image matches the caption or yes, output "Yes".DO NOT OUTPUT ANY OTHER WORDS.'''}],
    )
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_base="https://api.agicto.cn/v1")
    response = model.invoke([message])
    # print(response.content)
    r = response.content
    data['answer'] = r
    print(r)
with open(filename, 'w', encoding='utf-8') as file:
    # 使用json.dump()函数将字典转换为JSON格式并写入文件
    # indent参数用于美化输出，使其更易于阅读
    json.dump(result_list, file, ensure_ascii=False, indent=4)
