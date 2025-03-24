import base64
import cv2
from IPython.core.debugger import prompt
from openai import OpenAI
import os
import json
import re


def image_encoder(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_base = base64.b64encode(buffer).decode('utf-8')
    return img_base

def client_build():
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key=os.getenv("modelscope_token", "default_value") # ModelScope Token
    )
    return client

def run(client,img_base,personal_prompt):
    response = client.chat.completions.create(
        model='Qwen/Qwen2.5-VL-7B-Instruct', # ModelScope Model-Id
        messages=[{
            'role':
                'user',
            'content': [{
                'type': 'text',
                'text': personal_prompt,
            }, {
                'type': 'image_url',
                'image_url':{
                    'url': f"data:image/png;base64,{img_base}",
                }
            }],
        }],

    )

    return response.choices[0].message.content

def img2text(image,personal_prompt):
    client = client_build()
    img_base = image_encoder(image)
    text = run(client,img_base,personal_prompt)
    return extract_json_from_string(text)

def extract_json_from_string(s):
    json_match = re.search(r'\{.*?\}', s, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None
    else:
        print("未找到 JSON 格式的字符串")
        return None

if __name__ == "__main__":
    image = cv2.imread("//test1.png")
    prompt = """
    描述图片人物的相貌，请按照如下格式返回JSON格式字符串：
    {"年龄":"少年" || "老年","眼镜":"带眼镜" || "不带眼镜"}
    不要返回多余内容。
    """
    ans = img2text(image,prompt)
    print(ans["年龄"])

