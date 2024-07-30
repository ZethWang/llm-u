from zhipuai import ZhipuAI
from dotenv import find_dotenv, load_dotenv
import os

# 读取本地的 .env 文件
_ = load_dotenv(find_dotenv())

# 获取 ZhipuAI 的 API 密钥
ZHIPU_key = os.environ['ZHIPUAI_API_KEY']
client = ZhipuAI(api_key=ZHIPU_key)

# 定义生成 embedding 的函数
def zhipu_embedding(text: str):
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

# 测试文本
text = '要生成 embedding 的输入文本，字符串形式。'
response = zhipu_embedding(text=text)

# 打印 embedding 相关信息
print(f'response 类型为：{type(response)}')
print(f'embedding 类型为：{response.object}')
print(f'生成 embedding 的 model 为：{response.model}')
print(f'生成的 embedding 长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:10]}')

# 设置流式传输标志
stream = True

# 创建聊天补全请求
response = client.chat.completions.create(
    model="glm-4-flash",  # 填写需要调用的模型名称
    messages=[
        {"role": "user", "content": "你好！你叫什么名字"},
    ],
    tools=[
        {
            "type": "retrieval",
            "retrieval": {
                "knowledge_id": "your knowledge id",
                "prompt_template": (
                    "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n的答案，"
                    "找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n"
                    "不要复述问题，直接开始回答。"
                )
            }
        }
    ],
    stream=stream,
)

# 打印聊天补全的响应
if stream:
    for chunk in response:
        print(chunk.choices[0].delta)
else:
    print(response.choices[0].message)
