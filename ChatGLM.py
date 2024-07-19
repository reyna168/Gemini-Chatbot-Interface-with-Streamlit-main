from typing import List, Optional

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import requests
import json

URL = 'http://0.0.0.0:5000/api/v1/generate'

class ChatGLM(LLM):
    max_new_tokens = 1000
    temperature = 0.1
    top_p = 1
    penalty_alpha = 0

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # headers中添加上content-type这个参数，指定为json格式
        headers = {'Content-Type': 'application/json'}
        data = {
            'prompt': prompt,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'penalty_alpha': self.penalty_alpha,
        }
        # 调用api
        response = requests.post(URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['results'][0]['text']
        return "查询结果错误"