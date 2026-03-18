"""
实体抽取器 - 从对话中提取实体信息
使用 LLM 自动识别人名、事件、喜好、情绪等实体
"""
from typing import List, Dict, Any
import json
import re


# 实体抽取 Prompt 模板
ENTITY_EXTRA_PROMPT = """从以下对话中提取实体信息，返回 JSON 格式。

对话内容：
{conversation}

提取以下类型的实体：
1. Person - 提到的人物（家人、朋友、同事等）
2. Event - 重要事件（生日、面试、会议等）
3. Preference - 喜好（食物、活动、物品等）
4. EmotionState - 当前情绪状态
5. Location - 地点

返回格式（只返回 JSON，不要其他内容）：
{{
    "entities": [
        {{"type": "Person", "name": "张三", "relation_type": "MENTIONED", "relation_to_user": "同事"}},
        {{"type": "Preference", "name": "草莓蛋糕", "relation_type": "LIKES", "sentiment": 0.9}},
        {{"type": "EmotionState", "name": "焦虑", "relation_type": "FEELS", "intensity": 0.7, "trigger": "工作"}},
        {{"type": "Event", "name": "生日", "relation_type": "HAS_EVENT", "date": "09-21"}}
    ]
}}

如果没有实体，返回空列表：{{"entities": []}}
"""


class EntityExtractor:
    """实体抽取器类"""
    
    def __init__(self, llm_client=None, model=None, tokenizer=None):
        """
        初始化实体抽取器
        
        Args:
            llm_client: LLM 客户端（可选，用于调用外部 API）
            model: 本地模型（可选）
            tokenizer: 本地 tokenizer（可选）
        """
        self.llm_client = llm_client
        self.model = model
        self.tokenizer = tokenizer
    
    def extract(self, conversation: str, use_local_model: bool = False) -> List[Dict[str, Any]]:
        """
        从对话中提取实体
        
        Args:
            conversation: 对话内容
            use_local_model: 是否使用本地模型
            
        Returns:
            实体列表
        """
        prompt = ENTITY_EXTRA_PROMPT.format(conversation=conversation)
        
        if use_local_model and self.model is not None and self.tokenizer is not None:
            response = self._generate_local(prompt)
        elif self.llm_client is not None:
            response = self.llm_client.generate(prompt, response_format="json")
        else:
            # 默认使用规则抽取（降级方案）
            return self._rule_based_extract(conversation)
        
        try:
            # 尝试解析 JSON
            data = json.loads(response)
            return data.get('entities', [])
        except json.JSONDecodeError:
            # JSON 解析失败，尝试从文本中提取
            return self._parse_json_from_text(response)
    
    def _generate_local(self, prompt: str) -> str:
        """使用本地模型生成"""
        import torch
        
        messages = [
            {"role": "system", "content": "你是一个实体抽取助手，只返回 JSON 格式的结果。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt")
        
        # 获取设备
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,  # 低温度保证 JSON 格式稳定
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def _rule_based_extract(self, conversation: str) -> List[Dict[str, Any]]:
        """
        基于规则的实体抽取（降级方案）
        当 LLM 不可用时使用
        """
        entities = []
        
        # 情绪关键词匹配
        emotion_keywords = {
            "焦虑": ["焦虑", "紧张", "担心", "不安"],
            "开心": ["开心", "高兴", "快乐", "兴奋"],
            "难过": ["难过", "伤心", "沮丧", "低落"],
            "愤怒": ["生气", "愤怒", "恼火", "烦躁"],
            "迷茫": ["迷茫", "困惑", "不知道", "犹豫"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in conversation:
                    entities.append({
                        "type": "EmotionState",
                        "name": emotion,
                        "relation_type": "FEELS",
                        "intensity": 0.7,
                        "trigger": keyword
                    })
                    break
        
        # 喜好关键词匹配
        like_patterns = [
            r"喜欢 (\S+)",
            r"爱吃 (\S+)",
            r"最爱 (\S+)",
            r"讨厌 (\S+)"
        ]
        
        for pattern in like_patterns:
            matches = re.findall(pattern, conversation)
            for match in matches:
                relation = "LIKES" if "讨厌" not in pattern else "DISLIKES"
                entities.append({
                    "type": "Preference",
                    "name": match,
                    "relation_type": relation,
                    "sentiment": 0.9 if relation == "LIKES" else 0.2
                })
        
        # 日期匹配（生日等）
        date_pattern = r"(\d{1,2}) 月 (\d{1,2}) 日"
        date_matches = re.findall(date_pattern, conversation)
        for month, day in date_matches:
            entities.append({
                "type": "Event",
                "name": "生日",
                "relation_type": "HAS_BIRTHDAY",
                "date": f"{month}-{day}"
            })
        
        return entities
    
    def _parse_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中解析 JSON"""
        # 尝试找到 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('entities', [])
            except json.JSONDecodeError:
                pass
        return []
    
    def extract_emotion(self, conversation: str) -> Dict[str, Any]:
        """
        专门抽取情绪状态
        
        Returns:
            情绪状态字典
        """
        entities = self.extract(conversation)
        emotions = [e for e in entities if e.get('type') == 'EmotionState']
        
        if emotions:
            return emotions[0]
        
        # 默认返回平静状态
        return {
            "type": "EmotionState",
            "name": "平静",
            "relation_type": "FEELS",
            "intensity": 0.5,
            "trigger": ""
        }
    
    def extract_preferences(self, conversation: str) -> List[Dict[str, Any]]:
        """
        专门抽取喜好
        
        Returns:
            喜好列表
        """
        entities = self.extract(conversation)
        return [e for e in entities if e.get('type') == 'Preference']
