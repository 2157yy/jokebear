"""
轻量情感分类模块
先用规则法实现统一接口，后续可无缝替换为模型分类器。
"""
from typing import Dict, Any


class EmotionClassifier:
    """独立情感分类器，提供统一预测接口"""

    def __init__(self):
        self.emotion_keywords = {
            "焦虑": ["焦虑", "紧张", "不安", "担心", "害怕"],
            "开心": ["开心", "高兴", "快乐", "兴奋", "满足"],
            "难过": ["难过", "伤心", "失落", "低落", "沮丧"],
            "愤怒": ["生气", "愤怒", "烦躁", "恼火"],
            "迷茫": ["迷茫", "困惑", "不知道", "犹豫"],
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """预测输入文本的情感标签与强度"""
        text = (text or "").strip()
        if not text:
            return {"name": "平静", "intensity": 0.5, "source": "classifier"}

        best_emotion = "平静"
        best_score = 0

        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_emotion = emotion
                best_score = score

        if best_emotion == "平静":
            intensity = 0.5
        else:
            intensity = min(1.0, 0.55 + 0.12 * best_score)

        return {
            "name": best_emotion,
            "intensity": float(intensity),
            "source": "classifier",
        }
