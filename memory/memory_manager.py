"""
记忆管理器 - 协调实体抽取和图谱存储
处理对话记忆的全流程
"""
from typing import List, Dict, Any, Optional
from .graph_store import JokeBearGraphStore
from .entity_extractor import EntityExtractor


class MemoryManager:
    """记忆管理器类"""
    
    def __init__(
        self, 
        graph_store: JokeBearGraphStore,
        entity_extractor: EntityExtractor,
        user_id: str = "default_user"
    ):
        """
        初始化记忆管理器
        
        Args:
            graph_store: 图数据库存储实例
            entity_extractor: 实体抽取器实例
            user_id: 用户 ID
        """
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor
        self.user_id = user_id
        self.conversation_count = 0
    
    def process_conversation(
        self,
        conversation: str,
        conversation_id: Optional[str] = None,
        extract_entities: bool = True
    ):
        """
        处理对话，提取并存储实体
        
        Args:
            conversation: 对话内容
            conversation_id: 对话 ID（可选）
            extract_entities: 是否抽取实体
        """
        # 确保用户存在
        self.graph_store.upsert_user(self.user_id)
        
        if extract_entities:
            # 提取实体
            entities = self.entity_extractor.extract(
                conversation,
                use_local_model=True
            )
            
            # 存储到图谱
            if entities:
                conv_id = conversation_id or f"conv_{self.conversation_count}"
                self.graph_store.extract_and_store_entities(
                    self.user_id, entities, conv_id
                )
                self.conversation_count += 1
    
    def retrieve_context(
        self,
        current_input: str,
        max_memories: int = 5
    ) -> str:
        """
        检索相关记忆，生成上下文
        
        Args:
            current_input: 当前输入
            max_memories: 最大记忆数量
            
        Returns:
            记忆上下文字符串
        """
        # 获取相关记忆
        recent_limit = max(1, max_memories * 3 // 5)
        positive_limit = max(1, max_memories - recent_limit)
        recent = self.graph_store.get_user_memories(self.user_id, "relevant", limit=recent_limit)
        positive = self.graph_store.get_user_memories(self.user_id, "positive", limit=positive_limit)
        
        # 构建记忆上下文
        context_parts = []
        
        if recent:
            context_parts.append("【用户信息】")
            for mem in recent:
                entity_type = mem.get('entity_type', 'Entity')
                name = mem.get('name', '')
                if name:
                    context_parts.append(f"- {entity_type}: {name}")
        
        if positive:
            context_parts.append("【用户喜好】")
            for mem in positive:
                name = mem.get('name', '')
                if name:
                    context_parts.append(f"- 喜欢：{name}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def format_memory_prompt(self, memories: str, current_input: str) -> str:
        """
        格式化包含记忆的 prompt
        
        Args:
            memories: 记忆上下文
            current_input: 当前输入
            
        Returns:
            格式化后的 prompt
        """
        if not memories:
            return current_input
        
        return f"""{memories}

---
当前对话：
{current_input}
"""
    
    def get_user_profile(self) -> Dict[str, Any]:
        """
        获取用户画像摘要
        
        Returns:
            用户画像字典
        """
        all_memories = self.graph_store.get_user_memories(
            self.user_id, "relevant", limit=50
        )
        
        profile = {
            "mentioned_people": [],
            "preferences": [],
            "emotions": [],
            "events": []
        }
        
        for mem in all_memories:
            entity_type = mem.get('entity_type', '')
            name = mem.get('name', '')
            
            if entity_type == 'Person':
                if name not in profile['mentioned_people']:
                    profile['mentioned_people'].append(name)
            elif entity_type == 'Preference':
                if name not in profile['preferences']:
                    profile['preferences'].append(name)
            elif entity_type == 'EmotionState':
                if name not in profile['emotions']:
                    profile['emotions'].append(name)
            elif entity_type == 'Event':
                if name not in profile['events']:
                    profile['events'].append(name)
        
        return profile
    
    def add_manual_memory(
        self,
        entity_type: str,
        entity_name: str,
        relation_type: str = "MENTIONED",
        **kwargs
    ):
        """
        手动添加记忆
        
        Args:
            entity_type: 实体类型
            entity_name: 实体名称
            relation_type: 关系类型
            **kwargs: 额外属性
        """
        entities = [{
            "type": entity_type,
            "name": entity_name,
            "relation_type": relation_type,
            **kwargs
        }]
        
        conv_id = f"manual_{self.conversation_count}"
        self.graph_store.extract_and_store_entities(
            self.user_id, entities, conv_id
        )
        self.conversation_count += 1
    
    def clear_memories(self):
        """清除所有记忆（谨慎使用）"""
        raise NotImplementedError("clear_memories 尚未实现，需要在 graph_store 中添加对应的清除逻辑")
    
    def get_memory_summary(self) -> str:
        """
        获取记忆摘要
        
        Returns:
            记忆摘要字符串
        """
        profile = self.get_user_profile()
        
        summary_parts = ["记忆摘要："]
        
        if profile['mentioned_people']:
            summary_parts.append(f"提到的人：{', '.join(profile['mentioned_people'])}")
        if profile['preferences']:
            summary_parts.append(f"喜好：{', '.join(profile['preferences'])}")
        if profile['emotions']:
            summary_parts.append(f"情绪：{', '.join(profile['emotions'])}")
        if profile['events']:
            summary_parts.append(f"事件：{', '.join(profile['events'])}")
        
        return "\n".join(summary_parts) if len(summary_parts) > 1 else "暂无记忆"
