"""
GraphRAG 记忆层 - 图数据库存储模块
使用 Neo4j 存储用户、实体和关系
"""
from neo4j import GraphDatabase
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib


class JokeBearGraphStore:
    """自嘲熊图数据库存储类"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "jokebear2024"):
        """初始化 Neo4j 连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()
    
    def _init_schema(self):
        """初始化图谱索引和约束"""
        with self.driver.session() as session:
            # 创建唯一约束 (Neo4j 5.x 语法)
            session.run("""
                CREATE CONSTRAINT user_id_constraint IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT entity_id_constraint IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """)
            # 创建索引加速查询
            session.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX emotion_index IF NOT EXISTS FOR (e:EmotionState) ON (e.emotion)")
    
    def _generate_entity_id(self, entity_type: str, name: str) -> str:
        """生成实体 ID"""
        content = f"{entity_type}:{name.lower().strip()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{entity_type.lower()}_{hash_val}"
    
    def upsert_user(self, user_id: str, name: Optional[str] = None):
        """创建或更新用户节点"""
        with self.driver.session() as session:
            session.run("""
                MERGE (u:User {user_id: $user_id})
                SET u.name = COALESCE($name, u.name),
                    u.last_active = datetime()
            """, user_id=user_id, name=name)
    
    def extract_and_store_entities(
        self,
        user_id: str,
        entities: List[Dict[str, Any]],
        conversation_id: str
    ):
        """从对话中提取并存储实体"""
        with self.driver.session() as session:
            for entity in entities:
                entity_type = entity.get('type', 'Entity')
                entity_name = entity.get('name', '')
                
                if not entity_name:
                    continue
                    
                entity_id = self._generate_entity_id(entity_type, entity_name)
                
                # 准备节点属性
                base_props = {
                    'entity_id': entity_id,
                    'name': entity_name,
                    'mentioned_count': 1,
                    'last_mentioned': datetime.now().isoformat()
                }
                
                # 添加额外属性
                if 'sentiment' in entity:
                    base_props['sentiment'] = entity['sentiment']
                if 'intensity' in entity:
                    base_props['intensity'] = entity['intensity']
                if 'trigger' in entity:
                    base_props['trigger'] = entity['trigger']
                if 'date' in entity:
                    base_props['date'] = entity['date']
                if 'relation_to_user' in entity:
                    base_props['relation_to_user'] = entity['relation_to_user']
                
                # 创建实体节点
                session.run(f"""
                    MERGE (e:{entity_type} {{entity_id: $entity_id}})
                    SET e.name = $name,
                        e.mentioned_count = COALESCE(e.mentioned_count, 0) + 1,
                        e.last_mentioned = datetime()
                """, entity_id=entity_id, name=entity_name)
                
                # 更新额外属性
                for key, value in base_props.items():
                    if key not in ['entity_id', 'name', 'mentioned_count', 'last_mentioned']:
                        session.run(f"""
                            MATCH (e:{entity_type} {{entity_id: $entity_id}})
                            SET e.{key} = $value
                        """, entity_id=entity_id, value=value)
                
                # 创建用户与实体的关系
                relation_type = entity.get('relation_type', 'MENTIONED')
                session.run(f"""
                    MATCH (u:User {{user_id: $user_id}})
                    MATCH (e:{entity_type} {{entity_id: $entity_id}})
                    MERGE (u)-[r:{relation_type}]->(e)
                    SET r.count = COALESCE(r.count, 0) + 1,
                        r.last_at = datetime(),
                        r.conversation_id = $conversation_id
                """, user_id=user_id, entity_id=entity_id, conversation_id=conversation_id)
    
    def get_user_memories(
        self,
        user_id: str,
        query_type: str = "relevant",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """检索用户相关记忆"""
        with self.driver.session() as session:
            if query_type == "relevant":
                # 获取最近提及的实体
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r]->(e)
                    RETURN
                        labels(e)[0] as entity_type,
                        e.name as name,
                        e.sentiment as sentiment,
                        r.count as mention_count,
                        r.last_at as last_mentioned
                    ORDER BY r.last_at DESC
                    LIMIT $limit
                """, user_id=user_id, limit=limit)
            elif query_type == "positive":
                # 获取用户喜欢的东西
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r:LIKES]->(e)
                    WHERE COALESCE(r.strength, e.sentiment, 0) > 0.7
                    RETURN
                        labels(e)[0] as entity_type,
                        e.name as name,
                        COALESCE(r.strength, e.sentiment, 0) as preference_strength
                    ORDER BY preference_strength DESC
                    LIMIT $limit
                """, user_id=user_id, limit=limit)
            elif query_type == "emotional":
                # 获取当前情绪状态
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r:FEELS]->(e:EmotionState)
                    WHERE r.intensity > 0.5
                    RETURN
                        e.emotion as emotion,
                        e.intensity as intensity,
                        e.trigger as trigger
                    ORDER BY r.intensity DESC
                    LIMIT $limit
                """, user_id=user_id, limit=limit)
            else:
                # 默认查询最近记忆
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r]->(e)
                    RETURN
                        labels(e)[0] as entity_type,
                        e.name as name,
                        r.count as mention_count,
                        r.last_at as last_mentioned
                    ORDER BY r.last_at DESC
                    LIMIT $limit
                """, user_id=user_id, limit=limit)
            
            return [record.data() for record in result]
    
    def get_memory_chain(self, user_id: str, topic: str) -> List[Dict]:
        """获取与某个主题相关的记忆链"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {user_id: $user_id})
                OPTIONAL MATCH (u)-[r1]-(e1)
                WHERE e1.name CONTAINS $topic OR e1.emotion = $topic
                OPTIONAL MATCH (e1)-[r2]-(e2)
                RETURN
                    e1.name as entity,
                    labels(e1)[0] as type,
                    e2.name as related,
                    type(r2) as relation
                ORDER BY r1.last_at DESC
                LIMIT 10
            """, user_id=user_id, topic=topic)
            return [record.data() for record in result]
    
    def add_preference(self, user_id: str, item_name: str, strength: float = 0.8):
        """添加用户喜好"""
        entity_id = self._generate_entity_id("Preference", item_name)
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Preference {entity_id: $entity_id})
                SET e.name = $name,
                    e.category = 'general'
            """, entity_id=entity_id, name=item_name)
            
            session.run("""
                MATCH (u:User {user_id: $user_id})
                MATCH (e:Preference {entity_id: $entity_id})
                MERGE (u)-[r:LIKES]->(e)
                SET r.strength = $strength,
                    r.created_at = datetime()
            """, user_id=user_id, entity_id=entity_id, strength=strength)
    
    def add_emotion(self, user_id: str, emotion: str, intensity: float, trigger: str = ""):
        """添加用户情绪状态"""
        entity_id = self._generate_entity_id("EmotionState", f"{emotion}_{datetime.now().strftime('%Y%m%d')}")
        with self.driver.session() as session:
            session.run("""
                MERGE (e:EmotionState {entity_id: $entity_id})
                SET e.emotion = $emotion,
                    e.intensity = $intensity,
                    e.trigger = $trigger
            """, entity_id=entity_id, emotion=emotion, intensity=intensity, trigger=trigger)
            
            session.run("""
                MATCH (u:User {user_id: $user_id})
                MATCH (e:EmotionState {entity_id: $entity_id})
                MERGE (u)-[r:FEELS]->(e)
                SET r.intensity = $intensity,
                    r.since = datetime()
            """, user_id=user_id, entity_id=entity_id, intensity=intensity)
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
