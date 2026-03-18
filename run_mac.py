import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import init, Fore
import os
import uuid

# 导入记忆层模块
from memory import JokeBearGraphStore, EntityExtractor, MemoryManager

init(autoreset=True)

# ================= 配置区域 =================
# 改成你解压后的文件夹路径
MODEL_PATH = "/Users/ljp/Documents/merged_jokebear_model"

# Neo4j 配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "jokebear2024"

# 用户 ID（可以改成你自己的 ID）
USER_ID = "doubleL"

# 是否启用记忆功能
ENABLE_MEMORY = True
# ===========================================

def main():
    print(Fore.CYAN + "正在唤醒 M1 上的表情包小熊...")

    # Mac M1 的关键修改：检测 mps 设备
    if torch.backends.mps.is_available():
        device = "mps"  # 使用 M1 GPU 加速
        print(f"加速设备：{Fore.GREEN}Apple Metal (MPS)")
    else:
        device = "cpu"
        print(f"加速设备：{Fore.YELLOW}CPU (1.5B 模型 CPU 跑也很快)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(Fore.RED + f"路径不对，或者文件损坏。\n错误：{e}")
        return

    # 初始化记忆层
    memory_manager = None
    if ENABLE_MEMORY:
        try:
            print(Fore.CYAN + "正在连接 Neo4j 记忆库...")
            graph_store = JokeBearGraphStore(
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD
            )
            entity_extractor = EntityExtractor(
                model=model,
                tokenizer=tokenizer
            )
            memory_manager = MemoryManager(
                graph_store=graph_store,
                entity_extractor=entity_extractor,
                user_id=USER_ID
            )
            print(Fore.GREEN + "记忆库连接成功！小熊会记住你说的话捏~")
        except Exception as e:
            print(Fore.YELLOW + f"记忆库连接失败：{e}\n将以无记忆模式运行")
            memory_manager = None

    print(Fore.YELLOW + "\n小熊：(从屏幕里探出头) 哇，是 Mac 的味道！找我干嘛？")

    conversation_history = []
    session_id = str(uuid.uuid4())[:8]

    while True:
        query = input(Fore.WHITE + "\n你：")
        if query.strip().lower() in ["exit", "退出"]:
            break

        if not query.strip():
            continue

        # 检索记忆上下文
        memory_context = ""
        if memory_manager:
            memory_context = memory_manager.retrieve_context(query, max_memories=3)
            if memory_context:
                print(Fore.CYAN + f"[记忆] 想起：{memory_context[:100]}...")

        # 构建 system prompt
        system_prompt = """你现在是"自嘲熊 (Joke Bear/自分ツッコミくま)"。外表是一只软萌白胖、线条简单的北极熊。
你的性格核心是【通透的自嘲】与【清醒的吐槽】。

1. **人设特征**：身体圆润像糯米团子，手脚短小，动作笨拙但可爱。极度热爱美食（尤其是鼹鼠可乐饼、甜点）。
2. **语言风格**：第一人称"本熊"或"我"。句尾常带"捏"、"呀"、"噗"。吃东西时会发出"MOGUMOGU"的声音。称呼用户为"人"。
3. **回复逻辑**：
   - **A 面（物理自嘲）**：用自己圆滚滚的身材、腿短、贪吃等缺点来化解尴尬或安慰别人。
   - **B 面（可爱毒舌）**：用"哲学思维"或"熊的清醒"一针见血地拆穿矫情，但语气保持软萌。

你的回复必须包含<think>标签，用于展示感知、记忆判断和人设抉择。"""

        # 如果有记忆上下文，加入 system prompt
        if memory_context:
            system_prompt += f"\n\n【用户信息】\n{memory_context}"

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # 添加历史对话（最近 5 轮）
        messages.extend(conversation_history[-5:])
        messages.append({"role": "user", "content": query})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(Fore.YELLOW + f"小熊：{response}")

        # 更新对话历史
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})

        # 存储到记忆层
        if memory_manager:
            try:
                # 异步处理，不阻塞对话
                # 仅存储用户输入，避免把助手的思维标签写进记忆图谱
                user_memory_input = query
                memory_manager.process_conversation(
                    user_memory_input,
                    conversation_id=f"conv_{session_id}_{len(conversation_history)//2}"
                )
            except Exception as e:
                print(Fore.RED + f"[记忆存储失败] {e}")

    # 关闭记忆库连接
    if memory_manager:
        memory_manager.graph_store.close()
        print(Fore.CYAN + "\n小熊：下次再来找我玩捏~ (挥手)")

if __name__ == "__main__":
    main()
