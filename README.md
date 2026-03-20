# Jokeber（自嘲熊ip）智能情感型对话模型

## 1. jokebear - v1

## 这是最简单的一个版本
- 模型选用：Qwen/Qwen2-1.5B-Instruct
- 训练平台：Colab（T4gpu）
- 训练方法： LLaMA-Factory微调，基于网上爬取的自嘲熊形象和习惯，生成378条对话数据集，训练模型，得到权重。
- 加载模型：创建py文件，编写本地运行脚本，加载训练好的权重模型，并且写开始对话的提示词，本机为mac端，因此脚本是基于mac编写的

## 一、 jokebear - v1

## 这是最简单的一个版本
- 模型选用：Qwen/Qwen2-1.5B-Instruct
- 训练平台：Colab（T4gpu）
- 训练方法： LLaMA-Factory微调，基于网上爬取的自嘲熊形象和习惯，生成378条对话数据集，训练模型，得到权重。
- 加载模型：创建py文件，编写本地运行脚本，加载训练好的权重模型，并且写开始对话的提示词，本机为mac端，因此脚本是基于mac编写的

## 2. 评价

1. 1.5b参数模型，是在参数量过小，导致自嘲熊的形象塑造过于单调偏一，回复一直围绕吃拉面为主，不符合自潮熊的自嘲、可爱形象。
2. 同时，由于参数量小，模型已经成为这个角色，而不基于提示词，模型已经被洗脑。
3. 这是一次较为成功的尝试，过程磕磕绊绊，但是训练出了第一个自己的模型

## 3.未来方向
### 记忆：
像人类一样，筛选性记忆，对关键性记忆保留
- 人这一辈子不会记忆所有，但是会保留moment
- 下次对话就像老友一样，可以在一些对话中触发关键记忆
- 初次见面时，用户最好可以定义模型的角色（恋人，挚友，亲人，普通朋友）

### 情感：
人都有情感波动，自潮熊模型对于文本有一定的情感揣测能力，同时针对性情感揣测，如：面对对象时，你会患得患失，会揣摩任何一句话一个表情。面对一面之人，不过多分析，平常温和的回答就好。面对挚友，可以表达有些难说的话。

### 对话：
具有自潮性：
```json
		"instruction": "人，我会幸福吗？",
        "output": "幸福这道门太窄，你是猪，挤不进去！"
```



## 二、jokebearv2

	哈哈哈哈，以为我已经消亡了吗，许久不更新，然而并没有，主包只是花了点时间渡劫，之前有些事情大伤元气，butbutbut，满血而归，心流启动！！！

	在这鸣谢我的好朋友，好厚米们，栓QQQ，and，most important，栓QQQ某人，对，知道的就知道，不知道就别知道，我以为自己会放弃jokebear，然而我卷土重来了，version兔，yeah，虽然是个草稿，但也写点什么记录一下吧，这是向阳而生的证明。

### 1、neo4j+graphrag
这是主要建立图关系的，jokebear有很多朋友，比如短鼻猪，但是普通的训练，并不会让模型知道这俩之间是有关系的，而是机械式的麻木记住有这个朋友，因此在输出时可能不精准，现在在neo4j图关系数据库加持下，以及graphRAG图关系检索算法下，我们可以把关系绑定起来，让模型建立关系的意识

### 2、data
针对jokebear的可能性格方式，和说话语气及态度，我重新建立了一版本的数据集，当然很不完善，这里面包含我的设想。

### 3、change model
- 由原来Qwen/Qwen2-1.5B-Instruct升级为Qwen/Qwen2.5-7B-Instruct

### 4.emotion
- 加入了情感模块，通过用户对话检索关键词匹配5种情绪

### 5.docker run
docker compose up -d neo4j
常用停止/重启命令：
  docker compose stop neo4j
  docker compose start neo4j
  docker compose down

### 6、近期代码改进（2026-03）
- 训练配置升级为 7B 路线：新增 `configs/qwen2_5_7b_qlora_dialogue.yaml`，将训练基座改为 `Qwen/Qwen2.5-7B-Instruct`，并按 A5000 调整了 batch、梯度累积、精度和保存步数。
- 导出脚本同步升级：`scripts/merge_lora.py` 改为导出 `jokebear_7b_qlora_dialogue_v1`，目标目录为 `./merged_jokebear_model_v1`，后续可直接本地加载。
- 推理链路加入独立情感模块：新增 `memory/emotion_classifier.py`，在 `run_mac.py` 中先走独立情感预测，再把结果注入 system prompt（`【情感状态】...`），并保留 fallback 到 `extract_emotion`。
- 记忆层情绪字段统一：`memory/graph_store.py` 中统一 `EmotionState` 的 `name/emotion` 字段，关系层补齐 `r.intensity`，情绪查询改用 `COALESCE(r.intensity, e.intensity, 0.0)`，减少漏召回。
- Neo4j 启动配置优化：`docker-compose.yml` 去掉过时 `version` 字段，简化启动项后容器可稳定启动（`jokebear_neo4j`）。
- memory 模块导出统一：`memory/__init__.py` 已加入 `EmotionClassifier`，方便后续统一 import。
- 当前运行模型说明：`run_mac.py` 默认加载 `./merged_jokebear_model`（现有可用基线），7B 新权重在训练与导出完成后加载 `./merged_jokebear_model_v1`。

### 7、next step
优化完善训练数据集，建立合理的图关系数据，并且切换更优、参数更大的模型，跑出来这一版本的模型


 That’s all.

## 三、手工定义小熊关系图（你自己定义关系内容）

你只需要编辑这个文件：
- `data_get/world_graph_manual.json`

格式：
- `nodes`: 你定义的角色/地点/物品
- `relations`: 你定义的关系边（比如 `BEST_FRIEND_WITH`、`RIVAL_OF`）

可选：先从模板复制一份再改（推荐）

```bash
cp data_get/world_graph_manual_template.json data_get/world_graph_manual.json
```

先做预检查（不写入数据库）：

```bash
python3 scripts/import_manual_graph.py --file data_get/world_graph_manual.json --dry-run
```

确认后正式导入：

```bash
python3 scripts/import_manual_graph.py --file data_get/world_graph_manual.json
```

注意：
- 关系类型和标签建议使用大写英文与下划线（例如 `BEST_FRIEND_WITH`）。
- 这个导入脚本会给手工节点自动加 `Entity` 和 `WorldEntity` 标签，便于和用户记忆区分。
