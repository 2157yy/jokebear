## 项目结构
```
jokebear/
├── data_get/
│   └── jokebear.json          # 角色对话数据集
├── memory/
│   ├──    # LLaMA-Factory 兼容的训练入口
│   ├── merge_lora.py          # 合并 LoRA 权重
│   └── convert_to_gguf.sh     # HF → GGUF 转换脚本
├── configs/
│   └── qwen2_5_7b_qlora.yaml  # 训练配置（A5000用）
├── jokebear_model.ipynb       # 训练过程
└── README.md
```
