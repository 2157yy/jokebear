import sys
from llamafactory.cli import main

# 定义合并参数
sys.argv = [
    "llamafactory-cli",
    "export",
    "--model_name_or_path", "Qwen/Qwen2.5-7B-Instruct", # 如果你训练的是7B，这里记得改成 7B
    "--adapter_name_or_path", "./saves/jokebear_7b_qlora_v1", # 你的训练结果路径
    "--template", "qwen",
    "--finetuning_type", "lora",
    
    # === 导出设置 ===
    "--export_dir", "./merged_jokebear_model_v1",  # 融合后的模型保存在这里
    "--export_size", "2",                       # 每个文件切片大小(GB)，方便下载
    "--export_device", "cpu",                   # 使用 CPU 合并，防止爆显存
    "--export_legacy_format", "false"
]

main()
