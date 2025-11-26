import ssl

# SSL配置（内网环境用）
ssl._create_default_https_context = ssl._create_unverified_context
# 环境变量配置
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型与服务配置
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.3  # 文档筛选阈值
RETRIEVE_TOP_K = 10  # 检索最相关文档数量