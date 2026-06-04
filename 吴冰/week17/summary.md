# redis_ai_cache 项目总结

## 架构概览

```
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Web 服务器 (:8001)              │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Dashboard │  │Semantic  │  │ Embed    │  │ Semantic   │ │
│  │           │  │ Cache    │  │ Cache    │  │ Router     │ │
│  └───────────┘  └──────────┘  └──────────┘  └────────────┘ │
│                       │              │              │      │
└───────────────────────┼──────────────┼──────────────┼──────┘
                        │              │              │
┌───────────────────────┼──────────────┼──────────────┼──────┐
│              Redis (localhost:6379)                        │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐               │
│  │ HASH     │  │ HASH     │  │ FT.SEARCH   │               │
│  │ prompt→  │  │ sha256→  │  │ KNN 向量    │               │
│  │ response │  │ embed    │  │ 混合过滤    │               │
│  └──────────┘  └──────────┘  └─────────────┘               │
└────────────────────────────────────────────────────────────┘
```

## 各模块功能

| 模块 | 文件 | 功能 |
|------|------|------|
| **core/connection** | `redis_ai_cache/core/connection.py` | Redis 连接池管理，`get_redis_client()` / `close_all_connections()` |
| **core/base** | `redis_ai_cache/core/base.py` | `RedisComponent` 抽象基类，提供 key 命名空间、序列化/反序列化 |
| **core/embedding** | `redis_ai_cache/core/embedding.py` | 嵌入提供者：`HFTextVectorizer`(sentence-transformers)、`OpenAIEmbedding`、`OllamaEmbedding` |
| **cache/semantic** | `redis_ai_cache/cache/semantic_cache.py` | 语义缓存：`store(prompt, response)` / `check(query, k)`，余弦相似度 + 阈值过滤 |
| **cache/embedding** | `redis_ai_cache/cache/embedding_cache.py` | 嵌入缓存：`store(text, vector)` / `get(text)`，SHA256 哈希键 |
| **message/history** | `redis_ai_cache/message/history.py` | 对话历史：`add_messages()` / `get_recent()` / `search(query)`，每条消息存嵌入支持语义检索 |
| **router/semantic** | `redis_ai_cache/router/semantic_router.py` | 语义路由：预计算参考语句嵌入，`__call__(query)` 分类到预定义 Route |
| **vector/schema** | `redis_ai_cache/vector/schema.py` | 向量索引 schema：`VectorIndexSchema` → `FT.CREATE` 命令参数生成 |
| **vector/search** | `redis_ai_cache/vector/search.py` | 向量搜索：`create_index()` / `search(vector, top_k, filters)`，Redis Stack FT.SEARCH + KNN + 混合过滤 |
| **api_server** | `api_server.py` | FastAPI 后端，管理界面 API + 静态文件服务 |
| **static/index.html** | `static/index.html` | 单页管理前端（暗色主题），4 个标签页：Dashboard / Semantic Cache / Embed Cache / Router |
| **tests** | `tests/test_integration.py` | 7 项集成测试，覆盖全部模块 |

## 设计要点

- 参考 RedisVL 设计模式，每个组件继承 `RedisComponent` 自动管理 key 命名空间
- 语义缓存用 MD5 伪嵌入（演示用），实际可替换为 sentence-transformers 或 OpenAI
- 向量搜索需要 **Redis Stack**（带 RediSearch 模块），用 `execute_command()` 直接发 FT.SEARCH
- 前端 API URL 写死为 `http://localhost:8001`

---

## 复现步骤

### 1. 启动 Redis（Docker）

```bash
# 带 RediSearch 模块的 Redis Stack（向量搜索必需）
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### 2. 安装依赖

```bash
conda create -n redis_ai python=3.10 -y
conda activate redis_ai
pip install redis fastapi uvicorn pydantic
# 可选：真实嵌入功能
pip install sentence-transformers
```

### 3. 代码部署

把 `redis_ai_cache/`、`api_server.py`、`run_server.py`、`static/` 复制到目标机器，或直接 git clone 整个项目。

### 4. 验证

```bash
# 运行集成测试（需要 Redis 运行中）
conda run -n redis_ai python tests/test_integration.py
```

应输出 `[RESULT] 测试结果: 7 通过, 0 失败`。

### 5. 启动管理界面

```bash
cd /path/to/redis_project
conda run -n redis_ai uvicorn api_server:app --host 0.0.0.0 --port 8001
```

浏览器打开 `http://localhost:8001`。

### 关键注意事项

1. **向量搜索**需要 Redis Stack（`redis/redis-stack`），普通 Redis 不行。7 项测试中向量搜索那项会自动检测模块并 skip
2. **端口冲突**：如果 8001 被占用，换一个端口并修改 `static/index.html` 第 487 行的 `API` 变量
3. **Python >= 3.9**：代码用了 `list[float]` 类型注解，Python 3.8 不支持
4. **前端纯静态**：`static/index.html` 是单 HTML 文件，无构建步骤，修改后重启 uvicorn 即可生效
5. **演示用伪嵌入**：`SimpleEmbedder` 用 MD5 哈希模拟嵌入向量，仅供界面演示。要真实语义匹配，在 `api_server.py` 里替换为 `HFTextVectorizer` 或 `OpenAIEmbedding`
