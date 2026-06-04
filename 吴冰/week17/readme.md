# 项目背景

随着公司各业务线对 AI 能力，特别是大语言模型（LLM）和检索增强生成（RAG）应用的深入探索，公司面临一系列共通的底层技术挑战。多个团队在独立开发中重复建设向量数据库接入、语义缓存、对话记忆管理等模块，导致技术栈碎片化、资源利用率低，且难以保证生产环境的性能与稳定性。与此同时，对高并发、低延迟的实时 AI 推理与检索需求日益增长，亟需一个统一、高效、可靠的基座服务。

**项目 A**：输入文本进行情感分析

**项目 B**：输入文本进行情感分析

导致重复调用大模型，浪费资金和时间 → agent 应用的缓存模块，避免重复调用。

参考：[redis/redis-vl-python](https://github.com/redis/redis-vl-python)

本项目旨在借鉴行业先进实践（如 RedisVL 的设计理念），构建一个公司内部统一的、生产就绪的向量检索与智能缓存服务平台。其核心是充分利用公司已部署的高性能 Redis 集群，通过封装成熟的 AI 原生数据模式与操作，为上层业务提供一个简单易用、功能强大且易于扩展的 Python SDK 与配套服务，从而赋能各业务团队快速、低成本地构建高质量的 AI 应用，并确保核心组件的性能与可维护性。

# 项目任务

本项目的核心任务是打造一个服务于内部 AI 应用开发的"能力中台"，具体任务分解如下：

1. **统一向量数据管理**：提供标准化的索引定义、数据灌入与向量检索接口，支持多种向量化算法和相似度度量，结束各团队重复造轮子的状态。

2. **实现混合查询引擎**：支持将向量相似性搜索与业务元数据过滤、关键词全文搜索进行灵活组合，满足复杂场景下的精准检索需求。

3. **构建智能缓存体系**：

   - **语义缓存**：针对 LLM 调用，实现基于语义相似的请求-结果缓存，显著降低模型调用成本与响应延迟。
   - **嵌入缓存**：缓存文本到向量的转换结果，避免对相同内容进行重复的嵌入计算，节约计算资源。

4. **提供 LLM 应用支持组件**：封装对话历史管理、语义路由等通用模式，为开发 AI 智能体和聊天应用提供开箱即用的基础模块。

```
对话1: 请判断用户的情感：我今天很开心。 → 情感分析agent
对话2: 请判断用户的情感：我今天很开心。 → 情感分析agent
对话3: 请判断用户的情感：我今天很开心。 → 情感分析agent
```

# redis-vl 核心功能

## 1. Schema 与检索

创建 schema、检索数据（支持向量检索、全文检索、非结构化数据检索）→ milvus

## 2. 语义缓存 (SemanticCache)

存储历史的标准提问和标准回答。

```python
from redisvl.extensions.cache.llm import SemanticCache

# init cache with TTL and semantic distance threshold
llmcache = SemanticCache(
    name="llmcache",
    ttl=360,
    redis_url="redis://localhost:6379",
    distance_threshold=0.1  # key与历史已缓存key的距离小于此阈值，则认为相似
)

# store user queries and LLM responses
llmcache.store(prompt="What is the capital city of France?", response="Paris")
llmcache.store(prompt="中国的首都是什么？", response="北京")

# quickly check the cache with a slightly different prompt
response = llmcache.check(prompt="What is France's capital city?")
print(response[0]["response"])
```

## 3. 嵌入缓存 (EmbeddingCache)

将历史 embedding 存储下来，避免重复计算。

```python
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import HFTextVectorizer

# Initialize embedding cache
embed_cache = EmbeddingsCache(
    name="embed_cache",
    redis_url="redis://localhost:6379",
    ttl=3600  # 1 hour TTL
)

# Initialize vectorizer with cache
vectorizer = HFTextVectorizer(
    model="sentence-transformers/all-MiniLM-L6-v2",
    cache=embed_cache
)

# First call computes and caches the embedding
embedding = vectorizer.embed("What is machine learning?")

# Subsequent calls retrieve from cache (much faster!)
cached_embedding = vectorizer.embed("What is machine learning?")
```

## 4. 对话历史 (SemanticMessageHistory)

存储历史对话，支持快速检索。

```python
from redisvl.extensions.message_history import SemanticMessageHistory

history = SemanticMessageHistory(
    name="my-session",
    redis_url="redis://localhost:6379",
    distance_threshold=0.7
)

# Supports roles: system, user, llm, tool
# Optional metadata field for additional context
history.add_messages([
    {"role": "user", "content": "hello, how are you?"},
    {"role": "llm", "content": "I'm doing fine, thanks."},
    {"role": "user", "content": "what is the weather going to be today?"},
    {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}}
])
```

## 5. 语义路由 (SemanticRouter)

快速实现意图识别。

```python
from redisvl.extensions.router import Route, SemanticRouter

routes = [
    Route(
        name="greeting",
        references=["hello", "hi"],
        metadata={"type": "greeting"},
        distance_threshold=0.3,
    ),
    Route(
        name="farewell",
        references=["bye", "goodbye"],
        metadata={"type": "farewell"},
        distance_threshold=0.3,
    ),
]

# build semantic router from routes
router = SemanticRouter(
    name="topic-router",
    routes=routes,
    redis_url="redis://localhost:6379",
)

router("Hi, good morning")
# 之前的结果会缓存，后续相同意图不需要重复计算
```

# 项目效果

使用 Redis 和 Faiss 实现了对大模型调用（LLM 对话、Embedding、意图识别等）过程的高效缓存。缓存组件支持按照 session id 分开存储对话，也可以缓存大模型历史的调用和回答。组件被用在多个项目中，减少了大模型和 Embedding 模型调用次数，并节约了计算资源。

- 减少了 **50%** 的大模型调用次数
- 每天节约 **200 元** 人民币
- 减少 **90%** 对话等待时间
