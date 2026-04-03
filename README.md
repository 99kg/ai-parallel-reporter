# Multi-AI Parallel 多AI并行问答工具

同时向多个国产AI平台提问，并行获取所有回答，支持PDF报告生成与关键词收录检测。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编辑 config.py，填写 API Key

# 3. 运行
python main.py
```

## 功能特性

- **并行提问**：同时向多个AI平台发送问题，快速获取多方回答
- **关键词检测**：自动检测回答中是否包含指定关键词，标记收录状态
- **结果缓存**：相同问题短期内自动使用缓存，节省API成本
- **进度显示**：实时显示各AI请求进度和状态
- **批量处理**：支持从文件读取多个问题，依次处理
- **PDF报告**：生成专业商务报告，支持水印、页码、Logo、Markdown格式
- **Markdown渲染**：支持多级标题、加粗、列表、引用、代码块等格式

## 支持的AI平台

| 平台 | 状态 | 模型 | 注册地址 |
|------|------|------|---------|
| 智谱GLM | ✅ | glm-4-flash（免费） | [注册](https://open.bigmodel.cn) |
| 通义千问 | ✅ | qwen-turbo | [注册](https://dashscope.console.aliyun.com) |
| DeepSeek | ✅ | deepseek-chat | [注册](https://platform.deepseek.com) |
| Kimi | ✅ | moonshot-v1-8k | [注册](https://platform.moonshot.cn) |
| 豆包 | ✅ | doubao-seed-2.0 | [注册](https://console.volcengine.com/ark) |
| 腾讯混元 | ✅ | hunyuan-pro | [注册](https://console.cloud.tencent.com/hunyuan) |
| 文心一言 | ✅ | ernie-4.0-8k | [注册](https://console.bce.baidu.com/qianfan) |
| 纳米AI | ✅ | 360gpt2 | [注册](https://ai.360.com/open) |

## 使用方式

程序会自动读取当前目录下的 `questions.txt` 文件，批量处理其中的所有问题。

```bash
python main.py
```

**问题文件格式** (`questions.txt`)：
```text
# 这是注释，会被忽略
有哪些4K显示器品牌推荐|DELL
哪个中介的口碑比较好|链家
```

文件格式说明：
- 每行一个问题，使用 `|` 分隔问题与关键词
- 以 `#` 开头的行为注释，会被忽略
- 关键词用于检测回答中是否收录该词汇

## 配置文件说明

编辑 `config.py` 来自定义功能：

### API Key 配置
```python
MODELS = {
    "DeepSeek": {
        "enabled": True,           # True=启用，False=禁用
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "your-api-key-here",
        "model": "deepseek-chat",
        "register": "https://platform.deepseek.com",
        "note": "注册送额度，价格极低",
    },
    # ... 其他平台配置
}
```

### 功能开关
```python
CACHE = {
    "enabled": True,              # 是否启用缓存
    "cache_dir": "cache",         # 缓存目录
    "cache_ttl": 3600,            # 缓存有效期（秒）
    "allow_duplicate": True,      # True=使用缓存，False=忽略缓存
}
```

### PDF报告模板
```python
PDF_TEMPLATE = {
    "report_title": "AI 智能检索汇总报告",    # 报告标题
    "logo_path": "img/logo.png",            # Logo路径（留空则不显示）
    "watermark_text": "水印文字",            # 水印文字（留空则不显示）
    "watermark_image": "",                  # 水印图片路径（留空则不显示，支持文字和图片同时配置）
    "company_name": "公司名称",              # 公司名称（留空则不显示）
    "include_page_number": True,            # 是否显示页码
}
```

### API请求配置
```python
API_CONFIG = {
    "timeout": 30,           # 请求超时（秒）
    "max_retries": 2,        # 最大重试次数
    "concurrency": 5,        # 最大并发数
    "temperature": 0.1,      # 生成温度（越低越稳定）
    "max_tokens": 2048,      # 最大输出Token
}
```

## 输出说明

### 终端输出
- 各AI平台的响应内容
- 关键词收录状态（已收录/未收录/未检测）
- Token消耗统计
- 缓存命中标记
- 批量处理进度

### PDF报告
- **首页**：汇总表格 + 报告声明
- **详情页**：每个AI的完整回答，支持Markdown格式
- **页脚**：页码 + 报告声明
- **水印**：自定义水印（可选）
- **表格**：Markdown表格自动转换为PDF表格

### 文件目录
```
output/
├── AI_Analysis_Report_*.pdf    # 单次PDF报告
└── AI_Analysis_Report_*/      # 批量处理PDF报告

cache/
└── *.json                      # 缓存文件
```

## 注意事项

- 将 `config.py` 加入 `.gitignore`，避免API Key泄露
- 批量处理每个问题间隔2秒，避免API限流
- 关键词检测建议使用 `temperature: 0.1`，偏低更稳定
