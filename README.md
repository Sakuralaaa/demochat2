# ChatSDK OpenAI Wrapper - Zeabur 部署指南

## 项目说明

这是一个将 ChatSDK Demo 转换为 OpenAI 兼容 API 的包装器服务。

## 支持的模型

- `anthropic/claude-opus-4.5`
- `anthropic/claude-sonnet-4.5`
- `anthropic/claude-haiku-4.5`
- `openai/gpt-4.1-mini`
- `openai/gpt-5.2`
- `google/gemini-2.5-flash-lite`
- `google/gemini-3-pro-preview`
- `xai/grok-4.1-fast-non-reasoning`
- `anthropic/claude-3.7-sonnet-thinking`
- `xai/grok-code-fast-1-thinking`

## Zeabur 部署步骤

### 方法一：通过 GitHub 部署（推荐）

1. **将项目推送到 GitHub**
   ```bash
   cd demo.chat2api-new
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/你的用户名/你的仓库名.git
   git push -u origin main
   ```

2. **登录 Zeabur**
   - 访问 [https://zeabur.com](https://zeabur.com)
   - 使用 GitHub 账号登录

3. **创建新项目**
   - 点击 "Create Project"
   - 选择一个区域（推荐选择离你最近的）

4. **添加服务**
   - 点击 "Add Service"
   - 选择 "Git" → 选择你的 GitHub 仓库
   - Zeabur 会自动识别为 Python 项目并开始构建

5. **配置域名**
   - 部署成功后，在 "Networking" 选项卡中
   - 点击 "Generate Domain" 获取免费域名
   - 或者绑定自己的自定义域名

### 方法二：通过 Zeabur CLI 部署

1. **安装 Zeabur CLI**
   ```bash
   npm install -g zeabur
   ```

2. **登录**
   ```bash
   zeabur auth login
   ```

3. **部署**
   ```bash
   cd demo.chat2api-new
   zeabur deploy
   ```

## API 使用示例

部署成功后，你可以这样使用：

### 获取模型列表
```bash
curl https://你的域名/v1/models
```

### 发送聊天请求（流式）
```bash
curl https://你的域名/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-pro-preview",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": true
  }'
```

### 发送聊天请求（非流式）
```bash
curl https://你的域名/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemini-3-pro-preview",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": false
  }'
```

## 在其他应用中使用

由于兼容 OpenAI API 格式，你可以在支持自定义 API 端点的应用中使用：

- **Base URL**: `https://你的域名/v1`
- **API Key**: 任意值（本服务不验证 Key）

## 文件结构

```
demo.chat2api-new/
├── main.py           # 主应用程序
├── requirements.txt  # Python 依赖
├── zeabur.json       # Zeabur 配置文件
└── DEPLOY.md         # 本部署指南
```

## 注意事项

1. 本服务依赖 `curl_cffi` 库进行浏览器指纹模拟
2. 上下文存储在内存中，服务重启后会丢失
3. 免费的 Zeabur 计划有资源限制，请根据需要选择合适的套餐
