# 火山引擎语音识别服务开通指南

火山引擎（Volcano Engine）是字节跳动旗下的云服务平台，提供高质量的中文语音识别服务。

## 服务优势

- **高精度中文识别**：针对中文语音优化，识别准确率高
- **实时转写**：支持实时语音转文字
- **多语言支持**：支持中文、英文等多种语言
- **按量计费**：根据实际使用量付费，无需预付费

## 开通步骤

### 1. 注册火山引擎账号

访问 [火山引擎官网](https://www.volcengine.com/) 注册账号。

### 2. 开通语音识别服务

1. 登录火山引擎控制台
2. 在产品列表中找到「语音技术」→「语音识别」
3. 点击「立即开通」
4. 根据提示完成实名认证和服务开通

### 3. 创建应用

1. 进入语音识别控制台
2. 点击「创建应用」
3. 填写应用名称和描述
4. 记录生成的 **AppID**

### 4. 获取访问密钥

1. 点击右上角头像 → 「访问密钥管理」
2. 创建新的访问密钥（Access Key）
3. 记录 **Access Key ID** 和 **Secret Access Key**

**重要**：Secret Access Key 只在创建时显示一次，请妥善保管。

## 配置方法

在 `config.json` 中配置火山引擎参数：

```json
{
    "asr_engine": "volcengine",
    "volcengine_appid": "你的AppID",
    "volcengine_token": "AccessKeyID:SecretAccessKey",
    "volcengine_language": "zh-CN"
}
```

### 配置说明

- **asr_engine**: 语音识别引擎选择
  - `"whisper"` - 使用 Whisper（默认）
  - `"volcengine"` - 使用火山引擎
  - `"auto"` - 自动选择（优先火山引擎，失败时回退到 Whisper）

- **volcengine_appid**: 火山引擎应用 ID

- **volcengine_token**: 访问密钥，格式为 `AccessKeyID:SecretAccessKey`
  - 示例：`"AKLT1234567890:SecretKey1234567890"`

- **volcengine_language**: 识别语言
  - `"zh-CN"` - 中文（简体）
  - `"en-US"` - 英语（美国）
  - 更多语言代码请参考火山引擎文档

## 使用示例

### 示例 1：使用火山引擎识别本地视频

```bash
# 配置 config.json
{
    "asr_engine": "volcengine",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "your_access_key:your_secret_key",
    "volcengine_language": "zh-CN",
    "generate_srt": true
}

# 运行脚本
python video_subtitle.py /path/to/video.mp4
```

### 示例 2：自动模式（优先火山引擎）

```json
{
    "asr_engine": "auto",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "your_access_key:your_secret_key",
    "whisper_mode": "local",
    "whisper_model": "medium"
}
```

在 `auto` 模式下，系统会先尝试使用火山引擎，如果失败则自动回退到 Whisper。

## 计费说明

火山引擎语音识别按识别时长计费：

- **免费额度**：新用户通常有一定的免费试用额度
- **按量计费**：超出免费额度后按实际使用时长计费
- **价格**：具体价格请参考 [火山引擎定价页面](https://www.volcengine.com/pricing)

## 常见问题

### Q: 如何查看我的使用量？

A: 登录火山引擎控制台 → 费用中心 → 用量明细，可以查看详细的使用记录。

### Q: 火山引擎和 Whisper 如何选择？

A: 
- **火山引擎**：中文识别精度更高，速度快，但需要联网和付费
- **Whisper**：支持离线使用，免费，但中文识别精度略低，速度较慢

建议：
- 中文视频 + 有网络 → 火山引擎
- 英文视频或离线场景 → Whisper
- 不确定 → 使用 `"asr_engine": "auto"` 自动选择

### Q: 为什么火山引擎识别失败？

A: 常见原因：
1. AppID 或访问密钥配置错误
2. 账户余额不足或超出免费额度
3. 网络连接问题
4. 音频格式不支持

解决方法：
- 检查配置是否正确
- 查看控制台余额和用量
- 确认网络连接正常
- 使用 `"asr_engine": "auto"` 启用自动回退

### Q: 支持哪些音频格式？

A: 火山引擎支持常见的音频格式（WAV、MP3、M4A 等）。本工具会自动将视频音频转换为 WAV 格式后再发送给火山引擎。

## 相关链接

- [火山引擎官网](https://www.volcengine.com/)
- [语音识别产品文档](https://www.volcengine.com/docs/6561/79817)
- [API 参考文档](https://www.volcengine.com/docs/6561/80818)
- [价格说明](https://www.volcengine.com/pricing)

## 技术支持

如遇到问题，可以：
1. 查看火山引擎官方文档
2. 联系火山引擎技术支持
3. 在本项目 GitHub Issues 中提问
