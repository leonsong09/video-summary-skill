---
name: multi-video-summarizer
description: "Summarize video/audio content from multiple platforms and local files into structured notes with keyframe screenshots. Supports Bilibili (B站), YouTube, Douyin (抖音), Xiaohongshu (小红书), TikTok, 1800+ sites via yt-dlp, and local video files. Triggers on video URLs from any of these platforms, local file paths (e.g., /path/to/video.mp4), or Chinese requests like '总结视频', '视频笔记', '视频内容'. Detects URLs containing bilibili.com, youtube.com, youtu.be, douyin.com, xiaohongshu.com, tiktok.com, and more."
---

# Multi-Platform Video Summarizer

Extract subtitles/transcripts from video platforms and local files, then generate structured notes with SRT subtitle files. No login or cookies required.

## Supported Sources

| Source | URL/Path patterns | Extraction method |
|---|---|---|
| **Local Files** | `/path/to/video.mp4`, `C:\videos\file.mp4` | ffmpeg + ASR (Whisper or Volcano Engine) |
| Bilibili (B站) | `bilibili.com/video/`, `b23.tv/`, `BV*` | Public API (WBI signing) |
| YouTube | `youtube.com/watch`, `youtu.be/`, `youtube.com/shorts/` | `youtube-transcript-api` |
| Douyin (抖音) | `douyin.com/`, `v.douyin.com/` | `yt-dlp` |
| Xiaohongshu (小红书) | `xiaohongshu.com/`, `xhslink.com/` | `yt-dlp` |
| TikTok | `tiktok.com/` | `yt-dlp` |
| Any other | Any URL supported by yt-dlp (1800+ sites) | `yt-dlp` |

## Prerequisites

Ensure Python 3 is available:

```bash
python --version
```

## Dependency Check

Before running the extraction script, check and install required dependencies based on the video platform:

**For Bilibili videos** -- no extra dependencies needed (uses Python stdlib only).

**For YouTube videos:**
```bash
pip install youtube-transcript-api
```

**For Douyin, Xiaohongshu, TikTok, or any other platform:**
```bash
pip install yt-dlp
```

**ASR (Automatic Speech Recognition) Options:**

Choose one or both ASR engines:

1. **Whisper** (OpenAI, offline capable):
   - OpenAI API mode: `pip install openai`
   - Local mode: `pip install faster-whisper`
   - Configure in `config.json` (see Whisper Setup below)

2. **Volcano Engine** (火山引擎, high-quality Chinese):
   - No installation required (uses Python stdlib)
   - Requires API credentials (see [VOLCENGINE.md](VOLCENGINE.md))
   - Best for Chinese video transcription

Only install what is needed. If the user provides a Bilibili URL, skip dependency installation entirely.

## Workflow

### For Local Video Files

**Prerequisites:**
- ffmpeg must be installed (for audio extraction)
- ASR engine must be configured in `config.json` (Whisper or Volcano Engine)

**Steps:**

1. **Configure ASR Engine**

Choose your ASR engine in `config.json`:

**Option A: Whisper (Local)**
```json
{
    "asr_engine": "whisper",
    "whisper_mode": "local",
    "whisper_model": "medium",
    "whisper_device": "cuda",
    "whisper_compute_type": "float16",
    "language": "zh",
    "generate_srt": true,
    "srt_output_dir": "subtitles"
}
```

**Option B: Volcano Engine (火山引擎)**
```json
{
    "asr_engine": "volcengine",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "access_key:secret_key",
    "volcengine_language": "zh-CN",
    "generate_srt": true,
    "srt_output_dir": "subtitles"
}
```

**Option C: Auto (Try Volcano Engine first, fallback to Whisper)**
```json
{
    "asr_engine": "auto",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "access_key:secret_key",
    "whisper_mode": "local",
    "whisper_model": "medium",
    "generate_srt": true
}
```

2. **Extract and Transcribe**

Run the script with the local file path:
```bash
python "<skill_path>/video_subtitle.py" "/path/to/video.mp4"
```

The script will:
- Extract audio using ffmpeg
- Transcribe with selected ASR engine
- Generate SRT subtitle file (if enabled)
- Extract keyframes if enabled
- Output JSON with transcript and metadata

3. **Generate Summary**

Same as online videos - I will read the JSON output and generate structured notes.

### For Online Videos

### Step 1: Install Dependencies (if needed)

Check the video URL to determine the platform. If it's NOT a Bilibili URL, ensure the required package is installed:

```bash
pip install youtube-transcript-api yt-dlp
```

If these are already installed, skip this step.

### Step 2: Extract Subtitles

Run the extraction script with the video URL:

```bash
python "<skill_path>/video_subtitle.py" "<VIDEO_URL>"
```

Replace `<skill_path>` with the absolute path to this skill's directory, and `<VIDEO_URL>` with the video URL provided by the user.

The script automatically:
1. Detects the platform from the URL
2. Tries platform-specific subtitle APIs (fast, free)
3. Falls back to yt-dlp subtitle extraction if needed
4. Falls back to Whisper audio transcription if configured

The script outputs a JSON object to stdout containing:
- `title` - Video title
- `author` - Uploader name
- `duration` - Video duration
- `description` - Video description
- `platform` - Detected platform: `bilibili`, `youtube`, `douyin`, `xiaohongshu`, `tiktok`, or `generic`
- `url` - Canonical video URL
- `source` - Extraction method: `subtitle`, `ai_conclusion`, `transcript_api`, `yt_dlp_subs`, `whisper_local`, `whisper_api`
- `subtitle_text` - Full subtitle/transcript text (if available)
- `frames` - Array of keyframe screenshots (if ffmpeg is installed and `extract_frames` is enabled in config.json). Each frame has `path` (absolute file path) and `timestamp` (e.g. "01:30")
- `error` - Error message (if extraction failed)

### Step 3: Handle Errors

If the output contains an `error` field:

1. Check if the required dependencies are installed for that platform
2. If the error mentions missing packages, install them and retry
3. If Whisper is not enabled and no subtitles were found, suggest the user enable Whisper in `config.json`
4. After fixing, retry the extraction

### Step 4: Generate Structured Notes (BibiGPT Style)

When subtitle text is successfully extracted, summarize it into this BibiGPT-style format.
Use the EXACT structure below, including emojis, section naming, and formatting.

**If `frames` array is present in the JSON output**, embed one screenshot per content section using the Read tool to view the frame image file, then insert it with markdown image syntax right after the section header. Match frames to sections by timestamp order -- assign one frame per section sequentially. If there are more sections than frames, some sections will have no image. If there are more frames than sections, distribute evenly.

```markdown
# AI 一键总结：[{title}]({url})

# 🤖 {title} — 通俗解释


### 🏷️ {Section 1 Title}
![{Section 1 Title}]({frame_path_for_section_1})
- {Key point from transcript, preserving original examples and analogies}
- {Another key point}
*   {Sub-point or example}
*   {Sub-point or example}


### 💡 {Section 2 Title}
![{Section 2 Title}]({frame_path_for_section_2})
- {Content organized by topic}
- {Preserve vivid analogies from the video}


### 🧠 {Section 3 Title}
![{Section 3 Title}]({frame_path_for_section_3})
- {Continue grouping content logically}


### 🍎 {Section 4 Title}
![{Section 4 Title}]({frame_path_for_section_4})
- {More content sections as needed}


(... more sections using rotating emojis: 🏷️ 💡 🧠 🍎 🔑 🔢 🧱 🎯 ...)


### Summary
- {One paragraph summarizing the entire video content concisely}


### Highlights
*   🧠 {Highlight 1 with emoji} [#tag1] [#tag2] [#tag3]
*   🔪 {Highlight 2 with emoji} [#tag1] [#tag2] [#tag3]
*   🧮 {Highlight 3 with emoji} [#tag1] [#tag2] [#tag3]
*   🔢 {Highlight 4 with emoji} [#tag1] [#tag2] [#tag3]
*   🧱 {Highlight 5 with emoji} [#tag1] [#tag2] [#tag3]


[#tag1] [#tag2] [#tag3] [#tag4] [#tag5]


### Questions
*   {Thought-provoking question 1 related to the video content}
*   {Thought-provoking question 2 that extends the topic}
```

Guidelines for summarization:
- Use the original language of the subtitle (Chinese for Chinese videos, English for English videos, etc.)
- Use emoji-prefixed section headers (### 🏷️, ### 💡, ### 🧠, etc.)
- Use `-` for main bullet points and `*` (with 4 spaces indent) for sub-points/examples
- Group content into logical sections based on topic flow
- Preserve vivid examples, analogies and metaphors from the video
- Keep the summary concise but comprehensive
- Generate 5 highlights with emojis and 3 hashtags each
- Generate 2 thought-provoking follow-up questions
- Add 5 relevant hashtag topics at the end
- If the source is "ai_conclusion", note that the summary comes from B站's built-in AI summary feature
- If the source is "whisper_local", "whisper_api", or "volcengine", note that the transcript was generated via speech recognition and may contain minor inaccuracies
- **Keyframe images**: When frames are provided, embed them using `![title](absolute_path)` syntax. Each section should have at most one image, placed directly after the `### ` header line. Use the absolute `path` value from the frames array as-is

## ASR Engine Configuration

### Whisper Setup

For videos without subtitles, Whisper can transcribe the audio. Edit `config.json` in this skill's directory:

**Option A -- OpenAI Whisper API** (fast, requires API key, costs ~$0.006/min):
```json
{
    "asr_engine": "whisper",
    "whisper_mode": "api",
    "openai_api_key": "sk-your-key-here",
    "language": "zh",
    "generate_srt": true
}
```

**Option B -- Local faster-whisper** (free, requires model download ~1-3GB):
```json
{
    "asr_engine": "whisper",
    "whisper_mode": "local",
    "whisper_model": "medium",
    "whisper_device": "cuda",
    "whisper_compute_type": "float16",
    "language": "zh",
    "generate_srt": true
}
```

Model sizes: `tiny` (fast, less accurate) / `base` (balanced) / `small` / `medium` / `large` (slow, most accurate).

### Volcano Engine Setup (火山引擎)

For high-quality Chinese transcription, use Volcano Engine ASR. See [VOLCENGINE.md](VOLCENGINE.md) for detailed setup instructions.

**Quick setup:**
```json
{
    "asr_engine": "volcengine",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "access_key:secret_key",
    "volcengine_language": "zh-CN",
    "generate_srt": true,
    "srt_output_dir": "subtitles"
}
```

**Auto mode** (try Volcano Engine first, fallback to Whisper):
```json
{
    "asr_engine": "auto",
    "volcengine_appid": "your_app_id",
    "volcengine_token": "access_key:secret_key",
    "whisper_mode": "local",
    "whisper_model": "medium",
    "generate_srt": true
}
```

### SRT Subtitle Generation

When `generate_srt` is enabled, the script will automatically generate SRT subtitle files with timestamps:

```json
{
    "generate_srt": true,
    "srt_output_dir": "subtitles"
}
```

- SRT files are saved to the `srt_output_dir` directory (relative to skill directory or absolute path)
- Filename matches the video title (for online videos) or video filename (for local files)
- Only works with ASR engines that return segment timestamps (Whisper local mode, Volcano Engine)
- Whisper API mode does not support SRT generation (no timestamps)

## Markdown Summary Generation

When `generate_markdown` is enabled, the script will automatically generate a detailed Markdown summary file:

```json
{
    "generate_markdown": true,
    "markdown_output_dir": "summaries",
    "markdown_style": "detailed",
    "publish_dir": "D:\\Documents\\Code\\Video summarize"
}
```

- `generate_markdown`: `true` to auto-generate .md files, `false` to skip (default: `false`)
- `markdown_output_dir`: directory for saving summary files (default: `summaries`)
- `markdown_style`: output style
  - `"detailed"` - full paragraph descriptions with context and explanations (recommended)
  - `"bullet"` - concise bullet points (BibiGPT style)
- `publish_dir`: optional directory for publishing final summaries (default: none)

### File Naming and Organization

**For online videos:**
- Markdown filename format: `YYYYMMDD-视频标题.md` (e.g., `20260417-婚礼化妆避坑建议.md`)
- Files are saved directly to `publish_dir` if configured, otherwise to `markdown_output_dir`

**For local video files:**
- A subfolder is created using the video filename (without extension)
- Markdown, SRT subtitles, and screenshots are all saved to this subfolder
- Example structure:
  ```
  D:\Documents\Code\Video summarize\
  └── 婚礼化妆避坑建议\
      ├── 20260417-婚礼化妆避坑建议.md
      ├── 婚礼化妆避坑建议.srt
      └── screenshots\
          ├── frame_001.jpg
          ├── frame_002.jpg
          └── ...
  ```

The generated Markdown file includes:
- Video title and metadata
- Detailed content summary with embedded keyframe screenshots
- Section-by-section breakdown
- Key highlights with hashtags
- Thought-provoking questions

## Douyin / Xiaohongshu / TikTok Cookie Setup

These platforms require browser cookies for yt-dlp to access video content. The script tries `--cookies-from-browser` automatically, but on Windows with Chrome 127+ this often fails due to DPAPI encryption.

**Recommended: export cookies.txt manually**

1. Install the browser extension "[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)"
2. Open `douyin.com` (or `xiaohongshu.com`) in your browser (login is NOT required, just visit the page)
3. Click the extension icon and export cookies
4. Save the file as `cookies.txt` in this skill's directory (`<skill_path>/cookies.txt`)

The script will automatically detect and use this file for Douyin/Xiaohongshu/TikTok requests.

## Keyframe Screenshots (Optional)

The script can extract keyframe screenshots from videos to embed in the summary. This requires `ffmpeg` to be installed.

To enable/disable, edit `config.json`:
```json
{
    "extract_frames": true,
    "frames_per_video": 6
}
```

- `extract_frames`: `true` (default) to capture keyframes, `false` to skip
- `frames_per_video`: number of evenly-spaced frames to extract (default `6`)

Screenshots are cached in the `screenshots/` directory. If ffmpeg is not installed, frame extraction is silently skipped.
