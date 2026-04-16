#!/usr/bin/env python3
"""
Multi-Platform Video Subtitle Extractor
Extracts subtitles/transcripts from multiple video platforms for AI summarization.

Supported platforms:
  - Bilibili (public API with WBI signing)
  - YouTube (youtube-transcript-api or yt-dlp)
  - Douyin / TikTok (yt-dlp)
  - Xiaohongshu (yt-dlp)
  - Any yt-dlp supported site (1800+ sites)

Fallback chain:
  1. Platform-specific subtitle API (free, no auth)
  2. yt-dlp subtitle extraction
  3. yt-dlp audio download + Whisper ASR (local or API)
"""

import glob
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from functools import reduce

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    # Add CUDA 12 DLL paths to PATH for GPU acceleration
    try:
        import site
        for site_packages in site.getsitepackages():
            cuda_paths = [
                os.path.join(site_packages, "nvidia", "cublas", "bin"),
                os.path.join(site_packages, "nvidia", "cudnn", "bin"),
                os.path.join(site_packages, "nvidia", "cuda_nvrtc", "bin"),
                os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
            ]
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path) and cuda_path not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass  # Silently ignore if CUDA paths not found

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "screenshots")
COOKIES_PATHS = [
    os.path.join(SCRIPT_DIR, "cookies.txt"),
    os.path.join(SCRIPT_DIR, "www.douyin.com_cookies.txt"),
    os.path.join(SCRIPT_DIR, "www.xiaohongshu.com_cookies.txt"),
    os.path.join(SCRIPT_DIR, "www.tiktok.com_cookies.txt"),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Result cache (avoids re-downloading audio / re-running Whisper)
# ---------------------------------------------------------------------------

def _cache_key(url):
    """Normalize URL and produce a stable hash for caching."""
    normalized = re.sub(r'[?&](vd_source|spm_id_from|from|seid)=[^&]*', '', url)
    normalized = normalized.rstrip('/?')
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _read_cache(url):
    key = _cache_key(url)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        ttl_days = load_config().get("cache_ttl_days", 7)
        if ttl_days > 0:
            age_days = (time.time() - os.path.getmtime(path)) / 86400
            if age_days > ttl_days:
                log(f"Cache expired ({age_days:.1f}d > {ttl_days}d): {key}")
                return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("subtitle_text"):
                log(f"Cache hit: {key}")
                data["_cached"] = True
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _write_cache(url, result):
    """Cache a successful extraction result."""
    if not result or not result.get("subtitle_text"):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(url)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log(f"Cached result: {key}")
    except IOError as e:
        log(f"Failed to write cache: {e}", "WARN")


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

PLATFORM_PATTERNS = [
    ("bilibili", [
        r"bilibili\.com/video/",
        r"b23\.tv/",
        r"^BV[a-zA-Z0-9]+$",
    ]),
    ("youtube", [
        r"youtube\.com/watch",
        r"youtube\.com/shorts/",
        r"youtu\.be/",
        r"youtube\.com/live/",
    ]),
    ("douyin", [
        r"douyin\.com/",
        r"v\.douyin\.com/",
        r"iesdouyin\.com/",
    ]),
    ("xiaohongshu", [
        r"xiaohongshu\.com/",
        r"xhslink\.com/",
    ]),
    ("tiktok", [
        r"tiktok\.com/",
    ]),
]


def detect_platform(url):
    for platform, patterns in PLATFORM_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, url):
                return platform
    return "generic"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def http_get(url, headers=None, raw_bytes=False):
    hdrs = dict(DEFAULT_HEADERS)
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
        if data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)
        if raw_bytes:
            return data
        return data.decode("utf-8")


def api_request(url, params=None, headers=None):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        text = http_get(url, headers)
        return json.loads(text)
    except urllib.error.HTTPError as e:
        log(f"HTTP {e.code}: {url}", "ERROR")
    except urllib.error.URLError as e:
        log(f"URL error: {e.reason}", "ERROR")
    except Exception as e:
        log(f"Request failed: {e}", "ERROR")
    return None


def log(msg, level="INFO"):
    print(f"[{level}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Bilibili extractor (public API, no cookies)
# ---------------------------------------------------------------------------

BILI_HEADERS = {
    **DEFAULT_HEADERS,
    "Referer": "https://www.bilibili.com",
    "Origin": "https://www.bilibili.com",
}

API_VIDEO_VIEW = "https://api.bilibili.com/x/web-interface/view"
API_PLAYER_V2 = "https://api.bilibili.com/x/player/wbi/v2"
API_NAV = "https://api.bilibili.com/x/web-interface/nav"
API_CONCLUSION = "https://api.bilibili.com/x/web-interface/view/conclusion/get"

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]


def _get_mixin_key(orig):
    return reduce(lambda s, i: s + orig[i], MIXIN_KEY_ENC_TAB, "")[:32]


def _get_wbi_keys():
    try:
        data = api_request(API_NAV, headers=BILI_HEADERS)
        if data and data.get("code") == 0:
            wbi_img = data["data"]["wbi_img"]
            img_key = wbi_img["img_url"].rsplit("/", 1)[1].split(".")[0]
            sub_key = wbi_img["sub_url"].rsplit("/", 1)[1].split(".")[0]
            return img_key, sub_key
    except Exception as e:
        log(f"Failed to get WBI keys: {e}", "WARN")
    return None, None


def _sign_wbi(params, img_key, sub_key):
    mixin_key = _get_mixin_key(img_key + sub_key)
    params["wts"] = int(time.time())
    params = dict(sorted(params.items()))
    params = {
        k: "".join(c for c in str(v) if c not in "!'()*")
        for k, v in params.items()
    }
    query = urllib.parse.urlencode(params)
    params["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params


def _extract_bvid(url_or_bvid):
    if re.match(r"^BV[a-zA-Z0-9]+$", url_or_bvid):
        return url_or_bvid
    for pattern in [
        r"bilibili\.com/video/(BV[a-zA-Z0-9]+)",
        r"b23\.tv/(BV[a-zA-Z0-9]+)",
        r"(BV[a-zA-Z0-9]{10})",
    ]:
        match = re.search(pattern, url_or_bvid)
        if match:
            return match.group(1)
    return None


def _bili_parse_subtitle_list(subtitles):
    urls = []
    for sub in subtitles:
        sub_url = sub.get("subtitle_url", "")
        if sub_url:
            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url
            urls.append({
                "url": sub_url,
                "lang": sub.get("lan", "unknown"),
                "lang_doc": sub.get("lan_doc", "unknown"),
            })
    return urls


def _bili_download_subtitle(url):
    try:
        text = http_get(url, headers=BILI_HEADERS)
        data = json.loads(text)
        body = data.get("body", [])
        if not body:
            return None
        lines = [item.get("content", "").strip() for item in body]
        return "\n".join(line for line in lines if line)
    except Exception as e:
        log(f"Failed to download subtitle: {e}", "ERROR")
        return None


def _bili_pick_and_download(subtitle_urls):
    preferred_langs = ["zh-CN", "zh-Hans", "ai-zh", "zh"]
    for lang in preferred_langs:
        for sub in subtitle_urls:
            if lang in sub["lang"]:
                text = _bili_download_subtitle(sub["url"])
                if text:
                    return text, sub["lang_doc"]
    for sub in subtitle_urls:
        text = _bili_download_subtitle(sub["url"])
        if text:
            return text, sub["lang_doc"]
    return None, None


def extract_bilibili(url):
    bvid = _extract_bvid(url)
    if not bvid:
        return None

    data = api_request(API_VIDEO_VIEW, params={"bvid": bvid}, headers=BILI_HEADERS)
    if not data or data.get("code") != 0:
        log(f"Failed to get video info: {data}", "ERROR")
        return None

    video = data["data"]
    info = {
        "title": video.get("title", ""),
        "author": video.get("owner", {}).get("name", ""),
        "duration": video.get("duration", 0),
        "description": video.get("desc", ""),
    }
    cid = video.get("cid")
    aid = video.get("aid")
    mid = video.get("owner", {}).get("mid")
    bvid = video.get("bvid", bvid)

    log(f"Bilibili video: {info['title']} (aid={aid}, cid={cid})")

    if not cid:
        log("Could not find cid", "ERROR")
        return _make_result(info, "bilibili", url, error="Could not find cid for this video")

    # Method 1: subtitle list from view API
    subtitle_urls = _bili_parse_subtitle_list(
        video.get("subtitle", {}).get("list", [])
    )

    # Method 2: player v2 API with WBI signing
    if not subtitle_urls:
        log("Trying player v2 API with WBI signing...")
        params = {"bvid": bvid, "cid": cid}
        img_key, sub_key = _get_wbi_keys()
        if img_key and sub_key:
            params = _sign_wbi(params, img_key, sub_key)
        resp = api_request(API_PLAYER_V2, params=params, headers=BILI_HEADERS)
        if resp and resp.get("code") == 0:
            subtitle_urls = _bili_parse_subtitle_list(
                resp.get("data", {}).get("subtitle", {}).get("subtitles", [])
            )

    # Method 3: page HTML scraping
    if not subtitle_urls:
        log("Trying page HTML scraping...")
        try:
            html = http_get(f"https://www.bilibili.com/video/{bvid}/", headers=BILI_HEADERS)
            for match in re.findall(r'"subtitle_url"\s*:\s*"(//[^"]+)"', html):
                sub_url = "https:" + match
                if not any(u["url"] == sub_url for u in subtitle_urls):
                    subtitle_urls.append({"url": sub_url, "lang": "zh-CN", "lang_doc": "中文（自动生成）"})
            for block in re.findall(r'"subtitles"\s*:\s*\[(\{.*?\})\]', html, re.DOTALL):
                try:
                    for sub in json.loads(f"[{block}]"):
                        sub_url = sub.get("subtitle_url", "")
                        if sub_url:
                            if sub_url.startswith("//"):
                                sub_url = "https:" + sub_url
                            if not any(u["url"] == sub_url for u in subtitle_urls):
                                subtitle_urls.append({
                                    "url": sub_url,
                                    "lang": sub.get("lan", "unknown"),
                                    "lang_doc": sub.get("lan_doc", "unknown"),
                                })
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            log(f"Page scraping failed: {e}", "WARN")

    # Method 4: multi-page video
    if not subtitle_urls:
        pages = video.get("pages", [])
        if len(pages) > 1:
            for page in pages:
                page_cid = page.get("cid")
                if page_cid and page_cid != cid:
                    params = {"bvid": bvid, "cid": page_cid}
                    img_key, sub_key = _get_wbi_keys()
                    if img_key and sub_key:
                        params = _sign_wbi(params, img_key, sub_key)
                    resp = api_request(API_PLAYER_V2, params=params, headers=BILI_HEADERS)
                    if resp and resp.get("code") == 0:
                        subtitle_urls = _bili_parse_subtitle_list(
                            resp.get("data", {}).get("subtitle", {}).get("subtitles", [])
                        )
                        if subtitle_urls:
                            break

    # Download best subtitle
    if subtitle_urls:
        text, lang = _bili_pick_and_download(subtitle_urls)
        if text:
            return _make_result(info, "bilibili", url, text, "subtitle")

    # Method 5: B站 AI conclusion API
    if aid and cid and mid:
        log("Trying B站 AI conclusion API...")
        params = {"aid": aid, "cid": cid, "up_mid": mid}
        img_key, sub_key = _get_wbi_keys()
        if img_key and sub_key:
            params = _sign_wbi(params, img_key, sub_key)
        resp = api_request(API_CONCLUSION, params=params, headers=BILI_HEADERS)
        if resp and resp.get("code") == 0:
            model_result = resp.get("data", {}).get("model_result", {})
            if model_result:
                parts = []
                summary = model_result.get("summary", "")
                if summary:
                    parts.append(summary)
                for section in model_result.get("outline", []):
                    title = section.get("title", "")
                    if title:
                        parts.append(f"\n## {title}")
                    for kp in section.get("key_point", []):
                        content = kp.get("content", "")
                        if content:
                            parts.append(f"- {content}")
                if parts:
                    return _make_result(info, "bilibili", url, "\n".join(parts), "ai_conclusion")

    return _make_result(info, "bilibili", url)


# ---------------------------------------------------------------------------
# YouTube extractor
# ---------------------------------------------------------------------------

def _extract_youtube_id(url):
    patterns = [
        r"youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/live/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_youtube(url):
    video_id = _extract_youtube_id(url)
    if not video_id:
        return None

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    info = {"title": "", "author": "", "duration": 0, "description": ""}

    # Try youtube-transcript-api first (lightweight, no deps beyond pip)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        log("Using youtube-transcript-api...")
        try:
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.list(video_id)
            transcript = None
            for lang in ["zh-Hans", "zh-CN", "zh", "zh-Hant", "en"]:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except Exception:
                    continue
            if not transcript:
                try:
                    transcript = transcript_list.find_generated_transcript(["zh-Hans", "zh-CN", "zh", "en"])
                except Exception:
                    transcripts = list(transcript_list)
                    if transcripts:
                        transcript = transcripts[0]

            if transcript:
                fetched = transcript.fetch()
                lines = []
                for entry in fetched:
                    if isinstance(entry, dict):
                        lines.append(entry.get("text", "").strip())
                    elif hasattr(entry, "text"):
                        lines.append(entry.text.strip())
                    else:
                        lines.append(str(entry).strip())
                text = "\n".join(line for line in lines if line)
                if text:
                    _fill_youtube_info(info, video_id)
                    return _make_result(info, "youtube", canonical_url, text, "transcript_api")
        except Exception as e:
            log(f"youtube-transcript-api failed: {e}", "WARN")
    except ImportError:
        log("youtube-transcript-api not installed, will try yt-dlp", "WARN")

    _fill_youtube_info(info, video_id)
    return _make_result(info, "youtube", canonical_url)


def _fill_youtube_info(info, video_id):
    """Best-effort fill of YouTube video metadata via oembed (no API key needed)."""
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        data = json.loads(http_get(oembed_url))
        info["title"] = data.get("title", "")
        info["author"] = data.get("author_name", "")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Douyin extractor (fetches page HTML to extract video description)
# ---------------------------------------------------------------------------

DOUYIN_HEADERS = {
    **DEFAULT_HEADERS,
    "Referer": "https://www.douyin.com",
    "Cookie": "s_v_web_id=verify_placeholder",
}

DOUYIN_MOBILE_UA = (
    "com.ss.android.ugc.aweme/110101 "
    "(Linux; U; Android 12; en_US; Pixel 6; Build/SD1A.210817.036; "
    "Cronet/TTNetVersion:b4d74d15 2023-04-08)"
)

AUTO_COOKIES_PATH = os.path.join(SCRIPT_DIR, "_auto_douyin_cookies.txt")


def _fetch_fresh_douyin_cookies():
    """Generate fresh Douyin cookies (s_v_web_id, ttwid, etc.) required by yt-dlp.
    s_v_web_id is a client-side cookie generated by JS — we synthesize it here.
    ttwid is obtained by hitting douyin.com and collecting the Set-Cookie header.
    Writes Netscape cookie-jar format for yt-dlp. Returns file path or None."""
    import http.cookiejar
    import secrets
    import time as _time

    try:
        cj = http.cookiejar.MozillaCookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj),
        )
        req = urllib.request.Request("https://www.douyin.com/", headers={
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        try:
            opener.open(req, timeout=15).read()
        except Exception:
            pass

        existing_names = {c.name for c in cj}

        if "s_v_web_id" not in existing_names:
            verify_id = f"verify_{secrets.token_hex(16)}"
            cj.set_cookie(http.cookiejar.Cookie(
                version=0, name="s_v_web_id", value=verify_id,
                port=None, port_specified=False,
                domain=".douyin.com", domain_specified=True, domain_initial_dot=True,
                path="/", path_specified=True,
                secure=True, expires=int(_time.time()) + 86400 * 30,
                discard=False, comment=None, comment_url=None,
                rest={"HttpOnly": None},
            ))

        if "ttwid" not in existing_names:
            try:
                ttwid_req = urllib.request.Request(
                    "https://ttwid.bytedance.com/ttwid/union/register/",
                    data=json.dumps({
                        "region": "cn",
                        "aid": 1128,
                        "needFid": False,
                        "service": "www.ixigua.com",
                        "migrate_info": {"ticket": "", "source": "node"},
                        "cbUrlProtocol": "https",
                        "union": True,
                    }).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": DEFAULT_HEADERS["User-Agent"],
                    },
                    method="POST",
                )
                resp = urllib.request.urlopen(ttwid_req, timeout=10)
                for header_val in resp.headers.get_all("Set-Cookie") or []:
                    if "ttwid=" in header_val:
                        ttwid_val = header_val.split("ttwid=")[1].split(";")[0]
                        cj.set_cookie(http.cookiejar.Cookie(
                            version=0, name="ttwid", value=ttwid_val,
                            port=None, port_specified=False,
                            domain=".douyin.com", domain_specified=True, domain_initial_dot=True,
                            path="/", path_specified=True,
                            secure=True, expires=int(_time.time()) + 86400 * 30,
                            discard=False, comment=None, comment_url=None,
                            rest={"HttpOnly": None},
                        ))
            except Exception as e:
                log(f"ttwid fetch failed (non-fatal): {e}", "WARN")

        cj.save(AUTO_COOKIES_PATH, ignore_discard=True, ignore_expires=True)
        names = [c.name for c in cj]
        log(f"Auto-generated {len(names)} Douyin cookies: {', '.join(names)}")
        return AUTO_COOKIES_PATH
    except Exception as e:
        log(f"Auto-fetch Douyin cookies failed: {e}", "WARN")
        return None


def _resolve_douyin_url(url):
    """Resolve v.douyin.com short link to full URL, extract video ID."""
    try:
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resolved = resp.url
        match = re.search(r'douyin\.com/video/(\d+)', resolved)
        if match:
            return match.group(1), resolved
    except Exception as e:
        log(f"Failed to resolve Douyin URL: {e}", "WARN")
    match = re.search(r'douyin\.com/video/(\d+)', url)
    if match:
        return match.group(1), url
    return None, url


def _douyin_share_api(video_id):
    """Fetch video metadata from iesdouyin.com share page (mobile UA, no login).
    Returns (info_dict, play_url) or (None, None)."""
    share_url = f"https://www.iesdouyin.com/share/video/{video_id}/"
    try:
        req = urllib.request.Request(share_url, headers={
            "User-Agent": DOUYIN_MOBILE_UA,
            "Accept": "*/*",
        })
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")

        m = re.search(r"_ROUTER_DATA\s*=\s*", html)
        if not m:
            return None, None
        json_start = m.end()
        script_end = html.find("</script>", json_start)
        if script_end == -1:
            return None, None
        raw_json = html[json_start:script_end].strip()

        router_data = json.loads(raw_json)
        loader = router_data.get("loaderData", {})
        item = None
        for v in loader.values():
            if not isinstance(v, dict):
                continue
            video_res = v.get("videoInfoRes") or v
            items = video_res.get("item_list", [])
            if items:
                item = items[0]
                break
        if not item:
            return None, None

        info = {
            "title": item.get("desc", ""),
            "author": (item.get("author") or {}).get("nickname", ""),
            "duration": (item.get("video") or {}).get("duration", 0),
            "description": item.get("desc", ""),
        }
        dur = info["duration"]
        if isinstance(dur, (int, float)) and dur > 10000:
            info["duration"] = dur / 1000.0

        play_url = None
        video_obj = item.get("video", {})
        for addr_key in ("play_addr", "play_addr_h264", "download_addr"):
            addr = video_obj.get(addr_key, {})
            if isinstance(addr, dict):
                urls = addr.get("url_list", [])
                if urls:
                    play_url = urls[0]
                    break

        return info, play_url
    except Exception as e:
        log(f"Douyin share API failed: {e}", "WARN")
    return None, None


def _download_douyin_audio(play_url, tmp_dir):
    """Download Douyin video (as audio source) directly from play_url.
    Returns the saved file path or None."""
    if not play_url:
        return None
    try:
        audio_path = os.path.join(tmp_dir, "douyin_audio.mp4")
        req = urllib.request.Request(play_url, headers={
            "User-Agent": DOUYIN_MOBILE_UA,
            "Referer": "https://www.douyin.com/",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(audio_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            log(f"Downloaded Douyin video: {os.path.getsize(audio_path) // 1024}KB")
            return audio_path
    except Exception as e:
        log(f"Douyin direct download failed: {e}", "WARN")
    return None



# ---------------------------------------------------------------------------
# Xiaohongshu extractor (direct page parsing, no cookies needed)
# ---------------------------------------------------------------------------

XHS_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 "
    "Mobile/15E148 Safari/604.1"
)


def _resolve_xhs_url(url):
    """Resolve xhslink.com short link and extract note ID."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": XHS_MOBILE_UA,
            "Accept": "text/html,*/*",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            resolved = resp.url
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        log(f"Failed to resolve XHS URL: {e}", "WARN")
        return None, None, url

    for pat in [
        r"/discovery/item/([a-f0-9]+)",
        r"/explore/([a-f0-9]+)",
        r"noteId[\"=:]\"?([a-f0-9]{24})",
    ]:
        m = re.search(pat, resolved)
        if m:
            return m.group(1), html, resolved

    return None, html, resolved


def _parse_xhs_page(html):
    """Parse XHS page HTML for note data and video URLs from __SETUP_SERVER_STATE__."""
    m = re.search(r"window\.__SETUP_SERVER_STATE__\s*=\s*", html)
    if not m:
        return None, None

    json_start = m.end()
    script_end = html.find("</script>", json_start)
    if script_end == -1:
        return None, None

    raw = html[json_start:script_end].strip()
    if raw.endswith(";"):
        raw = raw[:-1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, None

    note_data = data.get("LAUNCHER_SSR_STORE_PAGE_DATA", {}).get("noteData", {})
    if not note_data:
        return None, None

    user = note_data.get("user", {})
    video_obj = note_data.get("video", {})
    media = video_obj.get("media", {})
    capa = video_obj.get("capa", {})
    vid = media.get("video", {})

    duration = capa.get("duration") or vid.get("duration", 0)
    if isinstance(duration, (int, float)) and duration > 10000:
        duration = duration / 1000.0

    info = {
        "title": note_data.get("title", ""),
        "author": user.get("nickName", ""),
        "duration": duration,
        "description": note_data.get("desc", ""),
    }

    play_url = None
    stream = media.get("stream", {})
    for codec in ("h264", "h265", "av1", "h266"):
        streams = stream.get(codec, [])
        if streams and isinstance(streams, list):
            best = streams[0]
            play_url = best.get("masterUrl")
            if play_url:
                break
            backup = best.get("backupUrls", [])
            if backup:
                play_url = backup[0]
                break

    return info, play_url


def extract_xiaohongshu(url):
    """Extract Xiaohongshu video info via mobile page HTML (no cookies needed)."""
    note_id, html, resolved_url = _resolve_xhs_url(url)

    if html:
        info, play_url = _parse_xhs_page(html)
        if info:
            log(f"XHS note: {info['title']} (duration={info['duration']}s)")
            result = _make_result(info, "xiaohongshu", resolved_url)
            result["_play_url"] = play_url
            return result

    return _make_result(
        {"title": "", "author": "", "duration": 0, "description": ""},
        "xiaohongshu", resolved_url,
    )


def _download_xhs_video(play_url, tmp_dir):
    """Download XHS video directly from CDN URL. Returns file path or None."""
    if not play_url:
        return None
    try:
        video_path = os.path.join(tmp_dir, "xhs_video.mp4")
        req = urllib.request.Request(play_url, headers={
            "User-Agent": XHS_MOBILE_UA,
            "Referer": "https://www.xiaohongshu.com/",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(video_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
            log(f"Downloaded XHS video: {os.path.getsize(video_path) // 1024}KB")
            return video_path
    except Exception as e:
        log(f"XHS direct download failed: {e}", "WARN")
    return None


def extract_douyin(url):
    """Extract Douyin video info via the iesdouyin share page (no cookies needed).
    Returns result dict with info (and optionally subtitle_text if found)."""
    video_id, resolved_url = _resolve_douyin_url(url)
    if not video_id:
        return None

    info, play_url = _douyin_share_api(video_id)
    if not info:
        info = {"title": "", "author": "", "duration": 0, "description": ""}

    result = _make_result(info, "douyin", resolved_url)
    result["_play_url"] = play_url
    return result


# ---------------------------------------------------------------------------
# yt-dlp based extractor (generic, works for all yt-dlp supported sites)
# ---------------------------------------------------------------------------

PLATFORMS_NEEDING_COOKIES = {"douyin", "xiaohongshu", "tiktok"}


def _get_ytdlp_cmd():
    """Return the command prefix for yt-dlp. Tries the binary first, then python -m."""
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    try:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True, timeout=10,
        )
        return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        return None


def _check_ytdlp():
    return _get_ytdlp_cmd() is not None


def _run_ytdlp(args, timeout=60):
    """Run yt-dlp and return (returncode, stdout_str). Handles Windows encoding.
    args[0] should be 'yt-dlp'; it will be replaced with the correct command."""
    cmd = _get_ytdlp_cmd()
    if cmd is None:
        return 1, "yt-dlp not found"
    actual_args = cmd + list(args[1:])
    result = subprocess.run(
        actual_args, capture_output=True, timeout=timeout,
    )
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    return result.returncode, stdout


def _detect_browser():
    """Detect an available browser for --cookies-from-browser.
    Checks common browser data directories on Windows/Mac/Linux."""
    home = os.path.expanduser("~")
    browser_paths = {
        "chrome": [
            os.path.join(home, "AppData", "Local", "Google", "Chrome", "User Data"),
            os.path.join(home, ".config", "google-chrome"),
            os.path.join(home, "Library", "Application Support", "Google", "Chrome"),
        ],
        "edge": [
            os.path.join(home, "AppData", "Local", "Microsoft", "Edge", "User Data"),
        ],
        "firefox": [
            os.path.join(home, "AppData", "Roaming", "Mozilla", "Firefox", "Profiles"),
            os.path.join(home, ".mozilla", "firefox"),
            os.path.join(home, "Library", "Application Support", "Firefox", "Profiles"),
        ],
        "brave": [
            os.path.join(home, "AppData", "Local", "BraveSoftware", "Brave-Browser", "User Data"),
        ],
    }
    for browser, paths in browser_paths.items():
        for p in paths:
            if os.path.isdir(p):
                return browser
    return None


_cached_browser = None
_cookie_args_tested = False
_cookie_args_result = []


def _get_cookie_args(platform):
    """Return extra yt-dlp args for platforms that need browser cookies.
    Priority: user cookies file > auto-fetched cookies > --cookies-from-browser.
    Tests once whether the chosen method actually works."""
    global _cached_browser, _cookie_args_tested, _cookie_args_result
    if platform not in PLATFORMS_NEEDING_COOKIES:
        return []
    if _cookie_args_tested:
        return _cookie_args_result

    _cookie_args_tested = True

    # Priority 1: user-provided cookies file in skill directory
    for cp in COOKIES_PATHS:
        if os.path.isfile(cp):
            log(f"Using cookies file: {os.path.basename(cp)}")
            _cookie_args_result = ["--cookies", cp]
            return _cookie_args_result
    for f in glob.glob(os.path.join(SCRIPT_DIR, "*cookies*.txt")):
        if os.path.isfile(f) and "_auto_" not in os.path.basename(f):
            log(f"Using cookies file: {os.path.basename(f)}")
            _cookie_args_result = ["--cookies", f]
            return _cookie_args_result

    # Priority 2: auto-fetch fresh cookies from the platform
    if platform == "douyin":
        auto_path = _fetch_fresh_douyin_cookies()
        if auto_path:
            _cookie_args_result = ["--cookies", auto_path]
            return _cookie_args_result

    # Priority 3: --cookies-from-browser
    _cached_browser = _detect_browser() or ""
    if _cached_browser:
        try:
            rc, _ = _run_ytdlp(
                ["yt-dlp", "--cookies-from-browser", _cached_browser,
                 "--dump-json", "--no-download", "https://www.douyin.com/"],
                timeout=15,
            )
            if rc == 0:
                log(f"Using cookies from browser: {_cached_browser}")
                _cookie_args_result = ["--cookies-from-browser", _cached_browser]
                return _cookie_args_result
        except Exception:
            pass
        log(f"Browser cookie extraction failed ({_cached_browser})", "WARN")
    else:
        log("No browser detected for cookie extraction", "WARN")

    return []


def _ytdlp_extract_info(url, platform="generic"):
    """Use yt-dlp --dump-json to get video metadata without downloading."""
    try:
        cmd = ["yt-dlp", "--dump-json", "--no-download", "--no-playlist"]
        cmd += _get_cookie_args(platform)
        cmd.append(url)
        rc, stdout = _run_ytdlp(cmd)
        if rc == 0 and stdout.strip():
            return json.loads(stdout)
    except Exception as e:
        log(f"yt-dlp info extraction failed: {e}", "WARN")
    return None


def _ytdlp_extract_subs(url, tmp_dir, platform="generic"):
    """Use yt-dlp to write subtitle files without downloading the video."""
    try:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", "zh-Hans,zh-CN,zh,en,zh-Hant",
            "--sub-format", "json3/srv3/vtt/srt/best",
            "--no-playlist",
            "-o", os.path.join(tmp_dir, "%(id)s.%(ext)s"),
        ]
        cmd += _get_cookie_args(platform)
        cmd.append(url)
        _run_ytdlp(cmd)
    except Exception as e:
        log(f"yt-dlp subtitle download failed: {e}", "WARN")
        return None

    sub_files = (
        glob.glob(os.path.join(tmp_dir, "*.vtt"))
        + glob.glob(os.path.join(tmp_dir, "*.srt"))
        + glob.glob(os.path.join(tmp_dir, "*.json3"))
        + glob.glob(os.path.join(tmp_dir, "*.srv3"))
    )
    if not sub_files:
        return None

    for sf in sub_files:
        text = _parse_subtitle_file(sf)
        if text:
            return text
    return None


def _parse_subtitle_file(filepath):
    """Parse VTT/SRT/JSON3 subtitle file into plain text."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    if not content.strip():
        return None

    if ext == ".json3":
        return _parse_json3_subtitle(content)
    elif ext in (".vtt", ".srt"):
        return _parse_vtt_srt(content)
    elif ext == ".srv3":
        return _parse_srv3_subtitle(content)
    return None


def _parse_json3_subtitle(content):
    try:
        data = json.loads(content)
        events = data.get("events", [])
        lines = []
        for event in events:
            segs = event.get("segs", [])
            text = "".join(s.get("utf8", "") for s in segs).strip()
            if text and text != "\n":
                lines.append(text)
        return "\n".join(lines) if lines else None
    except Exception:
        return None


def _parse_srv3_subtitle(content):
    lines = []
    for match in re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL):
        text = re.sub(r'<[^>]+>', '', match).strip()
        if text:
            lines.append(text)
    return "\n".join(lines) if lines else None


def _parse_vtt_srt(content):
    lines = []
    seen = set()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'^WEBVTT', line):
            continue
        if re.match(r'^NOTE\s', line):
            continue
        if re.match(r'^\d{2}:\d{2}', line):
            continue
        if '-->' in line:
            continue
        text = re.sub(r'<[^>]+>', '', line).strip()
        if text and text not in seen:
            seen.add(text)
            lines.append(text)
    return "\n".join(lines) if lines else None


def _ytdlp_download_audio(url, tmp_dir, platform="generic"):
    """Download audio only via yt-dlp for Whisper transcription."""
    audio_path = os.path.join(tmp_dir, "audio.m4a")
    try:
        cmd = [
            "yt-dlp",
            "-f", "m4a/bestaudio[ext=m4a]/bestaudio",
            "--no-playlist",
            "-o", audio_path,
        ]
        cmd += _get_cookie_args(platform)
        cmd.append(url)
        rc, _ = _run_ytdlp(cmd, timeout=300)
        if rc == 0 and os.path.exists(audio_path):
            return audio_path
    except Exception as e:
        log(f"yt-dlp audio download failed: {e}", "WARN")

    found = glob.glob(os.path.join(tmp_dir, "audio.*"))
    return found[0] if found else None


def _ytdlp_download_video(url, tmp_dir, platform="generic"):
    """Download lowest-quality video via yt-dlp for frame extraction."""
    video_path = os.path.join(tmp_dir, "video.mp4")
    try:
        cmd = [
            "yt-dlp",
            "-f", "worstvideo[ext=mp4]+worstaudio/worst[ext=mp4]/worstvideo+worstaudio/worst",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", video_path,
        ]
        cmd += _get_cookie_args(platform)
        cmd.append(url)
        rc, out = _run_ytdlp(cmd, timeout=300)
        if rc == 0 and os.path.exists(video_path):
            return video_path
        log(f"yt-dlp video download returned rc={rc}", "WARN")
    except Exception as e:
        log(f"yt-dlp video download failed: {e}", "WARN")

    found = glob.glob(os.path.join(tmp_dir, "video.*"))
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Keyframe extraction
# ---------------------------------------------------------------------------

_ffmpeg_path_cache = None


def _get_ffmpeg():
    """Find ffmpeg executable. Caches after first lookup."""
    global _ffmpeg_path_cache
    if _ffmpeg_path_cache is not None:
        return _ffmpeg_path_cache or None

    path = shutil.which("ffmpeg")
    if path:
        _ffmpeg_path_cache = path
        return path

    if sys.platform == "win32":
        winget_links = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links"
        )
        candidate = os.path.join(winget_links, "ffmpeg.exe")
        if os.path.isfile(candidate):
            _ffmpeg_path_cache = candidate
            return candidate

    _ffmpeg_path_cache = ""
    return None


def _check_ffmpeg():
    return _get_ffmpeg() is not None


def _get_ffprobe():
    """Find ffprobe next to ffmpeg."""
    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        return None
    ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe" + (".exe" if sys.platform == "win32" else ""))
    if os.path.isfile(ffprobe):
        return ffprobe
    return shutil.which("ffprobe")


def _get_video_duration_ffprobe(video_path):
    """Get video duration in seconds using ffprobe."""
    ffprobe = _get_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _extract_frames_ffmpeg(video_path, output_dir, num_frames, duration=None):
    """Extract evenly-spaced frames from a video using ffmpeg.
    Returns list of frame dicts with 'path' and 'timestamp'."""
    ffmpeg = _get_ffmpeg()
    if not ffmpeg:
        log("ffmpeg not installed, skipping frame extraction", "WARN")
        return []

    if duration is None:
        duration = _get_video_duration_ffprobe(video_path)
    if not duration or duration <= 0:
        return []

    if duration < 10:
        num_frames = min(num_frames, 3)

    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for i in range(num_frames):
        ts = duration * (i + 0.5) / num_frames
        out_path = os.path.join(output_dir, f"frame_{i + 1:03d}.jpg")
        try:
            subprocess.run(
                [ffmpeg, "-ss", f"{ts:.2f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "3",
                 "-vf", "scale='min(640,iw)':-1",
                 "-y", out_path],
                capture_output=True, timeout=15,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                frames.append({"path": out_path, "timestamp": ts})
        except Exception:
            pass

    log(f"Extracted {len(frames)} keyframes from video")
    return frames


def extract_keyframes(url, platform, config, play_url=None, duration=None):
    """Download video and extract keyframes. Returns list of frame dicts or [].
    Frame dicts have 'path' (absolute) and 'timestamp' (seconds)."""
    if not config.get("extract_frames", True):
        return []
    if not _check_ffmpeg():
        log("ffmpeg not installed, skipping frame extraction", "WARN")
        return []

    num_frames = config.get("frames_per_video", 6)
    cache_key = _cache_key(url)
    frame_dir = os.path.join(SCREENSHOTS_DIR, cache_key)

    existing = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if existing:
        ttl_days = config.get("cache_ttl_days", 7)
        if ttl_days > 0:
            age_days = (time.time() - os.path.getmtime(existing[0])) / 86400
            if age_days > ttl_days:
                log(f"Cached frames expired ({age_days:.1f}d > {ttl_days}d)")
                shutil.rmtree(frame_dir, ignore_errors=True)
                existing = []
    if existing:
        log(f"Using {len(existing)} cached frames")
        frames = []
        for p in existing:
            idx = int(os.path.basename(p).split("_")[1].split(".")[0]) - 1
            ts = (duration or 0) * (idx + 0.5) / max(len(existing), 1)
            frames.append({"path": p, "timestamp": ts})
        return frames

    tmp_dir = tempfile.mkdtemp(prefix="frames_")
    try:
        video_path = None

        if play_url and platform == "douyin":
            video_path = _download_douyin_audio(play_url, tmp_dir)
        elif play_url and platform == "xiaohongshu":
            video_path = _download_xhs_video(play_url, tmp_dir)

        if not video_path and _check_ytdlp():
            log("Downloading video for frame extraction...")
            video_path = _ytdlp_download_video(url, tmp_dir, platform)

        if not video_path:
            return []

        if not duration:
            duration = _get_video_duration_ffprobe(video_path)

        return _extract_frames_ffmpeg(video_path, frame_dir, num_frames, duration)
    except Exception as e:
        log(f"Frame extraction failed: {e}", "WARN")
        return []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_with_ytdlp(url, platform="generic"):
    """Generic extraction using yt-dlp: metadata + subtitles."""
    if not _check_ytdlp():
        return None

    yt_info = _ytdlp_extract_info(url, platform)
    info = {
        "title": "",
        "author": "",
        "duration": 0,
        "description": "",
    }
    if yt_info:
        info["title"] = yt_info.get("title", "") or yt_info.get("fulltitle", "")
        info["author"] = yt_info.get("uploader", "") or yt_info.get("channel", "")
        info["duration"] = yt_info.get("duration", 0) or 0
        info["description"] = yt_info.get("description", "")

    tmp_dir = tempfile.mkdtemp(prefix="video_sub_")
    try:
        log(f"Trying yt-dlp subtitle extraction for {platform}...")
        text = _ytdlp_extract_subs(url, tmp_dir, platform)
        if text:
            return _make_result(info, platform, url, text, "yt_dlp_subs")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return _make_result(info, platform, url)


# ---------------------------------------------------------------------------
# Whisper ASR fallback
# ---------------------------------------------------------------------------

def transcribe_with_whisper(url, platform, info, config, play_url=None):
    """Download audio then transcribe with Whisper.
    For Douyin, uses play_url for direct download if yt-dlp fails."""
    whisper_mode = config.get("whisper_mode", "disabled")
    if whisper_mode == "disabled":
        return None

    tmp_dir = tempfile.mkdtemp(prefix="whisper_")
    try:
        log("Downloading audio for Whisper transcription...")
        audio_path = None

        if _check_ytdlp():
            audio_path = _ytdlp_download_audio(url, tmp_dir, platform)

        if not audio_path and play_url:
            log("Trying direct download via play_url...")
            if platform == "xiaohongshu":
                audio_path = _download_xhs_video(play_url, tmp_dir)
            else:
                audio_path = _download_douyin_audio(play_url, tmp_dir)

        if not audio_path:
            if not _check_ytdlp():
                log("yt-dlp not installed and no direct download available", "ERROR")
            else:
                log("Failed to download audio", "ERROR")
            return None

        text = None
        segments = []

        if whisper_mode == "api":
            text = _whisper_api(audio_path, config)
        elif whisper_mode == "local":
            text, segments = _whisper_local(audio_path, config)
        else:
            log(f"Unknown whisper_mode: {whisper_mode}", "ERROR")
            return None

        if text:
            # Generate SRT file if enabled
            if config.get("generate_srt", False) and segments:
                srt_dir = config.get("srt_output_dir", "subtitles")
                if not os.path.isabs(srt_dir):
                    srt_dir = os.path.join(SCRIPT_DIR, srt_dir)

                # Use video title or URL hash as filename
                video_title = info.get("title", "")
                if video_title:
                    # Sanitize filename
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', video_title)[:100]
                    srt_filename = f"{safe_title}.srt"
                else:
                    url_hash = _cache_key(url)
                    srt_filename = f"{url_hash}.srt"

                srt_path = os.path.join(srt_dir, srt_filename)
                _generate_srt(segments, srt_path)

            source = f"whisper_{whisper_mode}"
            return _make_result(info, platform, url, text, source)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return None


def _whisper_api(audio_path, config):
    """Transcribe using OpenAI Whisper API."""
    api_key = config.get("openai_api_key", "")
    if not api_key:
        log("openai_api_key not configured in config.json", "ERROR")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        language = config.get("language", "zh")

        file_size = os.path.getsize(audio_path)
        max_size = 25 * 1024 * 1024  # 25MB API limit

        if file_size > max_size:
            log(f"Audio file too large ({file_size // 1024 // 1024}MB), splitting...", "WARN")
            return _whisper_api_chunked(audio_path, client, language)

        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="text",
            )
        return resp if isinstance(resp, str) else str(resp)
    except ImportError:
        log("openai package not installed. Run: pip install openai", "ERROR")
    except Exception as e:
        log(f"Whisper API failed: {e}", "ERROR")
    return None


def _whisper_api_chunked(audio_path, client, language):
    """Split large audio and transcribe in chunks."""
    try:
        from pydub import AudioSegment
    except ImportError:
        log("pydub not installed, cannot split audio. Run: pip install pydub", "ERROR")
        return None

    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_ms = 10 * 60 * 1000  # 10 minutes per chunk
        chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]

        parts = []
        for i, chunk in enumerate(chunks):
            log(f"Transcribing chunk {i + 1}/{len(chunks)}...")
            chunk_path = audio_path + f".chunk{i}.m4a"
            chunk.export(chunk_path, format="ipod")
            with open(chunk_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language,
                    response_format="text",
                )
            text = resp if isinstance(resp, str) else str(resp)
            if text:
                parts.append(text.strip())
            os.unlink(chunk_path)

        return "\n".join(parts) if parts else None
    except Exception as e:
        log(f"Chunked transcription failed: {e}", "ERROR")
        return None


def _format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _generate_srt(segments, output_path):
    """Generate SRT subtitle file from segments.

    Args:
        segments: List of dicts with 'text', 'start', 'end' keys
        output_path: Path to save the SRT file

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start_time = _format_srt_timestamp(seg["start"])
                end_time = _format_srt_timestamp(seg["end"])
                text = seg["text"].strip()

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

        log(f"SRT file generated: {output_path}")
        return True
    except Exception as e:
        log(f"Failed to generate SRT file: {e}", "ERROR")
        return False


def _volcengine_asr(audio_path, config):
    """Transcribe using Volcano Engine Speech Recognition API."""
    appid = config.get("volcengine_appid", "")
    token = config.get("volcengine_token", "")
    language = config.get("volcengine_language", "zh-CN")

    if not appid or not token:
        log("volcengine_appid and volcengine_token must be configured in config.json", "ERROR")
        return None, []

    # Split token into access_key and secret_key (format: "access_key:secret_key")
    if ":" in token:
        access_key, secret_key = token.split(":", 1)
    else:
        log("volcengine_token format should be 'access_key:secret_key'", "ERROR")
        return None, []

    try:
        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Prepare request
        url = "https://openspeech.bytedance.com/api/v1/asr"
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        headers = {
            "Content-Type": "application/json",
            "X-Date": timestamp,
        }

        payload = {
            "app": {
                "appid": appid,
                "token": "access_token",
                "cluster": "volcano_asr"
            },
            "user": {
                "uid": "user_001"
            },
            "audio": {
                "format": "wav",
                "rate": 16000,
                "bits": 16,
                "channel": 1,
                "language": language
            },
            "request": {
                "reqid": hashlib.md5(str(time.time()).encode()).hexdigest(),
                "nbest": 1,
                "word_info": 1,
                "result_type": "full"
            }
        }

        payload_json = json.dumps(payload)

        # Sign request
        query_params = {}
        authorization = _sign_volcengine_request("POST", "/api/v1/asr", query_params, headers, payload_json, access_key, secret_key)
        headers["Authorization"] = authorization

        # Make request
        req = urllib.request.Request(url, data=payload_json.encode(), headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode())

        if result.get("code") != 0:
            error_msg = result.get("message", "Unknown error")
            log(f"Volcano Engine ASR failed: {error_msg}", "ERROR")
            return None, []

        # Parse results
        segments = []
        lines = []

        for utterance in result.get("result", []):
            text = utterance.get("text", "").strip()
            start_time = utterance.get("start_time", 0) / 1000.0  # Convert ms to seconds
            end_time = utterance.get("end_time", 0) / 1000.0

            if text:
                segments.append({
                    "text": text,
                    "start": start_time,
                    "end": end_time
                })
                lines.append(text)

        if not lines:
            return None, []

        return "\n".join(lines), segments

    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log(f"Volcano Engine API HTTP error: {e.code} {e.reason}\n{error_body}", "ERROR")
        return None, []
    except Exception as e:
        log(f"Volcano Engine ASR failed: {e}", "ERROR")
        return None, []


def _sign_volcengine_request(method, uri, query_params, headers, payload, access_key, secret_key):
    """Generate AWS Signature Version 4 for Volcano Engine API."""
    # Step 1: Create canonical request
    canonical_uri = uri
    canonical_querystring = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted(query_params.items())
    )

    canonical_headers = "\n".join(
        f"{k.lower()}:{v.strip()}" for k, v in sorted(headers.items())
    ) + "\n"

    signed_headers = ";".join(sorted(k.lower() for k in headers.keys()))

    payload_hash = hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()

    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    # Step 2: Create string to sign
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{headers['X-Date'][:8]}/cn-north-1/sami/request"
    string_to_sign = f"{algorithm}\n{headers['X-Date']}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    # Step 3: Calculate signature
    k_date = hmac.new(f"AWS4{secret_key}".encode(), headers['X-Date'][:8].encode(), hashlib.sha256).digest()
    k_region = hmac.new(k_date, b"cn-north-1", hashlib.sha256).digest()
    k_service = hmac.new(k_region, b"sami", hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

    # Step 4: Add authorization header
    authorization = f"{algorithm} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    return authorization


def _whisper_local(audio_path, config):
    """Transcribe using faster-whisper (local model)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log("faster-whisper not installed. Run: pip install faster-whisper", "ERROR")
        return None, []

    # Add CUDA DLL directories for Windows GPU acceleration
    if sys.platform == "win32":
        try:
            import site
            for site_packages in site.getsitepackages():
                cuda_bin_dirs = [
                    os.path.join(site_packages, "nvidia", "cublas", "bin"),
                    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
                    os.path.join(site_packages, "nvidia", "cuda_nvrtc", "bin"),
                    os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
                ]
                for cuda_dir in cuda_bin_dirs:
                    if os.path.exists(cuda_dir):
                        os.add_dll_directory(cuda_dir)
        except (AttributeError, Exception):
            pass  # os.add_dll_directory not available or failed

    model_size = config.get("whisper_model", "base")
    device = config.get("whisper_device", "auto")
    compute_type = config.get("whisper_compute_type", "auto")
    language = config.get("language", "zh")
    log(f"Loading Whisper model '{model_size}' on device '{device}' with compute_type '{compute_type}'...")

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, language=language)

        # Collect segments with timestamps
        segment_list = []
        lines = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                segment_list.append({
                    "text": text,
                    "start": seg.start,
                    "end": seg.end
                })
                lines.append(text)

        if not lines:
            return None, []

        return "\n".join(lines), segment_list
    except Exception as e:
        log(f"Local Whisper transcription failed: {e}", "ERROR")
        return None, []


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def format_duration(seconds):
    if not seconds or not isinstance(seconds, (int, float)):
        return "00:00"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _make_result(info, platform, url, subtitle_text=None, source=None, error=None):
    return {
        "info": info,
        "platform": platform,
        "url": url,
        "subtitle_text": subtitle_text,
        "source": source,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def extract(url, config):
    """
    Main entry point. Tries cache, then platform-specific extraction,
    then yt-dlp subs, then Whisper ASR. Returns a result dict.
    """
    # Phase 0: check cache
    cached = _read_cache(url)
    if cached:
        return cached

    platform = detect_platform(url)
    log(f"Detected platform: {platform}")

    # Phase 1: platform-specific subtitle extraction (fast, no heavy deps)
    result = None
    play_url = None

    if platform == "bilibili":
        result = extract_bilibili(url)
    elif platform == "youtube":
        result = extract_youtube(url)
    elif platform == "douyin":
        result = extract_douyin(url)
    elif platform == "xiaohongshu":
        result = extract_xiaohongshu(url)

    if result:
        play_url = result.get("_play_url")

    final_result = None

    if result and result.get("subtitle_text"):
        final_result = result
    else:
        # Phase 2: yt-dlp subtitle extraction (works for all platforms)
        info = result["info"] if result else {"title": "", "author": "", "duration": 0, "description": ""}

        if _check_ytdlp():
            ytdlp_result = extract_with_ytdlp(url, platform)
            if ytdlp_result:
                if ytdlp_result.get("subtitle_text"):
                    if not info.get("title") and ytdlp_result["info"].get("title"):
                        info = ytdlp_result["info"]
                    final_result = _make_result(info, platform, url,
                                                ytdlp_result["subtitle_text"],
                                                ytdlp_result["source"])
                if not info.get("title") and ytdlp_result["info"].get("title"):
                    info = ytdlp_result["info"]
        else:
            log("yt-dlp not installed. Install with: pip install yt-dlp", "WARN")

        # Phase 3: Whisper ASR fallback
        if not final_result:
            whisper_result = transcribe_with_whisper(url, platform, info, config, play_url=play_url)
            if whisper_result and whisper_result.get("subtitle_text"):
                final_result = whisper_result

        # All methods failed
        if not final_result:
            error_parts = ["No subtitles or transcript could be extracted."]
            if not _check_ytdlp():
                error_parts.append("Install yt-dlp for broader platform support: pip install yt-dlp")
            if platform == "youtube":
                try:
                    import youtube_transcript_api  # noqa: F401
                except ImportError:
                    error_parts.append("Install youtube-transcript-api for YouTube: pip install youtube-transcript-api")
            if platform in PLATFORMS_NEEDING_COOKIES and not _cookie_args_result:
                error_parts.append(
                    f"Douyin/Xiaohongshu/TikTok require browser cookies. "
                    f"Export cookies: install 'Get cookies.txt LOCALLY' browser extension, "
                    f"visit the site, export, and save the file (e.g. www.douyin.com_cookies.txt "
                    f"or cookies.txt) in: {SCRIPT_DIR}"
                )
            whisper_mode = config.get("whisper_mode", "disabled")
            if whisper_mode == "disabled":
                error_parts.append("Enable Whisper in config.json for audio transcription fallback.")
            return _make_result(info, platform, url, error="\n".join(error_parts))

    # Phase 4: Extract keyframes (runs for all successful extractions)
    duration = (final_result.get("info") or {}).get("duration", 0)
    frames = extract_keyframes(url, platform, config, play_url=play_url, duration=duration)
    if frames:
        final_result["frames"] = frames

    _write_cache(url, final_result)
    return final_result


def _clear_cache():
    """Remove all cached results and screenshots."""
    removed = 0
    for d in [CACHE_DIR, SCREENSHOTS_DIR]:
        if os.path.isdir(d):
            removed += sum(len(files) for _, _, files in os.walk(d))
            shutil.rmtree(d, ignore_errors=True)
    log(f"Cleared cache: {removed} files removed")


def extract_local_video(file_path, config):
    """Extract subtitles from a local video file using ASR."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    log(f"Processing local video: {file_path}")

    # Get video info using ffmpeg
    info = {"title": os.path.basename(file_path), "author": "", "duration": 0, "description": ""}

    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            info["duration"] = int(float(result.stdout.strip()))
    except Exception as e:
        log(f"Could not get video duration: {e}", "WARN")

    # Extract audio and transcribe
    tmp_dir = tempfile.mkdtemp(prefix="local_video_")
    try:
        audio_path = os.path.join(tmp_dir, "audio.wav")
        log("Extracting audio from local video...")

        cmd = ["ffmpeg", "-i", file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)

        if not os.path.exists(audio_path):
            return {"error": "Failed to extract audio from video"}

        # Determine ASR engine
        asr_engine = config.get("asr_engine", "whisper")
        text = None
        segments = []
        source = None

        if asr_engine == "volcengine":
            text, segments = _volcengine_asr(audio_path, config)
            source = "volcengine"
        elif asr_engine == "whisper":
            whisper_mode = config.get("whisper_mode", "disabled")
            if whisper_mode == "local":
                text, segments = _whisper_local(audio_path, config)
                source = f"whisper_{whisper_mode}"
            elif whisper_mode == "api":
                text = _whisper_api(audio_path, config)
                source = f"whisper_{whisper_mode}"
            else:
                return {"error": "Whisper is disabled. Enable it in config.json to transcribe local videos."}
        elif asr_engine == "auto":
            # Try Volcano Engine first, fallback to Whisper
            text, segments = _volcengine_asr(audio_path, config)
            if text:
                source = "volcengine"
            else:
                log("Volcano Engine failed, falling back to Whisper...")
                whisper_mode = config.get("whisper_mode", "disabled")
                if whisper_mode == "local":
                    text, segments = _whisper_local(audio_path, config)
                    source = f"whisper_{whisper_mode}"
                elif whisper_mode == "api":
                    text = _whisper_api(audio_path, config)
                    source = f"whisper_{whisper_mode}"
        else:
            return {"error": f"Unknown asr_engine: {asr_engine}"}

        if text:
            # Determine output directory for local video files
            video_name = os.path.splitext(os.path.basename(file_path))[0]
            safe_video_name = re.sub(r'[<>:"/\\|?*]', '_', video_name)

            # Check if publish_dir is configured
            publish_dir = config.get("publish_dir", "")
            if publish_dir and os.path.isabs(publish_dir):
                base_dir = publish_dir
            else:
                srt_dir = config.get("srt_output_dir", "subtitles")
                if not os.path.isabs(srt_dir):
                    srt_dir = os.path.join(SCRIPT_DIR, srt_dir)
                base_dir = srt_dir

            # Create subfolder for local video
            video_folder = os.path.join(base_dir, safe_video_name)
            os.makedirs(video_folder, exist_ok=True)

            # Generate SRT file if enabled
            if config.get("generate_srt", False) and segments:
                srt_path = os.path.join(video_folder, f"{safe_video_name}.srt")
                _generate_srt(segments, srt_path)

            # Extract frames if enabled
            frames = None
            if config.get("extract_frames", True):
                num_frames = config.get("frames_per_video", 6)
                # Save frames to subfolder's screenshots directory
                frame_dir = os.path.join(video_folder, "screenshots")
                frames = _extract_frames_ffmpeg(file_path, frame_dir, num_frames, duration=None)

            result = _make_result(info, "local", file_path, text, source)
            if frames:
                result["frames"] = frames
            return result
        else:
            return {"error": "ASR transcription failed"}

    except subprocess.TimeoutExpired:
        return {"error": "Audio extraction timed out"}
    except subprocess.CalledProcessError as e:
        return {"error": f"ffmpeg failed: {e.stderr.decode() if e.stderr else str(e)}"}
    except Exception as e:
        return {"error": f"Local video processing failed: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--clear-cache":
        _clear_cache()
        return

    if len(sys.argv) < 2:
        print("Usage: python video_subtitle.py <URL_or_FILE_PATH>", file=sys.stderr)
        print("       python video_subtitle.py --clear-cache", file=sys.stderr)
        print("  Supports: Bilibili, YouTube, Douyin, Xiaohongshu, TikTok, 1800+ sites, and local video files", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    config = load_config()

    # Check if input is a local file
    if os.path.exists(input_path) and os.path.isfile(input_path):
        log(f"Detected local video file: {input_path}")
        result = extract_local_video(input_path, config)
    else:
        log(f"Extracting subtitles for: {input_path}")
        result = extract(input_path, config)

    if not result:
        log("Extraction returned no result", "ERROR")
        sys.exit(1)

    info = result.get("info", {})
    output = {
        "title": info.get("title", ""),
        "author": info.get("author", ""),
        "duration": format_duration(info.get("duration", 0)),
        "description": info.get("description", ""),
        "platform": result.get("platform", "unknown"),
        "url": result.get("url", input_path),
    }

    if result.get("error"):
        output["error"] = result["error"]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(1)

    output["source"] = result.get("source", "unknown")
    output["subtitle_text"] = result["subtitle_text"]

    if result.get("frames"):
        output["frames"] = [
            {"path": f["path"], "timestamp": format_duration(int(f["timestamp"]))}
            for f in result["frames"]
        ]

    print(json.dumps(output, ensure_ascii=False, indent=2))

    # Generate Markdown summary if enabled
    if config.get("generate_markdown", True):
        _generate_markdown_summary(output, config)


def _generate_markdown_summary(output, config):
    """Generate detailed Markdown summary file."""
    if output.get("error") or not output.get("subtitle_text"):
        return

    # Determine if this is a local video file
    url = output.get("url", "")
    is_local = os.path.exists(url) and os.path.isfile(url)

    # Get title
    title = output.get("title", "video")
    if not title or title == "video":
        if is_local:
            title = os.path.splitext(os.path.basename(url))[0]
        else:
            title = "video_summary"

    # Sanitize title for filename
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)

    # Generate date prefix (YYYYMMDD format)
    date_prefix = datetime.now().strftime("%Y%m%d")
    markdown_filename = f"{date_prefix}-{safe_title}.md"

    # Determine output directory
    if is_local:
        # For local videos: create subfolder with video name
        publish_dir = config.get("publish_dir", "")
        if publish_dir and os.path.isabs(publish_dir):
            base_dir = publish_dir
        else:
            markdown_dir = config.get("markdown_output_dir", "summaries")
            if not os.path.isabs(markdown_dir):
                markdown_dir = os.path.join(SCRIPT_DIR, markdown_dir)
            base_dir = markdown_dir

        # Create subfolder with video name
        video_folder = os.path.join(base_dir, safe_title)
        os.makedirs(video_folder, exist_ok=True)
        markdown_path = os.path.join(video_folder, markdown_filename)
    else:
        # For online videos: save directly to publish_dir or markdown_output_dir
        publish_dir = config.get("publish_dir", "")
        if publish_dir and os.path.isabs(publish_dir):
            markdown_dir = publish_dir
        else:
            markdown_dir = config.get("markdown_output_dir", "summaries")
            if not os.path.isabs(markdown_dir):
                markdown_dir = os.path.join(SCRIPT_DIR, markdown_dir)

        os.makedirs(markdown_dir, exist_ok=True)
        markdown_path = os.path.join(markdown_dir, markdown_filename)

    style = config.get("markdown_style", "detailed")

    try:
        with open(markdown_path, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"# {title}\n\n")
            f.write(f"**来源**: {output.get('url', 'N/A')}  \n")
            f.write(f"**时长**: {output.get('duration', 'N/A')}  \n")
            f.write(f"**平台**: {output.get('platform', 'unknown')}  \n")
            if output.get('author'):
                f.write(f"**作者**: {output['author']}  \n")
            f.write(f"\n---\n\n")

            subtitle_text = output.get("subtitle_text", "")
            frames = output.get("frames", [])

            if style == "detailed":
                _write_detailed_summary(f, subtitle_text, frames, title, output.get('url', ''))
            else:  # bullet style
                _write_bullet_summary(f, subtitle_text, frames, title, output.get('url', ''))

        log(f"Markdown summary generated: {markdown_path}")
    except Exception as e:
        log(f"Failed to generate Markdown: {e}", "ERROR")


def _write_detailed_summary(f, subtitle_text, frames, title, url):
    """Write detailed paragraph-style summary with AI analysis."""
    # Split subtitle into logical sections (every ~500 characters or natural breaks)
    lines = subtitle_text.split('\n')
    sections = []
    current_section = []
    current_length = 0

    for line in lines:
        current_section.append(line)
        current_length += len(line)
        if current_length > 500 or line.endswith(('。', '！', '？', '.', '!', '?')):
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
                current_length = 0

    if current_section:
        sections.append('\n'.join(current_section))

    # Distribute frames across sections
    frames_per_section = {}
    if frames and sections:
        for i, frame in enumerate(frames):
            section_idx = int(i * len(sections) / len(frames))
            if section_idx not in frames_per_section:
                frames_per_section[section_idx] = frame

    # Write sections with AI-generated summaries
    emojis = ['🏷️', '💡', '🧠', '🍎', '🔑', '🔢', '🧱', '🎯', '✨', '🌟']

    for i, section in enumerate(sections):
        emoji = emojis[i % len(emojis)]

        # Generate AI summary for this section
        summary_data = _generate_section_summary(section, i+1)

        f.write(f"## {emoji} {summary_data['title']}\n\n")

        # Embed frame if available
        if i in frames_per_section:
            frame = frames_per_section[i]
            f.write(f"![截图]({frame['path']})\n\n")
            f.write(f"*时间戳: {frame['timestamp']}*\n\n")

        # Write main content summary
        f.write(f"**主要内容**：{summary_data['main_content']}\n\n")

        # Write key points
        f.write(f"**关键要点**：\n")
        for point in summary_data['key_points']:
            f.write(f"- {point}\n")
        f.write(f"\n")

        # Write important quotes if any
        if summary_data['important_quotes']:
            f.write(f"**重要语句**：\n")
            for quote in summary_data['important_quotes']:
                f.write(f"> {quote}\n\n")

    # Add overall summary section
    f.write(f"## 📝 总结\n\n")
    overall_summary = _generate_overall_summary(subtitle_text, len(sections))
    f.write(f"{overall_summary}\n\n")

    # Add note about transcription source
    f.write(f"---\n\n")
    f.write(f"*注：本文档由 AI 自动生成，字幕通过语音识别技术提取，可能存在少量识别误差。*\n")


def _write_bullet_summary(f, subtitle_text, frames, title, url):
    """Write BibiGPT-style bullet point summary."""
    f.write(f"# 🤖 {title} — AI 总结\n\n")

    # Split into sections
    lines = subtitle_text.split('\n')
    sections = []
    current_section = []

    for line in lines:
        current_section.append(line)
        if len(current_section) >= 10:
            sections.append(current_section)
            current_section = []

    if current_section:
        sections.append(current_section)

    # Distribute frames
    frames_per_section = {}
    if frames and sections:
        for i, frame in enumerate(frames):
            section_idx = int(i * len(sections) / len(frames))
            if section_idx not in frames_per_section:
                frames_per_section[section_idx] = frame

    emojis = ['🏷️', '💡', '🧠', '🍎', '🔑', '🔢', '🧱', '🎯']

    for i, section in enumerate(sections):
        emoji = emojis[i % len(emojis)]
        f.write(f"### {emoji} 要点 {i+1}\n")

        if i in frames_per_section:
            frame = frames_per_section[i]
            f.write(f"![要点{i+1}]({frame['path']})\n")

        for line in section[:5]:  # Max 5 lines per section
            f.write(f"- {line}\n")
        f.write(f"\n")

    f.write(f"---\n\n")
    f.write(f"*注：本总结由 AI 自动生成*\n")


def _generate_section_summary(section_text, section_num):
    """Generate AI summary for a section using Claude API."""
    try:
        # Try to import anthropic
        try:
            from anthropic import Anthropic
        except ImportError:
            log("anthropic package not installed. Run: pip install anthropic", "WARNING")
            return _fallback_section_summary(section_text, section_num)

        # Load config to get API key
        config = _load_config()
        api_key = config.get("anthropic_api_key", "")

        if not api_key:
            log("No Anthropic API key configured. Using fallback summary.", "WARNING")
            return _fallback_section_summary(section_text, section_num)

        client = Anthropic(api_key=api_key)

        prompt = f"""分析以下视频字幕片段，生成结构化总结：

字幕内容：
{section_text}

请以JSON格式返回，包含：
1. title: 这部分的简短标题（5-10字）
2. main_content: 主要内容概括（1-2句话，30-50字）
3. key_points: 关键要点列表（3-5个要点，每个10-20字）
4. important_quotes: 重要原文引用（0-2句，直接从字幕中提取）

返回格式：
{{
  "title": "标题",
  "main_content": "概括",
  "key_points": ["要点1", "要点2", "要点3"],
  "important_quotes": ["引用1", "引用2"]
}}"""

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result

    except Exception as e:
        log(f"AI summary failed: {e}", "WARNING")
        return _fallback_section_summary(section_text, section_num)


def _fallback_section_summary(section_text, section_num):
    """Fallback summary when AI is not available."""
    lines = [l.strip() for l in section_text.split('\n') if l.strip()]

    # Extract sentences
    sentences = []
    for line in lines:
        sentences.extend([s.strip() for s in re.split(r'[。！？.!?]', line) if s.strip()])

    # Take first sentence as title
    title = f"第 {section_num} 部分"
    if sentences:
        title_text = sentences[0][:15]
        if len(sentences[0]) > 15:
            title_text += "..."
        title = title_text

    # Main content: first 2-3 sentences
    main_content = '。'.join(sentences[:2]) + '。' if len(sentences) >= 2 else (sentences[0] if sentences else "")

    # Key points: longer sentences
    key_points = [s for s in sentences if len(s) > 15][:4]
    if not key_points:
        key_points = sentences[:3]

    # Important quotes: sentences with keywords
    keywords = ['重要', '关键', '一定', '必须', '建议', '注意', '避免', '推荐']
    important_quotes = [s for s in sentences if any(kw in s for kw in keywords)][:2]

    return {
        "title": title,
        "main_content": main_content,
        "key_points": key_points,
        "important_quotes": important_quotes
    }


def _generate_overall_summary(subtitle_text, num_sections):
    """Generate overall summary using Claude API."""
    try:
        try:
            from anthropic import Anthropic
        except ImportError:
            return f"本视频共 {num_sections} 个部分，详细内容见上文各章节。"

        config = _load_config()
        api_key = config.get("anthropic_api_key", "")

        if not api_key:
            return f"本视频共 {num_sections} 个部分，详细内容见上文各章节。"

        client = Anthropic(api_key=api_key)

        # Truncate subtitle if too long
        max_chars = 3000
        truncated_text = subtitle_text[:max_chars]
        if len(subtitle_text) > max_chars:
            truncated_text += "..."

        prompt = f"""请为以下视频字幕生成一个简洁的总结（100-150字），概括视频的主要内容和核心观点：

{truncated_text}

要求：
1. 用1-2段话总结
2. 突出核心主题和关键信息
3. 语言简洁流畅"""

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text.strip()

    except Exception as e:
        log(f"Overall summary failed: {e}", "WARNING")
        return f"本视频共 {num_sections} 个部分，详细内容见上文各章节。"



if __name__ == "__main__":
    main()
