#!/usr/bin/env python3
"""
Volcano Engine (火山引擎) Speech Recognition API Integration
https://www.volcengine.com/docs/6561/79817
"""

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime


def _sign_request(method, uri, query_params, headers, payload, access_key, secret_key):
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


def transcribe_with_volcengine(audio_path, config):
    """
    Transcribe audio using Volcano Engine Speech Recognition API.

    Args:
        audio_path: Path to audio file
        config: Configuration dict with volcengine_appid, volcengine_token, volcengine_language

    Returns:
        tuple: (text, segments) where segments is a list of dicts with text, start, end
    """
    appid = config.get("volcengine_appid", "")
    token = config.get("volcengine_token", "")
    language = config.get("volcengine_language", "zh-CN")

    if not appid or not token:
        print("ERROR: volcengine_appid and volcengine_token must be configured in config.json", file=sys.stderr)
        return None, []

    # Split token into access_key and secret_key (format: "access_key:secret_key")
    if ":" in token:
        access_key, secret_key = token.split(":", 1)
    else:
        print("ERROR: volcengine_token format should be 'access_key:secret_key'", file=sys.stderr)
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
        authorization = _sign_request("POST", "/api/v1/asr", query_params, headers, payload_json, access_key, secret_key)
        headers["Authorization"] = authorization

        # Make request
        req = urllib.request.Request(url, data=payload_json.encode(), headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode())

        if result.get("code") != 0:
            error_msg = result.get("message", "Unknown error")
            print(f"ERROR: Volcano Engine ASR failed: {error_msg}", file=sys.stderr)
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
        print(f"ERROR: Volcano Engine API HTTP error: {e.code} {e.reason}\n{error_body}", file=sys.stderr)
        return None, []
    except Exception as e:
        print(f"ERROR: Volcano Engine ASR failed: {e}", file=sys.stderr)
        return None, []


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python volcengine_asr.py <audio_file>")
        sys.exit(1)

    # Test with config
    config = {
        "volcengine_appid": os.getenv("VOLCENGINE_APPID", ""),
        "volcengine_token": os.getenv("VOLCENGINE_TOKEN", ""),
        "volcengine_language": "zh-CN"
    }

    text, segments = transcribe_with_volcengine(sys.argv[1], config)

    if text:
        print("Transcription:")
        print(text)
        print(f"\nSegments: {len(segments)}")
    else:
        print("Transcription failed")
        sys.exit(1)
