import os


VERSION = "0.4.0"
TAG = f"v{VERSION}"
RELEASE_CHANNEL = os.environ.get("EASY_ASR_RELEASE_CHANNEL", "prerelease")
RELEASE_COMMIT = os.environ.get("EASY_ASR_COMMIT", "unknown")
