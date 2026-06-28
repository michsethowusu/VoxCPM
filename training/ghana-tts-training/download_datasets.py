import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPOS = ["ghananlpcommunity/ghana-speech", "ghananlpcommunity/ghana-english-asr-2700hrs"]
for repo in REPOS:
    name = repo.split("/")[-1]
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"downloading {repo} (attempt {attempt})", flush=True)
            p = snapshot_download(
                repo_id=repo, repo_type="dataset",
                local_dir=f"/mnt/volume_d2wey28/data/{name}",
                max_workers=8,
                etag_timeout=30,
            )
            print("done", repo, p, flush=True)
            break
        except Exception as e:
            print(f"retry {repo}: {type(e).__name__}: {e}", flush=True)
            time.sleep(10)
print("ALL_DOWNLOADS_DONE", flush=True)
