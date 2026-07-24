# RunPod setup

The repo is the unit of transfer. Code is edited on the laptop and pulled on the
pod; data and adapters live on the pod's network volume and never travel.

## One-time

**1. Network volume — pick the region first.**

A network volume is locked to one datacenter region, and a pod can only mount a
volume in its own region. Check which regions actually have the GPU you want
*before* creating the volume, or you will end up with 150 GB of data in a region
with nothing to attach it to.

Size: 150 GB. At roughly $0.07/GB/month that is ~$10/month, and it persists
while no pod is running.

**2. Mount point.** RunPod mounts the network volume at `/workspace`. Everything
that must survive a pod being destroyed goes there — the repo, the data, the
model cache, the adapters.

## Each session

```bash
cd /workspace
git clone <your-repo-url> VLM_poc      # first time only
cd VLM_poc && git pull

cp .env.example .env                   # first time only, then fill it in
set -a && . ./.env && set +a

pip install -r requirements-gpu.txt
```

Keep `HF_HOME=/workspace/.cache/huggingface` (already the default in
`.env.example`) so model weights are cached on the volume. Otherwise every new
pod re-downloads several GB of Qwen weights before doing any work.

At the end of a session: **stop the pod.** GPU billing stops; the volume stays.

## Use a CPU pod for the data stages

Downloading, extracting, parsing and chipping are pure CPU work. Renting a 4090
to run `tar` is paying GPU rates to wait on I/O.

- **CPU pod** (or the cheapest available GPU) — `download_xbd.py`,
  `parse_annotations.py`, `build_chips.py`, `split.py`, `build_dataset.py`
- **GPU pod** (24 GB: A10G / L4 / RTX 4090) — `predict.py` and training only

Both mount the same volume, so switching costs nothing but a restart.

## Getting results back

Metrics and adapters are small. Everything else stays on the volume.

```bash
# on the pod
python scripts/evaluate.py --pred outputs/eval/tuned.jsonl \
    --baseline outputs/eval/base.jsonl --out outputs/eval/run1.json

# from the laptop
runpodctl send outputs/eval/run1.json     # or scp, or commit it — it is tracked
```

`outputs/eval/*.json` is deliberately **not** gitignored: the metrics are the
result, and they belong in the repo. The `.jsonl` generation dumps are ignored.

## Cost sanity check

| item | rate | expected |
|---|---|---|
| network volume, 150 GB | ~$0.07/GB/mo | ~$10/mo |
| CPU pod, data prep | ~$0.10–0.20/hr | ~3 hr |
| 24 GB GPU, train + eval | ~$0.30–0.50/hr | ~8–12 hr |

Total for the project: **$15–30**. If it is trending well past that, something
is being re-run that should have been cached — most likely model weights, or a
chipping pass that did not need repeating.
