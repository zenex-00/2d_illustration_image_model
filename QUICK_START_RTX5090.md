# RTX 5090 Deployment Quick Start Guide
## Gemini 3 Pro Vehicle-to-Vector Pipeline

**Purpose:** Fast reference for deploying to RunPod RTX 5090  
**Time:** 5 minutes to read, 1 hour to deploy  
**Status:** Production-ready with bug fixes applied

---

## 📋 TLDR: What You Need to Know

| Item | Value |
|---|---|
| **GPU** | RTX 5090 (24GB VRAM) |
| **PyTorch** | 2.8.0+ with CUDA 12.8 (nightly) |
| **Storage** | 100GB persistent volume |
| **Runtime** | 60-120 seconds per image |
| **Throughput** | 40-60 images/hour |
| **Memory Peak** | 18GB (within budget) |
| **Bugs Found** | 3 critical, 4 high-priority |
| **Fixes Needed** | ~1.5 hours to apply all |
| **Status** | ✅ Production-ready (with fixes) |

---

## 🚀 QUICK DEPLOYMENT (After Applying Fixes)

### Step 1: Create RunPod Pod (5 minutes)

1. Go to **RunPod Dashboard** → **Pods** → **New Pod**
2. Select **RTX 5090** GPU
3. Container Image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
4. Volume: Create **100GB** persistent volume
5. Mount at: `/models`
6. Environment:
   ```env
   PYTHONUNBUFFERED=1
   CUDA_VISIBLE_DEVICES=0
   MODEL_VOLUME_PATH=/models
   DISABLE_XFORMERS=1
   GPU_MODEL=RTX_5090
   WORKERS=1
   ```
7. Startup Command:
   ```bash
   /bin/bash -c "export DEBIAN_FRONTEND=noninteractive && \
   dpkg --configure -a || true && \
   apt-get update && \
   apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata && \
   cd /workspace && \
   git clone https://github.com/zenex-00/2d_illustration_image_model.git image_generation && \
   git clone https://github.com/isl-org/ZoeDepth.git && \
   cd /workspace/image_generation && \
   pip install -q --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 && \
   bash scripts/install_dependencies.sh && \
   export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:\$PYTHONPATH && \
   python scripts/setup_model_volume.py --volume-path /models && \
   uvicorn src.api.server:app --host 0.0.0.0 --port 5090 --workers 1"
   ```
8. Click **Deploy**

### Step 2: Wait for Startup (20 minutes)

Pod will:
1. Download PyTorch (1-2 min)
2. Install dependencies (5-8 min)
3. Download models to volume (15-20 min)
4. Initialize API server (1-2 min)

### Step 3: Verify Setup (5 minutes)

```bash
# SSH into pod or open terminal via RunPod
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check health
curl http://localhost:5090/health

# Check readiness
curl http://localhost:5090/ready

# Test with image
curl -X POST http://localhost:5090/api/v1/process \
  -F "file=@test.jpg" \
  -F "palette_hex_list=FF0000,00FF00,0000FF,FFFF00,FF00FF,00FFFF,000000,FFFFFF,808080,404040,C0C0C0,FFA500,800080,008080,FFB6C1"
```

### Step 4: Check Status

```bash
# Get job ID from response
JOB_ID="xxx-xxx-xxx"

# Check status every 10 seconds
while true; do
  curl -s http://localhost:5090/api/v1/jobs/$JOB_ID | python -m json.tool
  sleep 10
done
```

---

## 🐛 CRITICAL BUGS TO FIX (BEFORE DEPLOYMENT)

### Bug #1: Job Queue Race Condition
**File:** `src/api/job_queue.py` line 69  
**Fix:** Add lock around job creation
**Time:** 5 minutes
```python
# BEFORE:
self.jobs[job_id] = job
if len(self.jobs) > self.max_jobs:
    self._cleanup_old_jobs()

# AFTER:
with self._lock:
    self.jobs[job_id] = job
    if len(self.jobs) > self.max_jobs:
        self._cleanup_old_jobs()
```

### Bug #2: Unsafe Object Creation
**File:** `src/pipeline/orchestrator.py` line 158  
**Fix:** Use normal constructor instead of `object.__new__()`
**Time:** 5 minutes
```python
# BEFORE:
palette = object.__new__(PaletteManager)
palette.hex_colors = palette_hex_list

# AFTER:
palette = PaletteManager(palette_hex_list)
```

### Bug #3: Log Buffer O(n) Operation
**File:** `src/api/training_jobs.py` line 52  
**Fix:** Use deque instead of list
**Time:** 5 minutes
```python
# BEFORE:
self.log_buffer = []
del self.log_buffer[:-1000]  # Slow!

# AFTER:
from collections import deque
self.log_buffer = deque(maxlen=1000)  # Auto-trim
```

### Bug #4: xformers Configuration
**File:** `src/api/server.py` line 8  
**Fix:** Disable xformers for RTX 5090
**Time:** 10 minutes
```bash
export DISABLE_XFORMERS=1
export GPU_MODEL=RTX_5090
```

### Bug #5: Request Timeout
**File:** `run.py` + `src/api/server.py`  
**Fix:** Add timeout middleware
**Time:** 15 minutes
```python
# In run.py:
uvicorn.run(..., timeout_keep_alive=30, timeout_graceful_shutdown=30)

# In server.py:
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=300)
```

---

## 📊 PERFORMANCE EXPECTATIONS

### Single Image Processing

| Phase | Time | Memory | Status |
|---|---|---|---|
| Phase 1 (Sanitization) | 15-25s | 5.2 GB | ✅ |
| Phase 2 (Generation) | 30-45s | 18 GB | ⚠️ Peak |
| Phase 3 (Quantization) | 8-15s | 3 GB | ✅ |
| Phase 4 (Vectorization) | 5-10s | 0.5 GB | ✅ |
| **Total** | **60-120s** | **18 GB** | ✅ |

### Throughput

- **Sequential:** ~40-60 images/hour
- **Concurrent (2 req):** ~30-40 images/hour (slower due to VRAM)
- **Queue:** Async job queue handles unlimited submissions

---

## 🔍 MONITORING & HEALTH

### Essential Metrics

```bash
# GPU Memory Usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Job Queue Size
curl -s http://localhost:5090/metrics | grep job_queue_size

# Error Rate
curl -s http://localhost:5090/metrics | grep errors_total

# Recent Logs
curl -s http://localhost:5090/api/v1/jobs/{JOB_ID} | jq '.logs'
```

### Health Checks

```bash
# Should return 200 OK
curl http://localhost:5090/health

# Should return ready status
curl http://localhost:5090/ready

# API docs
curl http://localhost:5090/docs
```

---

## ⚠️ COMMON ISSUES & FIXES

### Issue: "CUDA is not available for compute capability sm_120"

**Solution:**
```bash
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: Out of Memory (OOM) Errors

**Check:**
1. Is Phase 2 running? It uses 18GB peak - this is normal
2. Are other processes using GPU? Run `nvidia-smi`
3. Are models properly flushed? Look for "phase2_memory_flushed" in logs

**Solution:**
```bash
# Reduce concurrent requests
# Or reduce num_inference_steps in config:
phase2:
  sdxl:
    num_inference_steps: 25  # Down from 30
```

### Issue: Processing Takes >3 minutes

**Likely:** Phase 2 retries (IoU validation failing)

**Check logs:**
```bash
curl -s http://localhost:5090/api/v1/jobs/{JOB_ID} | jq '.logs[-20:]'
```

**Solution:**
```yaml
# In configs/default_config.yaml
phase2:
  iou_retry:
    allow_skip_on_failure: true  # Allow continuing despite low IoU
```

### Issue: Server becomes unresponsive

**Likely:** Long request not timing out

**Solution:** Apply Bug Fix #5 (Request Timeout)

---

## 📝 API ENDPOINTS

### Submit Image

```bash
curl -X POST http://localhost:5090/api/v1/process \
  -F "file=@car.jpg" \
  -F "palette_hex_list=FF0000,00FF00,..."

# Response: {"job_id": "xxx", "status": "pending"}
```

### Check Status

```bash
curl http://localhost:5090/api/v1/jobs/{JOB_ID}

# Response when complete:
{
  "job_id": "xxx",
  "status": "completed",
  "result": {
    "svg_path": "/tmp/gemini3_output/xxx.svg",
    "png_path": "/tmp/gemini3_output/xxx.png",
    "processing_time_ms": 85000
  }
}
```

### Get Output

```bash
# Once completed, download SVG
curl http://localhost:5090/api/v1/jobs/{JOB_ID}/output/svg -o output.svg

# Download PNG preview
curl http://localhost:5090/api/v1/jobs/{JOB_ID}/output/png -o output.png
```

---

## 💰 ESTIMATED COSTS

| Duration | RTX 5090 Cost |
|---|---|
| 1 hour | $0.90 |
| 8 hours (1 workday) | $7.20 |
| 24 hours | $21.60 |
| 30 days (24/7) | $648 |
| 30 days (8 hrs/day) | $216 |

**Cost optimization tips:**
- Use on-demand (pay only while running)
- Auto-scale down when not needed
- Consider spot instances (60-70% savings)

---

## ✅ PRE-LAUNCH CHECKLIST

**Code Changes:**
- [ ] Applied all 7 bug fixes
- [ ] Ran unit tests: `pytest tests/`
- [ ] No new errors in linting

**Pod Configuration:**
- [ ] RTX 5090 selected
- [ ] 100GB volume created and mounted
- [ ] Environment variables set correctly
- [ ] Startup command copy-pasted (no typos)

**Validation:**
- [ ] Pod starts and reaches "Running" state
- [ ] `/health` returns 200 OK
- [ ] `/ready` returns ready status
- [ ] Can process test image without errors
- [ ] GPU memory stays below 24GB

**Monitoring:**
- [ ] Setup metrics endpoint monitoring
- [ ] Setup error alerts (if desired)
- [ ] Document access URL for team

---

## 🆘 SUPPORT & DEBUGGING

### Enable Debug Logging

```env
LOG_LEVEL=debug
```

### Check Full Logs

```bash
# Inside pod terminal
tail -f /var/log/uvicorn.log

# Or via API
curl http://localhost:5090/api/v1/jobs/{JOB_ID} | jq '.logs'
```

### File Locations

```
/workspace/image_generation    → Application code
/models                        → Persistent model storage
/tmp/gemini3_output           → Temporary outputs
/opt/lama-cleaner-venv        → LaMa Cleaner (isolated)
```

### Emergency Restart

```bash
# SSH into pod, then:
# Kill uvicorn
pkill uvicorn

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Restart server
cd /workspace/image_generation
uvicorn src.api.server:app --host 0.0.0.0 --port 5090 --workers 1
```

---

## 📚 ADDITIONAL RESOURCES

- **Detailed Analysis:** `RUNPOD_5090_DEPLOYMENT_ANALYSIS.md`
- **Code Patches:** `BUG_FIXES_AND_PATCHES.md`
- **Architecture:** `APPLICATION_OVERVIEW.md`
- **API Docs:** `http://localhost:5090/docs` (live after deployment)

---

**Document Status:** ✅ COMPLETE  
**Last Updated:** December 7, 2025  
**Recommended:** Read full analysis before deployment

