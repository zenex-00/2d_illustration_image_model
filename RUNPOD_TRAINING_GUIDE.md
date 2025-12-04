# RunPod Training & Testing Guide (Streamlined)

Quick guide to train and test LoRA models on RunPod. Assumes code is on GitHub.

---

## Prerequisites

- [ ] RunPod account with payment method
- [ ] Code pushed to GitHub repository
- [ ] Training data ready (10+ input/target image pairs)
- [ ] GitHub repository URL (e.g., `https://github.com/username/image_generation`)

---

## Part 1: Setup (5 minutes)

### 1.1 Create Persistent Volume

1. RunPod Dashboard → **Volumes** → **Create Volume**
2. **Name**: `gemini3-models`
3. **Size**: `50 GB`
4. **Region**: Choose closest
5. Click **Create**
6. **Note Volume ID** (you'll need it)

### 1.2 Create GPU Pod

1. RunPod Dashboard → **Pods** → **New Pod**
2. **GPU**: Select **RTX 3090** or **A10G** (24GB VRAM)
3. **Container Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
4. **Volume Mounts**:
   - Select your volume: `gemini3-models`
   - **Mount Path**: `/models`
5. **Environment Variables** (add each):
   ```
   PYTHONUNBUFFERED=1
   CUDA_VISIBLE_DEVICES=0
   MODEL_VOLUME_PATH=/models
   API_OUTPUT_DIR=/tmp/gemini3_output
   TRAIN_DATA_ROOT=/workspace/training_data
   TRAIN_OUTPUT_ROOT=/workspace/training_output
   ```
6. **Ports**: Add port `8000` (TCP)
7. **Startup Command** (replace `YOUR_GITHUB_URL`):
   ```bash
   cd /workspace && git clone YOUR_GITHUB_URL image_generation && cd image_generation && pip install -q -r requirements.txt && python scripts/setup_model_volume.py --volume-path /models && uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```
8. **Pod Name**: `gemini3-training`
9. Click **Deploy**

**Wait 5-10 minutes** for pod to start and setup to complete.

---

## Part 2: Verify Setup (2 minutes)

### 2.1 Check Pod Status

1. Pod should show **"Running"**
2. Click **Connect** → **Terminal**
3. Verify setup completed:
   ```bash
   ls /workspace/image_generation/src
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ls /models
   ```

**Expected**: See directories, `CUDA: True`, model folders exist.

### 2.2 Get Pod URL

1. In pod page, find **HTTP Service** URL
2. Format: `https://xxxxx-8000.proxy.runpod.net` or `http://xxx.xxx.xxx.xxx:8000`
3. Test: Open `YOUR_POD_URL/health` in browser
4. Should return: `{"status":"healthy",...}`

---

## Part 3: Prepare Training Data (5 minutes)

### 3.1 Organize Locally

Create this structure on your computer:
```
training_data/
├── inputs/
│   ├── 0001_car.jpg
│   ├── 0002_car.jpg
│   └── ... (10+ images)
└── targets/
    ├── 0001_car.jpg  (same filename as input)
    ├── 0002_car.jpg
    └── ... (same count as inputs)
```

**Requirements**:
- Minimum 10 pairs
- Input and target must have **same filename**
- Images: JPEG/PNG, 512x512 to 2048x2048 recommended

### 3.2 Upload to RunPod

**Method 1: File Manager (Easiest)**
1. Pod → **Files** tab
2. Navigate to `/workspace/`
3. Create folders: `training_data/inputs` and `training_data/targets`
4. Upload input images to `inputs/`
5. Upload target images to `targets/`

**Method 2: Zip Upload (Faster)**
1. Zip your `training_data` folder locally
2. Upload `training_data.zip` to `/workspace/`
3. In terminal:
   ```bash
   cd /workspace && unzip training_data.zip
   ```

**Verify**:
```bash
ls /workspace/training_data/inputs | wc -l
ls /workspace/training_data/targets | wc -l
```
Both should show same number (≥10).

---

## Part 4: Training (15-30 minutes)

### 4.1 Access Training UI

1. Open: `YOUR_POD_URL/ui/training`
2. Should see training form

### 4.2 Submit Training Job

1. **Upload Input Images**: Click "Choose Files" → Select all input images
2. **Upload Target Images**: Click "Choose Files" → Select all target images (same count!)
3. **Training Parameters** (defaults are fine for first run):
   - Learning Rate: `0.0001`
   - Batch Size: `1`
   - Epochs: `10`
   - LoRA Rank: `32`
   - LoRA Alpha: `16`
   - Validation Split: `0.2`
4. Click **"Start Training"**

### 4.3 Monitor Progress

- Page redirects to job status page
- Status: `pending` → `processing` → `completed`
- Watch logs for:
  - `"Found X image pairs"`
  - `"Starting epoch 1/10"`
  - `"Step X/Y: loss=0.XXXX"` (loss should decrease)
  - `"Training completed!"`

**Training Time**: 30 minutes to 2 hours (depends on data size and epochs)

**Don't close browser tab** - keep it open to monitor.

---

## Part 5: Testing (5 minutes)

### 5.1 Locate Trained Model

When training completes:
- Status shows `completed`
- Message: `"Weights saved to: /workspace/training_output/{job_id}/vector_style_lora.safetensors"`
- **Note the job_id** from URL or message

**Verify file exists**:
```bash
ls -lh /workspace/training_output/*/vector_style_lora.safetensors
```

### 5.2 Test via Web UI

1. Go to: `YOUR_POD_URL/ui/inference`
2. **Select LoRA**: Choose your trained model from dropdown
3. **Upload Test Image**: Choose a vehicle photo
4. Click **"Process Image"**
5. Wait 1-3 minutes
6. Download results (SVG and PNG)

### 5.3 Test via API (Alternative)

```bash
curl -X POST "YOUR_POD_URL/api/v1/process" \
  -F "file=@/path/to/test_image.jpg" \
  -F "lora_checkpoint=/workspace/training_output/JOB_ID/vector_style_lora.safetensors"
```

Replace `JOB_ID` with your actual job ID.

---

## Part 6: Download Results

### 6.1 Download Trained Model

**File Manager**:
1. Pod → **Files** → `/workspace/training_output/{job_id}/`
2. Right-click `vector_style_lora.safetensors` → **Download**

**Terminal** (if you have SSH):
```bash
# On your local computer:
scp root@POD_IP:/workspace/training_output/JOB_ID/vector_style_lora.safetensors ./
```

### 6.2 Download Test Outputs

From inference job page, click **"Download SVG"** and **"Download PNG"** buttons.

---

## Troubleshooting

### Server Not Running

```bash
cd /workspace/image_generation
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### CUDA Out of Memory

- Reduce batch size to `1`
- Resize images to 512x512
- Reduce LoRA rank to `16`

### Can't Access Web UI

1. Check server is running: `ps aux | grep uvicorn`
2. Verify port 8000 is exposed in pod settings
3. Try different network/VPN

### Training Fails - "Insufficient image pairs"

```bash
# Verify counts match:
ls /workspace/training_data/inputs | wc -l
ls /workspace/training_data/targets | wc -l
# Both must be ≥10 and equal
```

### Models Not Found

```bash
# Re-download models:
cd /workspace/image_generation
python scripts/setup_model_volume.py --volume-path /models
```

### Restart Pod

1. Dashboard → Pod → **Stop**
2. Wait for stop
3. Click **Start**
4. Reconnect terminal
5. Restart server:
   ```bash
   cd /workspace/image_generation
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```

**Note**: Data in `/workspace` is lost on restart, but `/models` volume persists.

---

## Quick Command Reference

```bash
# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check training data
ls /workspace/training_data/inputs | wc -l
ls /workspace/training_data/targets | wc -l

# Find trained model
find /workspace/training_output -name "*.safetensors"

# Start server
cd /workspace/image_generation
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Start server in background
nohup uvicorn src.api.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
tail -f server.log  # View logs
```

---

## Cost Management

- **Stop pod when not using**: Dashboard → Pod → **Stop**
- **Estimated costs**:
  - Training (2 hours): $1-4
  - Testing (30 min): $0.25-1
  - Volume (50GB/month): ~$5/month
- **You only pay when pod is running**

---

## Checklist

**Before Training**:
- [ ] Code on GitHub
- [ ] Volume created (50GB)
- [ ] Pod running
- [ ] Can access `YOUR_POD_URL/health`
- [ ] Training data uploaded (10+ pairs)
- [ ] File counts match

**During Training**:
- [ ] Job submitted
- [ ] Status: `processing`
- [ ] Logs updating
- [ ] Loss decreasing

**After Training**:
- [ ] Status: `completed`
- [ ] Model file exists
- [ ] Tested via UI/API
- [ ] Downloaded model
- [ ] Stopped pod (save costs)

---

**Total Time**: ~30-45 minutes setup + 30min-2hr training + 5min testing

**Last Updated**: Streamlined version  
**Version**: 3.0.0
