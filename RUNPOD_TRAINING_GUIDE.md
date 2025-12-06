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
3. **Size**: `100 GB` (Recommended - minimum 50GB)
   - **Important**: Models require substantial disk space:
     - SDXL base model: ~14 GB
     - ControlNet Depth: ~2.5 GB
     - ControlNet Canny: ~2.5 GB
     - GroundingDINO: ~2.7 GB
     - SAM (Segment Anything): ~2.6 GB
     - RealESRGAN: ~200 MB
     - **Total: ~27 GB** (leaves ~23 GB for training data and cache)
   - If using 50 GB volume, you won't have space for training data
   - **Recommended: 100 GB** to avoid disk quota errors during setup
4. **Region**: Choose closest
5. Click **Create**
6. **Note Volume ID** (you'll need it)

### 1.2 Create GPU Pod

1. RunPod Dashboard → **Pods** → **New Pod**
2. **GPU**: Select your GPU:
   - **RTX 3090** or **A10G** (24GB VRAM) - Recommended
   - **RTX 5090** - Requires PyTorch 2.8+ (see note below)
   - **RTX 4090** - Compatible
   - **Avoid**: GPUs with less than 12GB VRAM
3. **Container Image**: 
   - **For all GPUs**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (Recommended)
   - **For RTX 5090**: 
     - **Recommended**: Use `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` and upgrade PyTorch to 2.8.0+ with CUDA 12.8+ in startup command (see below)
     - **Alternative**: Try `pytorch/pytorch:2.9.1-cuda12.9-cudnn9-runtime` (newer, verify availability)
   
   **Note**: PyTorch 2.8.0+ is now required for all GPUs (backward compatible). RTX 5090 requires CUDA 12.8+ support. Most reliable: Use 2.1 image + upgrade PyTorch in startup command (Recommended approach).
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
6. **Ports**: Add port `5090` (TCP) for RTX 5090, or `8000` (TCP) for other GPUs
   - **Note**: You can use any port you prefer. Just make sure the port in the startup command matches the port you configure here.
7. **Startup Command** (replace `YOUR_GITHUB_URL`):
   
   **For RTX 3090/A10G/RTX 4090 (PyTorch 2.8.0+)**:
   ```bash
   /bin/bash -c "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata && cd /workspace && rm -rf image_generation ZoeDepth && git clone YOUR_GITHUB_URL image_generation && git clone https://github.com/isl-org/ZoeDepth.git && cd /workspace/image_generation && pip install -q --upgrade torch>=2.8.0 torchvision>=0.23.0 && chmod +x scripts/install_dependencies.sh && bash scripts/install_dependencies.sh && export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:\$PYTHONPATH && python scripts/setup_model_volume.py --volume-path /models && uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
   ```
   
   **Note**: PyTorch 2.8.0+ is now required for all GPUs (backward compatible). Upgrading PyTorch before installing other dependencies ensures torchvision compatibility and prevents dtype mismatch errors.
   
   **For RTX 5090 (PyTorch 2.8.0+ required - sm_120 support)**:
   ```bash
   /bin/bash -c "export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata && cd /workspace && rm -rf image_generation ZoeDepth && git clone YOUR_GITHUB_URL image_generation && git clone https://github.com/isl-org/ZoeDepth.git && cd /workspace/image_generation && pip install -q --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 && chmod +x scripts/install_dependencies.sh && bash scripts/install_dependencies.sh && export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:\$PYTHONPATH && python scripts/setup_model_volume.py --volume-path /models && uvicorn src.api.server:app --host 0.0.0.0 --port 5090"
   ```
   
   **Note**: RTX 5090 requires PyTorch 2.8.0+ with CUDA 12.8+ support. CUDA 12.8 nightly builds (cu128) are confirmed to work with RTX 5090. If nightly fails, try cu129 stable builds (see troubleshooting section).
   
   **Important**: 
   - Replace `YOUR_GITHUB_URL` with your actual GitHub repository URL
   - RTX 5090 requires PyTorch 2.8+, so the RTX 5090 command upgrades PyTorch first
   - The installation script automatically handles xformers compatibility - if xformers is incompatible with your PyTorch/CUDA version, it will be automatically disabled and PyTorch's built-in SDPA will be used instead
   - Port 5090 is used for RTX 5090 in the example above - you can change it to any port you prefer, just make sure it matches the port you configure in step 6
8. **Pod Name**: `gemini3-training`
9. Click **Deploy**

**Wait 5-10 minutes** for pod to start and setup to complete.

**Alternative: If Startup Command Keeps Failing**

If you encounter container startup errors, use this simpler approach:

1. **Startup Command** (use this instead):
   ```bash
   sleep infinity
   ```

2. After pod starts, connect via terminal and run setup manually (see Part 2.3 below).

This avoids complex startup commands that can cause container errors.

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

### 2.2 Manual Setup (If Startup Command Failed)

If you used `sleep infinity` as startup command or automated setup failed, run setup manually:

```bash
# 1. Install git and system libraries (required for OpenCV)
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y -q git libgl1-mesa-glx libglib2.0-0 tzdata

# 2. Clone repositories
cd /workspace
rm -rf image_generation ZoeDepth  # Remove if exists from previous attempt
git clone YOUR_GITHUB_URL image_generation
git clone https://github.com/isl-org/ZoeDepth.git  # ZoeDepth doesn't have setup.py, add to PYTHONPATH

# 3. Install dependencies
cd /workspace/image_generation

# Upgrade PyTorch and torchvision together first (required for all GPUs with PyTorch 2.8.0+)
# This prevents dtype mismatch errors and ensures compatibility
pip install --upgrade torch>=2.8.0 torchvision>=0.23.0

# If using RTX 5090, use CUDA 12.8 nightly builds instead:
# pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# Note: CUDA 12.8 nightly builds (cu128) are confirmed to work with RTX 5090

# Use installation script to handle dependency conflicts (lama-cleaner)
chmod +x scripts/install_dependencies.sh
bash scripts/install_dependencies.sh

# Or install manually if script doesn't work:
# pip install -r requirements.txt
# pip install --no-deps lama-cleaner>=1.2.0
# pip install pydantic rich yacs omegaconf safetensors piexif || true

# 4. Set PYTHONPATH (include both image_generation and ZoeDepth)
export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:$PYTHONPATH

# 5. Setup models (takes 30-60 minutes)
python scripts/setup_model_volume.py --volume-path /models

# 6. Start server
# For RTX 5090, use port 5090; for other GPUs, use port 8000 (or any port you configured)
uvicorn src.api.server:app --host 0.0.0.0 --port 5090
```

**Note**: Keep terminal open - server must keep running. Or run in background:
```bash
# Replace 5090 with your configured port (8000 for other GPUs)
nohup uvicorn src.api.server:app --host 0.0.0.0 --port 5090 > server.log 2>&1 &
tail -f server.log  # View logs
```

### 2.3 Get Pod URL

1. In pod page, find **HTTP Service** URL
2. Format: 
   - For port 5090: `https://xxxxx-5090.proxy.runpod.net` or `http://xxx.xxx.xxx.xxx:5090`
   - For port 8000: `https://xxxxx-8000.proxy.runpod.net` or `http://xxx.xxx.xxx.xxx:8000`
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

### RTX 5090 GPU Compatibility

**Error**: `Warning: RTX 5090s only support Pytorch 2.8 and above`

**Cause**: RTX 5090 requires PyTorch 2.8+, but container image has PyTorch 2.1.0.

**Solution Options**:

1. **Upgrade PyTorch in Pod** (Recommended for RTX 5090):
   - Use PyTorch 2.1 image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
   - Use the RTX 5090 startup command (above) which auto-upgrades PyTorch to 2.8.0+ with CUDA 12.8+
   - Or manually upgrade if pod is already running:
   ```bash
   # RTX 5090 requires PyTorch 2.8.0+ with CUDA 12.8+ (cu128)
   pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install -r requirements.txt  # Reinstall to ensure compatibility
   ```
   
2. **Try PyTorch 2.9 Image** (If Available):
   - Try: `pytorch/pytorch:2.9.1-cuda12.9-cudnn9-runtime`
   - Or check Docker Hub for latest 2.8+ tags: https://hub.docker.com/r/pytorch/pytorch/tags
   - Note: Many 2.8 image tags don't exist - if image pull fails, use Option 1 instead
   
3. **Switch to Compatible GPU** (Alternative):
   - Use RTX 3090, A10G, or RTX 4090 instead
   - These work with PyTorch 2.1.0 image
   - No PyTorch upgrade needed

**Note**: If you see the RTX 5090 warning, you MUST upgrade PyTorch before training will work.

**Error**: `CUDA error: no kernel image is available for execution on the device`

**Cause**: PyTorch was not compiled for your GPU's compute capability. RTX 5090 requires PyTorch 2.8.0+ with CUDA 12.8+ support (compute capability sm_120). Versions 2.5.1, 2.6.x, and 2.7.x do NOT support RTX 5090.

**Solution - Step-by-Step Upgrade Guide**:

1. **Check current PyTorch version and CUDA support**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
   ```
   
   **Expected output for RTX 5090**: Should show PyTorch 2.8.0+ and CUDA 12.8+ (or 12.9+)

2. **Upgrade PyTorch to 2.8.0+ with CUDA 12.8** (Recommended - Nightly Build):
   ```bash
   # CUDA 12.8 nightly builds are confirmed to work with RTX 5090 (recommended)
   pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```
   
   **Why CUDA 12.8 nightly?**
   - RTX 5090 (sm_120) requires PyTorch 2.8.0+ with CUDA 12.8+ support
   - CUDA 12.8 nightly builds (cu128) are confirmed to work with RTX 5090
   - Nightly builds include the latest fixes and optimizations

3. **Verify installation and GPU compatibility**:
   ```bash
   # Check PyTorch version and CUDA version
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   
   # Test CUDA tensor operations
   python -c "import torch; x = torch.zeros(1).cuda(); print('CUDA test passed:', x.device, 'GPU:', torch.cuda.get_device_name(0))"
   ```
   
   **Expected output**: Should show `cuda:0` and your RTX 5090 GPU name without errors

4. **If CUDA 12.8 nightly doesn't work**, try CUDA 12.9 stable build:
   ```bash
   # Fallback: Try CUDA 12.9 stable build
   pip install --upgrade 'torch>=2.8.0' 'torchvision>=0.23.0' --index-url https://download.pytorch.org/whl/cu129
   ```
   
   **CUDA 12.8 vs 12.9**:
   - **CUDA 12.8 (nightly)**: Recommended, confirmed to work with RTX 5090
   - **CUDA 12.9 (stable)**: Alternative if nightly has issues, also supports RTX 5090
   - Both require PyTorch 2.8.0+

5. **If still failing**, perform clean uninstall and reinstall:
   ```bash
   # Uninstall existing PyTorch
   pip uninstall torch torchvision torchaudio -y
   
   # Clear pip cache (optional but recommended)
   pip cache purge
   
   # Reinstall with CUDA 12.8 nightly (recommended)
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   
   # Verify installation
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); x = torch.zeros(1).cuda(); print('CUDA test passed')"
   ```

6. **Reinstall other dependencies** (if needed):
   ```bash
   # Reinstall requirements to ensure compatibility
   pip install -r requirements.txt
   ```

**Important Notes**:
- **Backward Compatibility**: PyTorch 2.8.0+ with CUDA 12.8+ is backward compatible with older GPUs (RTX 3090, A10G, RTX 4090). You can use the same installation on all GPUs.
- **After upgrading PyTorch**: Restart the training job. The server doesn't need to be restarted, but the training process does.
- **Version Requirements**: RTX 5090 (sm_120) requires PyTorch 2.8.0 or higher. Versions 2.5.1, 2.6.x, and 2.7.x do NOT support RTX 5090.
- **CUDA Driver**: Ensure your NVIDIA driver supports CUDA 12.8+. Check with `nvidia-smi` - driver version 550.54.15+ is recommended.

**Troubleshooting Common Issues**:

- **"No module named 'torch'" after upgrade**: Restart Python process or pod
- **"CUDA out of memory"**: This is different from compatibility error - reduce batch size
- **"CUDA driver version is insufficient"**: Update NVIDIA drivers to 550.54.15+
- **Dependency conflicts**: Use `pip install --upgrade --force-reinstall` if needed

### Xformers Import Error (Fixed)

**Error**: `RuntimeError: Failed to import diffusers.loaders.ip_adapter because of the following error: /opt/conda/lib/python3.10/site-packages/xformers/flash_attn_3/_C.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`

**Cause**: xformers is installed but incompatible with your PyTorch/CUDA version. This is a common issue when PyTorch and xformers versions don't match.

**Solution**: 
- The installation script (`scripts/install_dependencies.sh`) now automatically detects and handles this issue
- If xformers is incompatible, it will be automatically uninstalled
- The application will use PyTorch's built-in SDPA (Scaled Dot Product Attention) instead, which works just as well
- If you see this error during server startup, the code will now catch it and provide a helpful error message

**Manual Fix** (if needed):
```bash
# Uninstall incompatible xformers
pip uninstall xformers -y

# Restart server - it will use PyTorch SDPA instead
uvicorn src.api.server:app --host 0.0.0.0 --port 5090
```

**Note**: xformers is optional - the application works perfectly fine without it using PyTorch's built-in attention mechanisms.

### Container Stuck in Restart Loop

**Symptoms**: Container shows "start container" repeatedly, keeps restarting, never fully starts.

**Cause**: Startup command is failing immediately, causing container to crash and restart in a loop.

**Solutions**:

1. **Check Pod Logs** (First Step):
   - In RunPod Dashboard → Your Pod
   - Click **Logs** tab or **View Logs**
   - Look for error messages that show why container is failing
   - Common causes:
     - RTX 5090 PyTorch version warning (if using 2.1 image)
     - Startup command syntax error
     - Missing dependencies
     - Git clone failure

2. **Use Simple Startup Command** (Break the Loop):
   - Stop the pod: Dashboard → Pod → **Stop**
   - Wait for it to stop
   - Edit pod settings or create new pod
   - Change **Startup Command** to:
     ```bash
     sleep infinity
     ```
   - Start pod
   - Connect via terminal and set up manually (see Part 2.2)

3. **Fix RTX 5090 Issue** (If using RTX 5090):
   - If you see PyTorch version warning in logs:
   - Use `sleep infinity` as startup command
   - Connect terminal and upgrade PyTorch to 2.8.0+ manually:
     ```bash
     # CUDA 12.8 nightly builds are confirmed to work with RTX 5090 (recommended)
     pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
     # If that fails, try CUDA 12.9 stable build:
     # pip install --upgrade 'torch>=2.8.0' 'torchvision>=0.23.0' --index-url https://download.pytorch.org/whl/cu129
     ```
   - Then continue with setup

**Prevention**: Test with `sleep infinity` startup command first, then manually run setup to see any errors clearly.

### RunPod Container Startup Errors

**Error**: `cannot join network namespace of a non running container` or `failed to create shim task`

**Cause**: RunPod infrastructure issue - container state is corrupted or stuck.

**Solutions** (try in order):

1. **Delete and Recreate Pod** (Most Reliable):
   - Dashboard → Pods → Find your pod
   - Click **Delete** or **Terminate**
   - Wait for deletion to complete
   - Create a new pod with same settings
   - This clears all container state

2. **Try Different GPU/Region**:
   - Sometimes specific GPU types or regions have issues
   - Try a different GPU (e.g., RTX 3090 instead of A10G)
   - Try a different region/data center

3. **Use Simpler Startup Command** (Temporary):
   - Use `sleep infinity` as startup command
   - Connect via terminal after pod starts
   - Run setup commands manually (see Part 2)

4. **Check RunPod Status**:
   - Visit: https://status.runpod.io
   - Check if RunPod is experiencing issues
   - Wait if there's a known outage

5. **Contact RunPod Support**:
   - If errors persist, contact RunPod support via dashboard
   - Provide error messages and pod ID

**Prevention**: If pod keeps failing, use manual setup approach (startup: `sleep infinity`) instead of automated startup command.

### Server Not Running

```bash
cd /workspace/image_generation
export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:$PYTHONPATH
# For RTX 5090, use port 5090; for other GPUs, use port 8000
uvicorn src.api.server:app --host 0.0.0.0 --port 5090
```

### CUDA Out of Memory

- Reduce batch size to `1`
- Resize images to 512x512
- Reduce LoRA rank to `16`

### Can't Access Web UI

1. Check server is running: `ps aux | grep uvicorn`
2. Verify your configured port (5090 for RTX 5090, 8000 for others) is exposed in pod settings
3. Try different network/VPN

### Training Fails - "Input type (float) and bias type (c10::Half) should be the same"

**Error**: `Training failed: Input type (float) and bias type (c10::Half) should be the same`

**Cause**: This is a dtype mismatch error. The model weights are in float16 (half precision) but input tensors are in float32. This commonly occurs after upgrading PyTorch to 2.8.0+ if PyTorch and torchvision weren't upgraded together, or if the training code doesn't properly handle dtype conversion.

**Solution**:

1. **Ensure PyTorch and torchvision are upgraded together** (most common fix):
   ```bash
   # Uninstall and reinstall PyTorch/torchvision together
   pip uninstall torch torchvision -y
   pip install torch>=2.8.0 torchvision>=0.23.0
   pip install -r requirements.txt  # Reinstall other dependencies
   ```

2. **If using RTX 4090/3090/A10G**, ensure you used the updated startup command that upgrades PyTorch before installing dependencies.

3. **If the error persists**, the training code may need to be updated to ensure input tensors match model dtype:
   - Check that inputs are converted to the same dtype as the model (typically float16)
   - Ensure VAE encoding outputs match UNet dtype
   - Verify mixed precision training is configured correctly

4. **Temporary workaround** (if code fix isn't available):
   ```bash
   # Force reinstall with compatible versions
   pip uninstall torch torchvision -y
   pip cache purge
   pip install torch==2.8.0 torchvision==0.23.0
   pip install -r requirements.txt --force-reinstall
   ```

**Note**: This error typically indicates a version mismatch between PyTorch and torchvision, or a bug in the training code's dtype handling. The updated startup commands ensure PyTorch/torchvision are upgraded together to prevent this issue.

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
export PYTHONPATH=/workspace/image_generation:$PYTHONPATH
python scripts/setup_model_volume.py --volume-path /models
```

### Disk Quota Exceeded

**Error**: `OSError: [Errno 122] Disk quota exceeded`

**Cause**: RunPod volume or container disk is full. Models require ~20-30GB of space.

**Solutions**:

1. **Check Disk Usage**:
   ```bash
   df -h
   du -sh /models
   ```

2. **Increase Volume Size** (if using persistent volume):
   - RunPod Dashboard → Volumes → Your Volume
   - Click **Resize**
   - Increase to at least 50GB (recommended: 100GB)

3. **Clean Up Unnecessary Files**:
   ```bash
   # Remove old model downloads if any
   rm -rf /models/*.tmp
   # Clear pip cache
   pip cache purge
   # Clear system package cache
   apt-get clean
   ```

4. **Use Larger Container Disk**:
   - When creating pod, increase Container Disk size
   - Minimum: 50GB, Recommended: 100GB

### Missing loguru Dependency

**Error**: `ModuleNotFoundError: No module named 'loguru'`

**Cause**: lama-cleaner requires loguru but it's not in our requirements.

**Solution**:
```bash
pip install loguru
```

Or the installation script now includes it automatically.

### Container Stuck at tzdata Configuration Prompt

**Error**: Container stops and waits at: `Please select the geographic area in which you live` (tzdata configuration)

**Cause**: tzdata package installation requires interactive input, blocking automated setup.

**Solution**: 
- The startup commands now include `export DEBIAN_FRONTEND=noninteractive` and `tzdata` in apt-get install
- If stuck in current pod, restart with updated startup command
- Or set timezone non-interactively:
  ```bash
  export DEBIAN_FRONTEND=noninteractive
  export TZ=UTC
  ln -fs /usr/share/zoneinfo/$TZ /etc/localtime
  dpkg-reconfigure -f noninteractive tzdata
  ```

### Missing libGL.so.1 (OpenCV Issue)

**Error**: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

**Cause**: Missing system graphics libraries required by OpenCV. This prevents the server from starting and blocks OpenCV imports.

**Solution**:
```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

**Then restart server**:
```bash
cd /workspace/image_generation
export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:$PYTHONPATH
# For RTX 5090, use port 5090; for other GPUs, use port 8000
uvicorn src.api.server:app --host 0.0.0.0 --port 5090
```

**Note**: The installation script and startup commands now include this automatically. If you're using the manual setup, install these libraries first.

### ModuleNotFoundError: No module named 'src'

If you see this error, set PYTHONPATH:

```bash
export PYTHONPATH=/workspace/image_generation:$PYTHONPATH
```

Then retry your command. To make it permanent for the session, add to your startup command or run it before each Python command.

### Restart Pod

1. Dashboard → Pod → **Stop**
2. Wait for stop
3. Click **Start**
4. Reconnect terminal
5. Restart server:
   ```bash
   cd /workspace/image_generation
   # Install system libraries if missing
   apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
   export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:$PYTHONPATH
   # For RTX 5090, use port 5090; for other GPUs, use port 8000
   uvicorn src.api.server:app --host 0.0.0.0 --port 5090
   ```

**Note**: Data in `/workspace` is lost on restart, but `/models` volume persists.

### Git Clone Fails - "destination path already exists"

**Error**: `fatal: destination path 'image_generation' already exists` or `fatal: destination path 'ZoeDepth' already exists`

**Cause**: Previous startup attempt created the directory but failed partway through.

**Solution**:
```bash
cd /workspace
rm -rf image_generation ZoeDepth
git clone YOUR_GITHUB_URL image_generation
git clone https://github.com/isl-org/ZoeDepth.git
```

The updated startup commands now include `rm -rf image_generation ZoeDepth &&` before git clone to prevent this issue.

### Package Installation Issues

**Error**: `ERROR: Could not find a version that satisfies the requirement zoedepth>=0.1.0`

**Cause**: ZoeDepth is not available on PyPI - it must be installed from GitHub.

**Solution**: 
- ZoeDepth doesn't have `setup.py` or `pyproject.toml`, so it can't be installed via pip
- It must be cloned and added to PYTHONPATH instead:
  ```bash
  cd /workspace
  git clone https://github.com/isl-org/ZoeDepth.git
  export PYTHONPATH=/workspace/ZoeDepth:$PYTHONPATH
  ```
- The startup commands in the guide now include cloning ZoeDepth automatically
- For manual setup, clone ZoeDepth and add `/workspace/ZoeDepth` to PYTHONPATH

**Error**: `ERROR: ResolutionImpossible: Cannot install -r requirements.txt (line 18) and diffusers>=0.30.0 because these package versions have conflicting dependencies`

**Cause**: lama-cleaner requires `diffusers==0.16.1` but our code needs `diffusers>=0.30.0` for SDXL ControlNet support.

**Solution**: 
- Use the provided installation script: `bash scripts/install_dependencies.sh`
- Or install manually:
  ```bash
  pip install -r requirements.txt
  pip install --no-deps lama-cleaner>=1.2.0
  pip install pydantic rich yacs omegaconf safetensors piexif || true
  ```
- The `--no-deps` flag installs lama-cleaner without its conflicting dependencies
- Our code uses newer diffusers version which works fine with lama-cleaner's ModelManager

**Error**: `ERROR: Could not find a version that satisfies the requirement colorspacious>=2.0.0`

**Cause**: colorspacious version 2.0.0 doesn't exist - latest is 1.1.2.

**Solution**: 
- The `requirements.txt` has been updated to use `colorspacious>=1.1.2`
- If you have an old version, update it manually:
  ```bash
  sed -i 's/colorspacious>=2.0.0/colorspacious>=1.1.2/' requirements.txt
  pip install -r requirements.txt
  ```

If `controlnet-aux` installation fails:
- The version has been fixed in requirements.txt (>=0.0.10)
- If you have an old requirements.txt, update it or run:
  ```bash
  sed -i 's/controlnet-aux>=0.4.0/controlnet-aux>=0.0.10/' requirements.txt
  pip install -r requirements.txt
  ```

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
export PYTHONPATH=/workspace/image_generation:/workspace/ZoeDepth:$PYTHONPATH
# For RTX 5090, use port 5090; for other GPUs, use port 8000
uvicorn src.api.server:app --host 0.0.0.0 --port 5090

# Start server in background
nohup uvicorn src.api.server:app --host 0.0.0.0 --port 5090 > server.log 2>&1 &
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
