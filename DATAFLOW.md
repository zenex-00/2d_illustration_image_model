# Dataflow: Gemini 3 Pro Vehicle-to-Vector Pipeline

## Overview

This document describes in detail how data flows through the Gemini 3 Pro Vehicle-to-Vector Pipeline application, from initial input to final output, across all three access methods (CLI, Python API, and Web API/UI).

---

## High-Level Dataflow

```
Input Image (JPEG/PNG/WebP)
    ↓
[Entry Point: CLI / Python API / Web API]
    ↓
Validation & Preprocessing
    ↓
[Phase I: Semantic Sanitization]
    ↓
[Phase II: Generative Steering]
    ↓
[Phase III: Chromatic Enforcement]
    ↓
[Phase IV: Vector Reconstruction]
    ↓
Output (SVG + PNG Preview)
```

---

## Entry Points & Initial Data Flow

### 1. CLI Entry Point

**File**: `cli/main.py`

**Flow**:
```
Command Line Arguments
    ↓
Click CLI Parser
    ↓
Config Loading (optional --config)
    ↓
Pipeline Initialization (Gemini3Pipeline)
    ↓
File Path Validation
    ↓
Image Loading (load_image())
    ↓
Pipeline Processing (process_image())
    ↓
File Output (save_image(), SVG write)
```

**Data Structures**:
- **Input**: File path (string) → Validated file path
- **Config**: YAML file → `Config` object
- **Image**: File → NumPy array (RGB, uint8, shape: [H, W, 3])

**Example Flow**:
```python
# User command: python -m cli.main process input.jpg output.svg
1. Click parses arguments → input_path="input.jpg", output_svg="output.svg"
2. Config loaded from default_config.yaml or --config file
3. Pipeline initialized: pipeline = Gemini3Pipeline(config_path)
4. Image loaded: image = load_image("input.jpg") → numpy.ndarray
5. Processing: svg_xml, metadata = pipeline.process_image(...)
6. SVG saved: with open("output.svg", 'w') as f: f.write(svg_xml)
```

---

### 2. Python API Entry Point

**File**: `src/pipeline/orchestrator.py`

**Flow**:
```
Python Code Call
    ↓
Pipeline Initialization (Gemini3Pipeline)
    ↓
process_image() Method Call
    ↓
Correlation ID Generation
    ↓
Config Override Validation
    ↓
Image Loading
    ↓
Pipeline Processing
    ↓
Return Tuple (svg_xml, metadata)
```

**Data Structures**:
- **Input Parameters**:
  - `input_image_path`: str → File path
  - `palette_hex_list`: Optional[List[str]] → 15 hex colors
  - `output_svg_path`: Optional[str] → Output path
  - `config_overrides`: Optional[Dict] → Phase configuration
- **Output**: `Tuple[str, Dict[str, Any]]`
  - `svg_xml`: str → SVG XML string
  - `metadata`: dict → Processing metadata

**Example Flow**:
```python
pipeline = Gemini3Pipeline()
svg_xml, metadata = pipeline.process_image(
    input_image_path="input.jpg",
    output_svg_path="output.svg",
    palette_hex_list=["#FF0000", "#00FF00", ...]
)
# svg_xml: "<svg xmlns=...>...</svg>"
# metadata: {"correlation_id": "...", "total_processing_time_ms": 1234, ...}
```

---

### 3. Web API Entry Point (Synchronous)

**File**: `src/api/server.py` → `POST /api/v1/process`

**Flow**:
```
HTTP POST Request (multipart/form-data)
    ↓
FastAPI Route Handler
    ↓
File Upload Validation (validate_uploaded_image)
    ↓
Palette Parsing (if provided)
    ↓
Temp File Creation (NamedTemporaryFile)
    ↓
Pipeline Processing (process_image)
    ↓
Output File Generation (OUTPUT_DIR)
    ↓
Response JSON (ProcessImageResponse)
```

**Data Structures**:
- **HTTP Request**:
  - `file`: `UploadFile` → PIL Image → Temp file path
  - `palette_hex_list`: Optional[str] → Comma-separated hex colors
- **Validation**: PIL Image → Validated dimensions, format, size
- **Temp File**: `tempfile.NamedTemporaryFile` → Path string
- **Response**: JSON with URLs and metadata

**Example Flow**:
```
1. Client uploads image via POST /api/v1/process
2. FastAPI receives UploadFile object
3. validate_uploaded_image() checks:
   - File size < 50MB
   - Format in [JPEG, PNG, WebP]
   - Dimensions < 4096px
4. Image saved to temp file: /tmp/tmpXXXXXX.png
5. Pipeline processes: pipeline.process_image(input_path, ...)
6. Outputs saved: OUTPUT_DIR/{correlation_id[:8]}.svg and .png
7. Response: {
     "status": "success",
     "svg_url": "/api/v1/download/abc12345.svg",
     "png_preview_url": "/api/v1/download/abc12345.png",
     "processing_time_ms": 1234,
     ...
   }
```

---

### 4. Web API Entry Point (Asynchronous)

**File**: `src/api/server.py` → `POST /api/v1/jobs`

**Flow**:
```
HTTP POST Request
    ↓
File Validation
    ↓
Job Creation (JobQueue.create_job)
    ↓
Background Task Scheduling (BackgroundTasks)
    ↓
Immediate Response (job_id)
    ↓
[Background Processing]
    ↓
Status Polling (GET /api/v1/jobs/{job_id})
    ↓
Result Retrieval
```

**Data Structures**:
- **Job Creation**:
  - `job_id`: str (UUID) → Generated unique ID
  - `Job` object: status=PENDING, metadata={input_path, ...}
- **Background Task**: `process_job_background()` function
- **Status Updates**: Job status → PENDING → PROCESSING → COMPLETED/FAILED

**Example Flow**:
```
1. POST /api/v1/jobs with image file
2. Job created: job_id = "abc-123-def-456"
3. Response: {"job_id": "abc-123-def-456", "status": "pending", ...}
4. Background task starts processing
5. Client polls: GET /api/v1/jobs/abc-123-def-456
   - Returns: {"status": "processing", ...}
6. Processing completes
7. Client polls again: {"status": "completed", "result_url": "...", ...}
```

---

### 5. Web UI Entry Point

**File**: `src/api/server.py` → `POST /ui/inference`

**Flow**:
```
Browser Form Submission
    ↓
HTMX/FastAPI POST Handler
    ↓
File Validation
    ↓
Config Override Construction (from form toggles)
    ↓
Job Creation
    ↓
Background Processing
    ↓
Redirect to Job Status Page
    ↓
HTMX Polling (every 2-3 seconds)
    ↓
Status Updates (inference_job_partial.html)
    ↓
Results Display
```

**Data Structures**:
- **Form Data**: HTML form → FastAPI Form data
  - `file`: UploadFile
  - `phase1_enabled`: bool
  - `phase2_enabled`: bool
  - `palette_hex_list`: Optional[str]
  - `lora_checkpoint`: Optional[str]
- **Config Overrides**: Form data → Nested dict structure
- **HTMX Updates**: Partial HTML templates

**Example Flow**:
```
1. User fills form on /ui/inference
2. Submits with image and phase toggles
3. POST /ui/inference processes:
   - Validates image
   - Builds config_overrides from form:
     {
       "phases": {
         "phase1": {"enabled": True},
         "phase2": {"enabled": True, "controlnet": {...}}
       }
     }
4. Job created and background processing starts
5. Redirect to /ui/inference/jobs/{job_id}
6. Page loads with HTMX polling:
   <div hx-get="/ui/inference/jobs/{job_id}/status" 
        hx-trigger="every 2s">
7. Status updates render via inference_job_partial.html
8. When completed, shows download links
```

---

## Phase-by-Phase Data Transformation

### Phase I: Semantic Sanitization

**Entry**: `src/pipeline/orchestrator.py:180` → `phase1.sanitize()`

**Input Data**:
- **Type**: `numpy.ndarray`
- **Shape**: `[H, W, 3]` (RGB, uint8)
- **Values**: 0-255 pixel values
- **Example**: `(1024, 2048, 3)` array representing vehicle photo

**Processing Flow**:
```
Raw Image (numpy.ndarray [H,W,3])
    ↓
[GroundingDINO Detector]
    ↓
Bounding Boxes (List[Dict])
    ↓
[SAM Segmenter]
    ↓
Binary Masks (numpy.ndarray [H,W] bool)
    ↓
[LaMa Inpainter]
    ↓
Clean Plate Image (numpy.ndarray [H,W,3])
    ↓
[IoU Validation]
    ↓
Validated Clean Plate
```

**Data Structures at Each Stage**:

1. **GroundingDINO Output**:
   ```python
   detections = [
       {
           "box": [x1, y1, x2, y2],  # Bounding box coordinates
           "score": 0.95,             # Confidence score
           "label": "car logo"        # Detected element type
       },
       ...
   ]
   ```

2. **SAM Output**:
   ```python
   masks = numpy.ndarray  # Shape: [H, W], dtype: bool
   # True = prohibited element, False = keep
   ```

3. **LaMa Output**:
   ```python
   clean_plate = numpy.ndarray  # Shape: [H, W, 3], dtype: uint8
   # Inpainted image with prohibited elements removed
   ```

4. **Validation**:
   ```python
   iou = float  # Intersection over Union score
   is_valid = iou > 0.97  # Quality threshold
   ```

**Output Data**:
- **Type**: `numpy.ndarray`
- **Shape**: `[H, W, 3]` (same as input)
- **Metadata**: `dict` with processing time, detected elements, IoU score

**Code Reference**: `src/phase1_semantic_sanitization/sanitizer.py:37-136`

---

### Phase II: Generative Steering

**Entry**: `src/pipeline/orchestrator.py:233` → `phase2.generate()`

**Input Data**:
- **Type**: `numpy.ndarray` (clean plate from Phase I)
- **Shape**: `[H, W, 3]` (RGB, uint8)

**Processing Flow**:
```
Clean Plate Image
    ↓
[Background Remover (BiRefNet)]
    ↓
Vehicle-Only Image + Alpha Mask
    ↓
[Depth Estimator (ZoeDepth)]
    ↓
Depth Map (numpy.ndarray [H,W] float32)
    ↓
[Edge Detector (Canny)]
    ↓
Edge Map (numpy.ndarray [H,W] uint8)
    ↓
[SDXL Generator with Multi-ControlNet]
    ↓
Vector-Style Raster (numpy.ndarray [H,W,3])
    ↓
[IoU Validation with Auto-Retry]
    ↓
Validated Vector Raster
```

**Data Structures at Each Stage**:

1. **Background Removal**:
   ```python
   vehicle_image = numpy.ndarray  # Shape: [H, W, 4], dtype: uint8
   # RGBA format with alpha channel
   alpha_mask = vehicle_image[:, :, 3]  # Extract alpha
   ```

2. **Depth Estimation**:
   ```python
   depth_map = numpy.ndarray  # Shape: [H, W], dtype: float32
   # Values: 0.0 (far) to 1.0 (near)
   ```

3. **Edge Detection**:
   ```python
   edge_map = numpy.ndarray  # Shape: [H, W], dtype: uint8
   # Values: 0 (no edge) or 255 (edge)
   ```

4. **SDXL Generation**:
   ```python
   # Input to SDXL:
   prompt = "minimalist vector illustration, clean lines, flat colors"
   control_images = {
       "depth": depth_map,      # ControlNet depth input
       "canny": edge_map        # ControlNet canny input
   }
   controlnet_weights = {
       "depth_weight": 0.6,
       "canny_weight": 0.4
   }
   
   # Output:
   vector_raster = numpy.ndarray  # Shape: [H, W, 3], dtype: uint8
   ```

5. **IoU Validation & Retry**:
   ```python
   # Extract masks for comparison
   original_mask = extract_alpha_mask(raw_img)
   generated_mask = extract_alpha_mask(vector_raster)
   
   # Validate geometric similarity
   iou = calculate_iou(original_mask, generated_mask)
   if iou < 0.85:
       # Retry with increased weights
       depth_weight = min(depth_weight + 0.1, 0.9)
       canny_weight = min(canny_weight + 0.1, 0.7)
       # Regenerate...
   ```

**Output Data**:
- **Type**: `numpy.ndarray`
- **Shape**: `[H, W, 3]` (RGB, uint8)
- **Metadata**: Processing time, IoU score, retry count, ControlNet weights, edge_map (for Phase IV)

**Code Reference**: `src/phase2_generative_steering/generator.py:39-152`

---

### Phase III: Chromatic Enforcement

**Entry**: `src/pipeline/orchestrator.py:309` → `phase3.enforce()`

**Input Data**:
- **Type**: `numpy.ndarray` (vector raster from Phase II)
- **Shape**: `[H, W, 3]` (RGB, uint8)

**Processing Flow**:
```
Vector Raster Image
    ↓
[Upscaler (RealESRGAN 4x)]
    ↓
High-Resolution Image (4H x 4W x 3)
    ↓
[Color Quantizer (CIEDE2000)]
    ↓
Quantized Image (15 colors only)
    ↓
[Noise Remover]
    ↓
Clean Quantized Image
    ↓
[Palette Audit]
    ↓
Validated Quantized Image
```

**Data Structures at Each Stage**:

1. **Upscaling**:
   ```python
   # Input: [H, W, 3]
   upscaled = numpy.ndarray  # Shape: [4*H, 4*W, 3], dtype: uint8
   # 4x resolution increase
   ```

2. **Color Quantization**:
   ```python
   # Palette: List[15 hex colors]
   palette = ["#FF0000", "#00FF00", ..., "#000000"]  # 15 colors
   
   # Convert to RGB
   palette_rgb = numpy.ndarray  # Shape: [15, 3], dtype: uint8
   
   # Quantize each pixel
   quantized = numpy.ndarray  # Shape: [4*H, 4*W, 3], dtype: uint8
   # Each pixel value matches one of 15 palette colors
   
   # Process:
   for each pixel in upscaled:
       distances = [ciede2000_distance(pixel, palette_color) 
                   for palette_color in palette_rgb]
       nearest_idx = argmin(distances)
       quantized[pixel] = palette_rgb[nearest_idx]
   ```

3. **Noise Removal**:
   ```python
   # Connected component analysis
   # Remove components < 0.1% of image area
   cleaned = remove_small_components(quantized, min_area=0.001)
   ```

4. **Palette Audit**:
   ```python
   # Verify all colors in output match palette
   unique_colors = get_unique_colors(quantized)
   invalid_colors = [c for c in unique_colors if c not in palette]
   # Should be empty list
   ```

**Output Data**:
- **Type**: `numpy.ndarray`
- **Shape**: `[4*H, 4*W, 3]` (upscaled dimensions)
- **Values**: Only 15 colors from palette
- **Metadata**: Processing time, quantization method, palette compliance

**Code Reference**: `src/phase3_chromatic_enforcement/enforcer.py:36-93`

---

### Phase IV: Vector Reconstruction

**Entry**: `src/pipeline/orchestrator.py:340` → `phase4.vectorize()`

**Input Data**:
- **Type**: `numpy.ndarray` (quantized image from Phase III)
- **Shape**: `[H, W, 3]` (RGB, uint8, 15 colors only)
- **Optional**: `edge_map` from Phase II (for Strategy B)

**Processing Flow**:
```
Quantized Image
    ↓
[VTracer Conversion]
    ↓
Raw SVG XML (string)
    ↓
[SVG Processor]
    ↓
Optimized SVG XML
    ↓
[Centerline Tracer] (Optional, Strategy B)
    ↓
Final SVG XML (string)
    ↓
[File Output] (if output_path provided)
```

**Data Structures at Each Stage**:

1. **VTracer Input**:
   ```python
   # Save quantized image to temporary PNG
   temp_png_path = "/tmp/quantized_XXXXXX.png"
   save_image(quantized_image, temp_png_path)
   
   # VTracer command:
   # vtracer --input temp_png_path --output temp_svg_path --mode stacked
   ```

2. **VTracer Output**:
   ```python
   svg_xml_raw = """
   <svg xmlns="http://www.w3.org/2000/svg" width="4096" height="3072">
     <path d="M100,100 L200,200 ..." fill="#FF0000"/>
     <path d="M300,300 L400,400 ..." fill="#00FF00"/>
     ...
   </svg>
   """
   # Raw SVG with paths but no strokes
   ```

3. **SVG Processing**:
   ```python
   # Inject stroke attributes
   svg_xml_processed = """
   <svg xmlns="http://www.w3.org/2000/svg" width="4096" height="3072">
     <path d="M100,100 L200,200 ..." 
           fill="#FF0000" 
           stroke="black" 
           stroke-width="2"/>
     ...
   </svg>
   """
   ```

4. **Centerline Tracing** (Strategy B, optional):
   ```python
   # Use edge_map from Phase II
   centerlines = extract_centerlines(edge_map)
   # Convert to SVG paths with strokes
   ```

**Output Data**:
- **Type**: `str` (SVG XML)
- **Format**: Valid SVG XML string
- **Content**: Path elements with fill colors and strokes
- **Metadata**: Processing time, strategy used, path count

**Code Reference**: `src/phase4_vector_reconstruction/vectorizer.py:45-135`

---

## Complete Pipeline Data Flow

### Synchronous Processing Flow

```
1. Input Image File (JPEG/PNG/WebP)
   ↓
2. Image Loading & Validation
   Type: numpy.ndarray [H, W, 3], dtype: uint8
   ↓
3. Phase I: Semantic Sanitization
   Input: [H, W, 3] uint8
   Output: [H, W, 3] uint8 (clean plate)
   ↓
4. Phase II: Generative Steering
   Input: [H, W, 3] uint8 (clean plate)
   Output: [H, W, 3] uint8 (vector raster)
   Metadata: edge_map [H, W] uint8 (for Phase IV)
   ↓
5. Phase III: Chromatic Enforcement
   Input: [H, W, 3] uint8 (vector raster)
   Output: [4*H, 4*W, 3] uint8 (quantized, 15 colors)
   ↓
6. Phase IV: Vector Reconstruction
   Input: [4*H, 4*W, 3] uint8 (quantized)
   Output: str (SVG XML)
   ↓
7. PNG Preview Generation
   Input: [4*H, 4*W, 3] uint8 (quantized)
   Output: Resized PNG file (2048px max dimension)
   ↓
8. File Outputs
   - SVG file: output.svg
   - PNG preview: preview.png (optional)
```

### Data Type Transformations

| Stage | Data Type | Shape | Dtype | Example |
|-------|-----------|-------|-------|---------|
| Input File | File | - | - | `input.jpg` |
| Loaded Image | numpy.ndarray | [1024, 2048, 3] | uint8 | Raw pixels |
| Phase I Output | numpy.ndarray | [1024, 2048, 3] | uint8 | Clean plate |
| Phase II Output | numpy.ndarray | [1024, 2048, 3] | uint8 | Vector raster |
| Phase III Output | numpy.ndarray | [4096, 8192, 3] | uint8 | Quantized |
| Phase IV Output | str | - | - | SVG XML |
| PNG Preview | File | - | - | `preview.png` |

---

## Background Job Processing Flow

### Job Creation & Processing

```
1. HTTP POST /api/v1/jobs
   ↓
2. File Validation
   ↓
3. Temp File Creation
   temp_path = "/tmp/tmpXXXXXX.png"
   ↓
4. Job Creation
   job_id = uuid.uuid4()
   job = Job(status=PENDING, metadata={input_path: temp_path})
   JobQueue.jobs[job_id] = job
   ↓
5. Background Task Scheduling
   background_tasks.add_task(process_job_background, job_id, ...)
   ↓
6. Immediate Response
   {"job_id": "...", "status": "pending"}
   ↓
7. [Background Processing Starts]
   ↓
8. Job Status Update: PROCESSING
   job.status = PROCESSING
   job.started_at = datetime.utcnow()
   ↓
9. Pipeline Execution
   svg_xml, metadata = pipeline.process_image(...)
   ↓
10. Output File Generation
    svg_path = OUTPUT_DIR / f"{job_id[:8]}.svg"
    png_path = OUTPUT_DIR / f"{job_id[:8]}.png"
    ↓
11. Job Completion
    job.status = COMPLETED
    job.result = {
        "svg_url": "/api/v1/download/abc12345.svg",
        "png_preview_url": "/api/v1/download/abc12345.png",
        ...
    }
    job.completed_at = datetime.utcnow()
    ↓
12. Temp File Cleanup
    os.unlink(temp_path)
```

### Status Polling Flow

```
Client: GET /api/v1/jobs/{job_id}
    ↓
Server: Retrieve job from JobQueue
    ↓
Response Based on Status:
    
    PENDING:
    {
        "job_id": "...",
        "status": "pending",
        "created_at": "..."
    }
    
    PROCESSING:
    {
        "job_id": "...",
        "status": "processing",
        "created_at": "...",
        "started_at": "..."
    }
    
    COMPLETED:
    {
        "job_id": "...",
        "status": "completed",
        "result_url": "/api/v1/download/abc12345.svg",
        "png_preview_url": "/api/v1/download/abc12345.png",
        "processing_time_ms": 1234,
        ...
    }
    
    FAILED:
    {
        "job_id": "...",
        "status": "failed",
        "error": "Error message..."
    }
```

---

## Training Job Flow (LoRA Training)

### Training Job Creation

```
1. POST /ui/training
   Form data: input_files[], target_files[], hyperparameters
   ↓
2. File Pair Validation
   - Check input_files.length == target_files.length
   - Minimum 10 pairs required
   ↓
3. Training Job Creation
   job_id = TrainingJobRegistry.create_job(params)
   ↓
4. File Storage
   TRAIN_DATA_ROOT / job_id / inputs / {i:04d}_{filename}
   TRAIN_DATA_ROOT / job_id / targets / {i:04d}_{filename}
   ↓
5. Background Training Start
   background_tasks.add_task(run_training_background, ...)
   ↓
6. Redirect to Job Status Page
   /ui/training/jobs/{job_id}
```

### Training Execution Flow

```
1. Job Status: PENDING → PROCESSING
   ↓
2. Dataset Preparation
   - Load input/target pairs
   - Create train/validation split
   - Apply data augmentation
   ↓
3. Model Initialization
   - Load SDXL base model
   - Initialize LoRA adapters (rank, alpha)
   ↓
4. Training Loop
   For each epoch:
       For each batch:
           - Forward pass
           - Loss calculation
           - Backward pass
           - Optimizer step
           - Update progress: epoch, step, train_loss
       - Validation
       - Update progress: val_loss
   ↓
5. Checkpoint Saving
   weights_path = TRAIN_OUTPUT_ROOT / job_id / "vector_style_lora.safetensors"
   ↓
6. Job Completion
   job.status = COMPLETED
   job.artifacts = {"weights": weights_path}
   ↓
7. Status Updates (via HTMX polling)
   - Real-time log streaming
   - Progress percentage
   - Loss values
   - Epoch/step counters
```

**Data Structures**:
- **Training Job**: `TrainingJob` object with progress tracking
- **Logs**: List of log strings (last 1000 lines)
- **Progress**: `progress` (0.0-100.0), `current_epoch`, `train_loss`, `val_loss`

---

## Error Handling & Retry Flows

### Phase II IoU Retry Flow

```
1. Phase II Generation
   vector_raster, metadata = phase2.generate(...)
   ↓
2. IoU Validation
   iou = validate_geometric_similarity(original_mask, generated_mask)
   ↓
3. Check Threshold
   if iou < 0.85:
       ↓
4. Retry Logic (max 2 retries)
   retry_count += 1
   depth_weight = min(depth_weight + 0.1, 0.9)
   canny_weight = min(canny_weight + 0.1, 0.7)
   ↓
5. Regenerate with New Weights
   vector_raster = phase2.generate(..., controlnet_weights_override={...})
   ↓
6. Re-validate IoU
   if iou >= 0.85 or retry_count >= max_retries:
       break (accept result)
   else:
       goto step 4 (retry again)
```

### Error Propagation Flow

```
Pipeline Error Occurs
    ↓
Exception Caught in process_image()
    ↓
Error Logged (with correlation_id)
    ↓
PipelineError Raised
    ↓
FastAPI Error Handler (error_responses.py)
    ↓
RFC 7807 Error Response Created
    ↓
HTTP Response (500/400/503)
    ↓
Client Receives Error JSON
```

**Error Response Format**:
```json
{
    "type": "https://api.gemini3pro.example.com/errors/pipeline-error",
    "title": "Pipeline Error",
    "status": 500,
    "detail": "Error message...",
    "instance": "/api/v1/process",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Configuration & Model Loading Flow

### Configuration Loading

```
1. Application Startup
   ↓
2. Config Initialization (get_config())
   ↓
3. YAML File Loading
   configs/default_config.yaml → dict
   ↓
4. Environment Variable Overrides
   os.getenv("DEVICE") → config["hardware"]["device"]
   os.getenv("PRECISION") → config["hardware"]["precision"]
   ↓
5. Config Object Creation
   Config._config = merged_config
   ↓
6. Singleton Storage
   _config_instance = Config()
   ↓
7. Access via get_config()
   Returns singleton instance
```

### Model Loading (Lazy)

```
1. Phase Component Initialization
   phase1 = Phase1Sanitizer(config)
   # Models NOT loaded yet
   ↓
2. First Use Trigger
   phase1.sanitize(image) called
   ↓
3. Lazy Model Loading
   if self.detector is None:
       self.detector = GroundingDINODetector(...)
       # Model downloaded/loaded here
   ↓
4. Model Caching
   Model stored in memory for reuse
   ↓
5. Subsequent Uses
   Reuse cached model (no reload)
```

### Model Volume Flow (Serverless)

```
1. ModelLoader Initialization
   loader = ModelLoader(volume_path="/models")
   ↓
2. Check Volume Availability
   if Path("/models").exists():
       volume_available = True
   ↓
3. Model Path Resolution
   model_path = volume_path / model_id
   ↓
4. Check if Model Exists on Volume
   if model_path.exists():
       # Use volume model
   else:
       # Download from HuggingFace
   ↓
5. Model Loading
   model = model_loader(model_id, local_dir=volume_path)
```

---

## Metrics & Logging Flow

### Correlation ID Flow

```
1. Request Received
   ↓
2. Correlation ID Generation
   correlation_id = str(uuid.uuid4())
   ↓
3. Correlation ID Set
   set_correlation_id(correlation_id)
   logger.set_correlation_id(correlation_id)
   ↓
4. All Logs Include Correlation ID
   {"correlation_id": "abc-123", "event": "phase1_start", ...}
   ↓
5. Metrics Collection
   metrics_collector.set_correlation_id(correlation_id)
   ↓
6. All Metrics Tagged with Correlation ID
   ↓
7. Response Includes Correlation ID
   {"correlation_id": "abc-123", ...}
```

### Metrics Collection Flow

```
1. MetricsCollector Initialization
   collector = MetricsCollector()
   ↓
2. Per-Phase Timing
   phase_start = datetime.now()
   # ... phase processing ...
   phase_time = (datetime.now() - phase_start).total_seconds() * 1000
   collector.record_phase_time("phase1", phase_time)
   ↓
3. GPU Memory Tracking
   if torch.cuda.is_available():
       memory_used = torch.cuda.memory_allocated()
       collector.record_gpu_memory(memory_used)
   ↓
4. Pipeline Summary
   summary = collector.get_pipeline_summary()
   {
       "total_time_ms": 1234,
       "phase_timings": {...},
       "gpu_memory_peak_mb": 8192,
       ...
   }
```

---

## File Management Flow

### Temp File Lifecycle

```
1. Temp File Creation
   with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
       img.save(tmp.name, 'PNG')
       input_path = tmp.name
   # File: /tmp/tmpXXXXXX.png
   ↓
2. Processing Uses Temp File
   pipeline.process_image(input_image_path=input_path, ...)
   ↓
3. Cleanup in Finally Block
   finally:
       if os.path.exists(input_path):
           os.unlink(input_path)
   # Temp file deleted
```

### Output File Management

```
1. Output Directory Setup
   OUTPUT_DIR = Path("/tmp/gemini3_output")
   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
   ↓
2. Output ID Generation
   output_id = correlation_id[:8]  # First 8 chars of UUID
   ↓
3. Output File Paths
   svg_path = OUTPUT_DIR / f"{output_id}.svg"
   png_path = OUTPUT_DIR / f"{output_id}.png"
   ↓
4. File Writing
   with open(svg_path, 'w') as f:
       f.write(svg_xml)
   save_image(quantized_image, png_path)
   ↓
5. Download Endpoint
   GET /api/v1/download/{filename}
   - Validates filename (sanitize_filename)
   - Checks path traversal (validate_path_within_directory)
   - Returns FileResponse
```

---

## Related Documentation

- **[APPLICATION_OVERVIEW.md](APPLICATION_OVERVIEW.md)** - What the application does
- **[FILE_STRUCTURE.md](FILE_STRUCTURE.md)** - File structure and component functions
- **[README.md](README.md)** - Quick start guide

---

**Last Updated**: Generated from codebase analysis  
**Version**: 3.0.0

