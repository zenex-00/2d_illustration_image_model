"""Background training task runner"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import gc
import torch

from src.utils.logger import get_logger
from src.utils.image_utils import load_image, save_image
from src.api.job_queue import get_job_queue, JobStatus
from src.pipeline.config import get_config
from src.phase1_semantic_sanitization.sanitizer import Phase1Sanitizer
from src.phase2_generative_steering.background_remover import BackgroundRemover
from src.phase2_generative_steering.depth_estimator import DepthEstimator
from src.phase2_generative_steering.edge_detector import EdgeDetector
from .train_lora_sdxl import train_lora

logger = get_logger(__name__)


def run_training_background(
    job_id: str,
    input_files: list,
    target_files: list,
    training_params: Dict[str, Any]
) -> None:
    """
    Run training in background and update job status
    
    Args:
        job_id: Job ID
        input_files: List of uploaded input file objects
        target_files: List of uploaded target file objects
        training_params: Training parameters (learning_rate, batch_size, etc.)
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)
    
    if not job:
        logger.error("training_job_not_found", job_id=job_id)
        return
    
    try:
        # Get training directories from environment
        train_data_root = os.getenv("TRAIN_DATA_ROOT", "/workspace/training_data")
        train_output_root = os.getenv("TRAIN_OUTPUT_ROOT", "/workspace/training_output")
        
        # Create job-specific directories
        job_input_dir = Path(train_data_root) / job_id / "inputs"
        job_processed_input_dir = Path(train_data_root) / job_id / "processed_inputs"
        job_control_images_dir = Path(train_data_root) / job_id / "control_images"
        job_target_dir = Path(train_data_root) / job_id / "targets"
        job_output_dir = Path(train_output_root) / job_id
        
        job_input_dir.mkdir(parents=True, exist_ok=True)
        job_processed_input_dir.mkdir(parents=True, exist_ok=True)
        job_control_images_dir.mkdir(parents=True, exist_ok=True)
        job_target_dir.mkdir(parents=True, exist_ok=True)
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("training_setup", job_id=job_id, input_dir=str(job_input_dir))
        
        # Update status to processing
        queue.update_job(job_id, status="processing", progress=5)
        
        # Save uploaded files
        logger.info("saving_training_files", count=len(input_files))
        for idx, (input_file, target_file) in enumerate(zip(input_files, target_files)):
            # Handle both file objects and file paths
            if isinstance(input_file, str):
                # It's a file path
                input_source = input_file
                input_filename = Path(input_file).name
            else:
                # It's a file object
                input_source = input_file.file
                input_filename = input_file.filename
            
            if isinstance(target_file, str):
                # It's a file path
                target_source = target_file
                target_filename = Path(target_file).name
            else:
                # It's a file object
                target_source = target_file.file
                target_filename = target_file.filename
            
            # Save input file
            saved_input_filename = f"{idx:04d}_{input_filename}"
            input_path = job_input_dir / saved_input_filename
            if isinstance(input_file, str):
                # Copy from path
                shutil.copy2(input_source, input_path)
            else:
                # Copy from file object
                with open(input_path, "wb") as f:
                    shutil.copyfileobj(input_source, f)
            
            # Save target file
            saved_target_filename = f"{idx:04d}_{target_filename}"
            target_path = job_target_dir / saved_target_filename
            if isinstance(target_file, str):
                # Copy from path
                shutil.copy2(target_source, target_path)
            else:
                # Copy from file object
                with open(target_path, "wb") as f:
                    shutil.copyfileobj(target_source, f)
        
        logger.info("files_saved", input_count=len(input_files), target_count=len(target_files))
        queue.update_job(job_id, status="processing", progress=10)
        
        # Store intermediate images directory in metadata
        job.metadata['intermediate_images_dir'] = str(Path(train_data_root) / job_id)
        
        # Add log entry
        job.logs.append(f"[{datetime.utcnow()}] Found {len(input_files)} image pairs")
        job.logs.append(f"[{datetime.utcnow()}] Phase 1: Starting semantic sanitization ({len(input_files)} images)...")
        
        # Update phase status
        queue.update_job(job_id, metadata=job.metadata)
        
        # Step 2: Process raw inputs through Phase 1 (Sanitization)
        logger.info("phase1_processing_start", job_id=job_id)
        config = get_config()
        phase1 = Phase1Sanitizer(config)
        
        phase1_processed_paths = []
        phase1_success_count = 0
        
        # Get list of input files
        input_file_list = sorted([f for f in job_input_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        total_inputs = len(input_file_list)
        
        if hasattr(queue, 'update_phase_status'):
            queue.update_phase_status(job_id, "phase1", "processing", f"0/{total_inputs}")
        
        for idx, raw_input_path in enumerate(input_file_list):
                try:
                    # Update phase progress
                    job.logs.append(f"[{datetime.utcnow()}] Phase 1: Processing image {idx+1}/{total_inputs} - {raw_input_path.name}")
                    
                    # Load raw image
                    raw_img = load_image(str(raw_input_path))
                    
                    # Process through Phase 1
                    clean_plate, phase1_metadata = phase1.sanitize(
                        raw_img,
                        correlation_id=f"{job_id}_{idx}"
                    )
                    
                    # Save Phase 1 output
                    phase1_output_path = job_processed_input_dir / raw_input_path.name
                    save_image(clean_plate, str(phase1_output_path))
                    phase1_processed_paths.append(phase1_output_path)
                    phase1_success_count += 1
                    
                    detections = phase1_metadata.get("detections", 0)
                    job.logs.append(f"[{datetime.utcnow()}] Phase 1: Image {idx+1} - Removed {detections} prohibited elements")
                    
                    logger.info("phase1_processed", idx=idx, detections=detections)
                    
                    # Update phase progress
                    if hasattr(queue, 'update_phase_status'):
                        queue.update_phase_status(job_id, "phase1", "processing", f"{idx+1}/{total_inputs}")
                    
                except Exception as e:
                    logger.error("phase1_failed_for_image", idx=idx, error=str(e), exc_info=True)
                    # Check if this is a PhaseError (meaning Phase 1 failed completely after retries)
                    from src.utils.error_handler import PhaseError
                    # Use config that was already loaded earlier in the function
                    # Get config again to avoid potential scoping issues
                    try:
                        config = get_config()
                        training_config = config.get("training", {})
                        phase1_retry_config = training_config.get("phase1_retry", {})
                        skip_on_failure = phase1_retry_config.get("skip_on_failure", False)
                    except:
                        # Fallback to default value if config access fails
                        skip_on_failure = False

                    if isinstance(e, PhaseError) and e.phase == "phase1":
                        if skip_on_failure:
                            # Skip this image if configured to do so
                            job.logs.append(f"[{datetime.utcnow()}] Phase 1: Image {idx+1} - Failed after retries, skipping image")
                            continue  # Skip this image
                        else:
                            # Use original image as fallback (current approach)
                            # This maintains backward compatibility while still logging the failure
                            phase1_output_path = job_processed_input_dir / raw_input_path.name
                            shutil.copy2(raw_input_path, phase1_output_path)
                            phase1_processed_paths.append(phase1_output_path)

                            job.logs.append(f"[{datetime.utcnow()}] Phase 1: Image {idx+1} - Failed after retries, using original image")
                    else:
                        # For other types of errors, check the skip_on_failure setting
                        if skip_on_failure:
                            job.logs.append(f"[{datetime.utcnow()}] Phase 1: Image {idx+1} - Failed, skipping image")
                            continue  # Skip this image
                        else:
                            # Use original image
                            phase1_output_path = job_processed_input_dir / raw_input_path.name
                            shutil.copy2(raw_input_path, phase1_output_path)
                            phase1_processed_paths.append(phase1_output_path)

                            job.logs.append(f"[{datetime.utcnow()}] Phase 1: Image {idx+1} - Failed, using original image")
        
        logger.info("phase1_processing_complete", processed_count=len(phase1_processed_paths))
        
        # Cleanup Phase 1 models
        del phase1
        from src.pipeline.model_cache import get_model_cache
        cache = get_model_cache()
        cache.clear_cache(phase_prefix="phase1_")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("phase1_memory_flushed")
        
        queue.update_job(job_id, status="processing", progress=30)
        
        if hasattr(job, 'logs'):
            job.logs.append(f"[{datetime.utcnow()}] Phase 1: Complete - {phase1_success_count}/{total_inputs} images processed")
        
        if hasattr(queue, 'update_phase_status'):
            queue.update_phase_status(job_id, "phase1", "completed", f"{len(phase1_processed_paths)}/{total_inputs}")
            job.metadata['phase1_images_processed'] = len(phase1_processed_paths)
        
        # Step 3: Process Phase 1 outputs through Phase 2 preprocessing
        logger.info("phase2_preprocessing_start", job_id=job_id)
        
        phase2_config = config.get_phase_config("phase2")
        hardware_config = config.get_hardware_config()
        device = hardware_config.get("device", "cuda")
        
        bg_remover = BackgroundRemover(
            model_name=phase2_config["background_removal"]["model"],
            device=device
        )
        depth_estimator = DepthEstimator(
            model_type=phase2_config["depth_estimation"]["model"],
            device=device
        )
        edge_detector = EdgeDetector()
        
        job.logs.append(f"[{datetime.utcnow()}] Phase 2: Starting preprocessing (background removal, depth, edge detection)...")
        
        if hasattr(queue, 'update_phase_status'):
            queue.update_phase_status(job_id, "phase2", "processing", f"0/{len(phase1_processed_paths)}")
        
        phase2_success_count = 0
        
        for idx, phase1_path in enumerate(sorted(phase1_processed_paths)):
            try:
                job.logs.append(f"[{datetime.utcnow()}] Phase 2: Image {idx+1}/{len(phase1_processed_paths)} - Removing background...")
                
                # Load Phase 1 output
                clean_plate = load_image(str(phase1_path))
                
                # Step 1: Remove background
                bg_removed = bg_remover.remove_background(clean_plate)
                
                # Extract RGB from RGBA if needed
                if bg_removed.shape[2] == 4:
                    rgb_image = bg_removed[:, :, :3]
                else:
                    rgb_image = bg_removed
                
                job.logs.append(f"[{datetime.utcnow()}] Phase 2: Image {idx+1}/{len(phase1_processed_paths)} - Generating depth map...")
                
                # Step 2: Generate depth map
                depth_map = depth_estimator.estimate_depth(rgb_image)
                
                job.logs.append(f"[{datetime.utcnow()}] Phase 2: Image {idx+1}/{len(phase1_processed_paths)} - Detecting edges...")
                
                # Step 3: Detect edges
                edge_map = edge_detector.detect_edges(rgb_image)
                
                # Save control images
                depth_path = job_control_images_dir / f"depth_{idx:04d}.png"
                edge_path = job_control_images_dir / f"edge_{idx:04d}.png"
                
                save_image(depth_map, str(depth_path))
                save_image(edge_map, str(edge_path))
                
                phase2_success_count += 1
                
                job.logs.append(f"[{datetime.utcnow()}] Phase 2: Image {idx+1}/{len(phase1_processed_paths)} - Control images saved")
                
                logger.info("phase2_preprocessed", idx=idx)
                
                # Update phase progress
                if hasattr(queue, 'update_phase_status'):
                    queue.update_phase_status(job_id, "phase2", "processing", f"{idx+1}/{len(phase1_processed_paths)}")
                
            except Exception as e:
                logger.error("phase2_preprocessing_failed_for_image", idx=idx, error=str(e), exc_info=True)
                # Fallback: create placeholder control images (zeros)
                import numpy as np
                depth_path = job_control_images_dir / f"depth_{idx:04d}.png"
                edge_path = job_control_images_dir / f"edge_{idx:04d}.png"
                
                # Create placeholder images (1024x1024)
                placeholder = np.zeros((1024, 1024, 3), dtype=np.uint8)
                save_image(placeholder, str(depth_path))
                save_image(placeholder, str(edge_path))
                
                job.logs.append(f"[{datetime.utcnow()}] Phase 2: Image {idx+1} - Failed, using placeholder control images")
        
        # Cleanup Phase 2 preprocessing models
        del bg_remover
        del depth_estimator
        del edge_detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("phase2_preprocessing_complete", processed_count=phase2_success_count)
        queue.update_job(job_id, status="processing", progress=50)
        
        job.logs.append(f"[{datetime.utcnow()}] Phase 2: Preprocessing complete - {phase2_success_count} control image pairs generated")
        
        if hasattr(queue, 'update_phase_status'):
            queue.update_phase_status(job_id, "phase2", "completed", f"{phase2_success_count}/{len(phase1_processed_paths)}")
            job.metadata['phase2_images_processed'] = phase2_success_count
        
        job.logs.append(f"[{datetime.utcnow()}] Training: Starting LoRA training with ControlNet conditioning...")
        
        # Progress callback
        def progress_callback(progress_data: Dict[str, Any]):
            """Update job progress during training"""
            epoch = progress_data.get("epoch", 0)
            step = progress_data.get("step", 0)
            loss = progress_data.get("loss", 0.0)
            progress_pct = progress_data.get("progress", 0.0)
            
            # Map training progress to 50-95% (since we're at 50% after preprocessing)
            training_progress = 50 + (progress_pct * 0.45)  # 50% to 95%
            
            # Update job
            job.current_epoch = epoch
            job.train_loss = loss
            job.progress = min(training_progress, 95.0)  # Leave 5% for final save
            
            # Add log
            log_msg = f"[{datetime.utcnow()}] Epoch {epoch}, Step {step}: loss={loss:.4f}"
            job.logs.append(log_msg)
            # Keep only last 1000 logs
            if len(job.logs) > 1000:
                job.logs = job.logs[-1000:]
        
        # Run training
        logger.info("training_start", job_id=job_id)
        
        # Get ControlNet weights from config
        controlnet_depth_weight = phase2_config["controlnet"]["depth_weight"]
        controlnet_canny_weight = phase2_config["controlnet"]["canny_weight"]
        
        weights_path = train_lora(
            input_dir=str(job_processed_input_dir),  # Use processed inputs from Phase 1
            target_dir=str(job_target_dir),
            output_dir=str(job_output_dir),
            control_images_dir=str(job_control_images_dir),
            controlnet_depth_weight=controlnet_depth_weight,
            controlnet_canny_weight=controlnet_canny_weight,
            num_epochs=training_params.get("num_epochs", 10),
            batch_size=training_params.get("batch_size", 1),
            learning_rate=training_params.get("learning_rate", 1e-4),
            rank=training_params.get("rank", 32),
            alpha=training_params.get("alpha", 16),
            validation_split=training_params.get("validation_split", 0.2),
            seed=training_params.get("seed", 42),
            progress_callback=progress_callback,
        )
        
        # Update job with results
        job.progress = 100.0
        job.total_epochs = training_params.get("num_epochs", 10)
        job.current_epoch = training_params.get("num_epochs", 10)
        
        job.artifacts["weights"] = weights_path
        
        job.logs.append(f"[{datetime.utcnow()}] Training completed!")
        job.logs.append(f"[{datetime.utcnow()}] Weights saved to: {weights_path}")
        
        queue.update_job(job_id, status="completed", progress=100)
        logger.info("training_complete", job_id=job_id, weights_path=weights_path)
        
    except Exception as e:
        logger.error("training_failed", job_id=job_id, error=str(e), exc_info=True)
        queue.update_job(job_id, status="failed", error=str(e))
        
        job.logs.append(f"[{datetime.utcnow()}] ERROR: {str(e)}")
