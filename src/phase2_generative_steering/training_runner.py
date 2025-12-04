"""Training runner wrapper for LoRA training with progress callbacks"""

import os
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

from src.phase2_generative_steering.train_lora_sdxl import train_lora, LoRADataset
from src.api.training_jobs import get_training_registry, TrainingJobStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingCallbacks:
    """Callbacks for training progress updates"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.registry = get_training_registry()
    
    def on_step(self, step: int, total_steps: int, loss: float):
        """Called on each training step"""
        progress = (step / total_steps) * 100.0 if total_steps > 0 else 0.0
        self.registry.update_job_progress(
            self.job_id,
            progress=progress,
            step=step,
            total_steps=total_steps,
            train_loss=loss
        )
        self.registry.add_job_log(
            self.job_id,
            f"Step {step}/{total_steps}: loss={loss:.4f}",
            level="info"
        )
    
    def on_epoch_start(self, epoch: int, total_epochs: int):
        """Called at the start of each epoch"""
        self.registry.update_job_progress(
            self.job_id,
            epoch=epoch,
            total_epochs=total_epochs
        )
        self.registry.add_job_log(
            self.job_id,
            f"Starting epoch {epoch}/{total_epochs}",
            level="info"
        )
    
    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Called at the end of each epoch"""
        self.registry.update_job_progress(
            self.job_id,
            train_loss=train_loss,
            val_loss=val_loss
        )
        self.registry.add_job_log(
            self.job_id,
            f"Epoch {epoch} complete: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}",
            level="info"
        )
    
    def on_complete(self, output_path: str):
        """Called when training completes"""
        self.registry.set_job_artifacts(self.job_id, {"weights": output_path})
        self.registry.update_job_progress(self.job_id, progress=100.0)
        self.registry.add_job_log(
            self.job_id,
            f"Training completed! Weights saved to: {output_path}",
            level="success"
        )
    
    def on_error(self, error: str):
        """Called when training fails"""
        self.registry.add_job_log(self.job_id, f"Error: {error}", level="error")


def run_lora_training(
    job_id: str,
    dataset_dir: str,
    params: Dict[str, Any],
    callbacks: Optional[TrainingCallbacks] = None
) -> str:
    """
    Run LoRA training with progress callbacks.
    
    Args:
        job_id: Training job ID
        dataset_dir: Directory containing input/ and target/ subdirectories
        params: Training parameters (learning_rate, batch_size, num_epochs, etc.)
        callbacks: Optional callbacks for progress updates
    
    Returns:
        Path to trained weights file
    """
    registry = get_training_registry()
    
    try:
        registry.update_job_status(job_id, TrainingJobStatus.PROCESSING)
        registry.add_job_log(job_id, "Starting LoRA training...", level="info")
        
        # Prepare dataset directories
        input_dir = Path(dataset_dir) / "inputs"
        target_dir = Path(dataset_dir) / "targets"
        
        if not input_dir.exists() or not target_dir.exists():
            raise ValueError(f"Dataset directories not found: {input_dir} or {target_dir}")
        
        # Count image pairs
        image_pairs = []
        for input_file in input_dir.glob("*.png"):
            target_file = target_dir / input_file.name
            if target_file.exists():
                image_pairs.append((str(input_file), str(target_file)))
        
        if len(image_pairs) < 10:
            raise ValueError(f"Insufficient image pairs: {len(image_pairs)}. Need at least 10.")
        
        registry.add_job_log(job_id, f"Found {len(image_pairs)} image pairs", level="info")
        
        # Prepare output directory
        output_dir = Path(os.getenv("TRAIN_OUTPUT_ROOT", "/tmp/gemini3_training")) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = str(output_dir / params.get("output_path", "vector_style_lora.safetensors"))
        
        # Extract training parameters
        base_model = params.get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        rank = params.get("rank", 32)
        alpha = params.get("alpha", 16)
        learning_rate = params.get("learning_rate", 1e-4)
        batch_size = params.get("batch_size", 1)
        num_epochs = params.get("num_epochs", 10)
        validation_split = params.get("validation_split", 0.2)
        save_steps = params.get("save_steps", 100)
        
        registry.add_job_log(
            job_id,
            f"Training parameters: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}, rank={rank}, alpha={alpha}",
            level="info"
        )
        
        # Run training with callbacks
        train_lora(
            input_dir=str(input_dir),
            output_dir=str(target_dir),
            base_model=base_model,
            rank=rank,
            alpha=alpha,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            validation_split=validation_split,
            save_steps=save_steps,
            output_path=output_path,
            callbacks=callbacks
        )
        
        # Training completed successfully
        if callbacks:
            callbacks.on_complete(output_path)
        
        registry.update_job_status(job_id, TrainingJobStatus.COMPLETED)
        registry.add_job_log(job_id, "Training completed successfully!", level="success")
        
        return output_path
        
    except Exception as e:
        error_msg = str(e)
        logger.error("training_failed", job_id=job_id, error=error_msg, exc_info=True)
        
        registry.add_job_log(job_id, f"Training failed: {error_msg}", level="error")
        registry.update_job_status(job_id, TrainingJobStatus.FAILED, error=error_msg)
        
        if callbacks:
            callbacks.on_error(error_msg)
        
        raise


def run_training_background(
    job_id: str,
    dataset_dir: str,
    params: Dict[str, Any]
):
    """
    Run training in a background thread (non-blocking).
    
    This function is designed to be called from FastAPI BackgroundTasks.
    """
    callbacks = TrainingCallbacks(job_id)
    
    # Run in thread pool to avoid blocking event loop
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run_lora_training, job_id, dataset_dir, params, callbacks)
    
    try:
        future.result()  # Wait for completion
    except Exception as e:
        logger.error("background_training_failed", job_id=job_id, error=str(e), exc_info=True)
    finally:
        executor.shutdown(wait=False)

