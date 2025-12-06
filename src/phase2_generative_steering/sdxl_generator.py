"""SDXL generator with ControlNet guidance and LoRA support"""

import os
import numpy as np
import torch
from typing import Optional, List, Tuple

# Disable xformers if it's causing import issues (set before importing diffusers)
# This prevents RuntimeError when xformers is installed but incompatible with PyTorch/CUDA
# Check if already disabled (set by server.py or environment)
if os.getenv("DISABLE_XFORMERS") != "1":
    try:
        import xformers
        try:
            from xformers.ops import fmha  # noqa: F401
            os.environ.setdefault("XFORMERS_DISABLED", "0")
        except Exception:
            os.environ["XFORMERS_DISABLED"] = "1"
            os.environ["DISABLE_XFORMERS"] = "1"
    except Exception:
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["DISABLE_XFORMERS"] = "1"

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from src.utils.logger import get_logger
from src.utils.error_handler import retry_on_failure, ModelLoadError, handle_gpu_oom
from src.pipeline.model_cache import get_model_cache
from src.pipeline.config import get_config

logger = get_logger(__name__)


class StepOffCallback:
    """Callback for ControlNet step-off schedule during generation"""
    
    def __init__(self, step_off_ratio: float = 0.75, initial_canny_weight: float = 0.4, initial_depth_weight: float = 0.6):
        """
        Initialize step-off callback
        
        Args:
            step_off_ratio: Ratio of steps before canny ControlNet is turned off (0.0-1.0)
            initial_canny_weight: Initial weight for canny ControlNet
            initial_depth_weight: Weight for depth ControlNet (constant)
        """
        self.step_off_ratio = step_off_ratio
        self.initial_canny_weight = initial_canny_weight
        self.initial_depth_weight = initial_depth_weight
        self.current_weights = [initial_depth_weight, initial_canny_weight]
    
    def __call__(self, pipe, step_index: int, timestep, callback_kwargs):
        """
        Adjust ControlNet weights based on step index
        
        Note: Diffusers may not support per-step controlnet_conditioning_scale changes
        in callbacks for all pipeline types. This callback attempts to modify weights
        but may fall back to workaround if not supported.
        
        Args:
            pipe: The pipeline instance
            step_index: Current step index (0-based)
            timestep: Current timestep
            callback_kwargs: Callback kwargs dict
        
        Returns:
            Modified callback_kwargs
        """
        total_steps = len(pipe.scheduler.timesteps)
        steps_before_off = int(total_steps * self.step_off_ratio)
        
        if step_index < steps_before_off:
            # Full canny weight before step-off
            canny_weight = self.initial_canny_weight
        else:
            # Fade to zero after step_off_ratio
            progress = (step_index - steps_before_off) / max(1, total_steps - steps_before_off)
            canny_weight = self.initial_canny_weight * (1 - progress)
        
        # Update weights
        self.current_weights = [self.initial_depth_weight, canny_weight]
        
        # Try to update controlnet_conditioning_scale if supported
        if "controlnet_conditioning_scale" in callback_kwargs:
            callback_kwargs["controlnet_conditioning_scale"] = self.current_weights
        
        return callback_kwargs


class SDXLGenerator:
    """SDXL generator with Multi-ControlNet and LoRA support"""
    
    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_path: Optional[str] = None,
        lora_scale: float = 0.8,
        device: str = "cuda",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "blurry, distorted, anatomically incorrect"
    ):
        """Initialize SDXL generator (lazy loading)"""
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.pipe = None
        self.controlnet_processor = None
        # Models loaded lazily on first use
    
    @retry_on_failure(max_attempts=3, exceptions=(Exception,))
    def _load_pipeline(self):
        """Load SDXL pipeline with ControlNet and LoRA"""
        try:
            config = get_config()
            hardware_config = config.get_hardware_config()
            
            # Load ControlNets (lazy load if not already loaded)
            from .controlnet_processor import ControlNetProcessor
            if self.controlnet_processor is None:
                self.controlnet_processor = ControlNetProcessor(device=self.device)
            # Ensure models are loaded
            if self.controlnet_processor.depth_controlnet is None or self.controlnet_processor.canny_controlnet is None:
                self.controlnet_processor._load_models()
            controlnets = [self.controlnet_processor.depth_controlnet, self.controlnet_processor.canny_controlnet]
            
            # Load VAE
            cache = get_model_cache()
            vae = cache.load_or_cache_model(
                model_id=self.base_model_id,
                model_loader=lambda model_id, **kwargs: AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                    torch_dtype=torch.float16
                ),
                cache_key=f"vae_{self.base_model_id}"
            )
            
            # Create pipeline
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.base_model_id,
                controlnet=controlnets,
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Apply torch.compile() optimization if available (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    self.pipe.unet = torch.compile(
                        self.pipe.unet,
                        mode="reduce-overhead",
                        fullgraph=True
                    )
                    logger.info("torch.compile_enabled")
                except Exception as e:
                    logger.warning("torch.compile_failed", error=str(e), exc_info=True)
            
            # Load LoRA if provided
            if self.lora_path:
                self.pipe.load_lora_weights(self.lora_path)
                # Set LoRA scale (if supported by pipeline)
                if hasattr(self.pipe, 'set_adapters'):
                    self.pipe.set_adapters(['default'], adapter_weights=[self.lora_scale])
                logger.info("lora_loaded", lora_path=self.lora_path, scale=self.lora_scale)
            
            # Set scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Memory optimizations
            if hardware_config.get("enable_xformers", True):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers_enabled")
                except ImportError:
                    # Fall back to PyTorch 2.0+ SDPA if xformers unavailable
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        logger.info("using_pytorch_sdpa")
                    else:
                        logger.warning("memory_efficient_attention_not_available")
                except Exception as e:
                    logger.warning("xformers_not_available", error=str(e), exc_info=True)
            
            # CPU offloading for SDXL (preferred over attention slicing for SDXL)
            if hardware_config.get("enable_model_cpu_offload", True):
                try:
                    self.pipe.enable_model_cpu_offload()
                    logger.info("model_cpu_offload_enabled")
                except Exception as e:
                    logger.warning("model_cpu_offload_failed", error=str(e), exc_info=True)
                    # Fall back to attention slicing if CPU offload fails
                    if hardware_config.get("enable_attention_slicing", True):
                        # Only use attention slicing if PyTorch < 2.0 or SDPA not available
                        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                            self.pipe.enable_attention_slicing(slice_size="auto")
                            logger.info("attention_slicing_enabled_fallback")
            elif hardware_config.get("enable_attention_slicing", True):
                # Only use attention slicing if PyTorch < 2.0 or SDPA not available
                if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    self.pipe.enable_attention_slicing(slice_size="auto")
                    logger.info("attention_slicing_enabled")
            
            logger.info("sdxl_pipeline_loaded", base_model=self.base_model_id)
            
        except Exception as e:
            raise ModelLoadError(
                phase="phase2",
                message=f"Failed to load SDXL pipeline: {str(e)}",
                original_error=e
            )
    
    @handle_gpu_oom
    def generate(
        self,
        prompt: str,
        control_images: List[np.ndarray],
        control_weights: Optional[List[float]] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        controlnet_processor: Optional[object] = None
    ) -> np.ndarray:
        """
        Generate image with ControlNet guidance and Canny step-off schedule
        
        Args:
            prompt: Text prompt
            control_images: List of control images [depth, canny]
            control_weights: Optional initial weights for each control image
            num_inference_steps: Override default steps
            guidance_scale: Override default guidance scale
            seed: Random seed for reproducibility
            controlnet_processor: Optional ControlNetProcessor instance for step-off schedule
        
        Returns:
            Generated image as numpy array (RGB, uint8)
        """
        if self.pipe is None:
            self._load_pipeline()
        
        steps = num_inference_steps or self.num_inference_steps
        guidance = guidance_scale or self.guidance_scale
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Prepare control images
        from PIL import Image
        control_pil = [Image.fromarray(img.astype(np.uint8)) for img in control_images]
        
        # Get initial control weights
        initial_weights = control_weights or [0.6, 0.4]
        depth_weight = initial_weights[0]
        canny_weight = initial_weights[1]
        
        # Use stored controlnet_processor or provided one
        processor = controlnet_processor or getattr(self, 'controlnet_processor', None)
        
        # Setup step-off callback if processor has step-off schedule
        callback = None
        control_weights_to_use = initial_weights
        
        if processor is not None and hasattr(processor, 'canny_step_off'):
            canny_step_off = processor.canny_step_off
            callback = StepOffCallback(
                step_off_ratio=canny_step_off,
                initial_canny_weight=canny_weight,
                initial_depth_weight=depth_weight
            )
            logger.info(
                "step_off_callback_created",
                step_off_ratio=canny_step_off,
                initial_canny_weight=canny_weight,
                initial_depth_weight=depth_weight
            )
            
            # Note: Diffusers may not support per-step controlnet_conditioning_scale
            # changes in callbacks. If callback doesn't work, we fall back to
            # averaged weights approximation
            # Calculate average canny weight as fallback
            steps_before_off = int(steps * canny_step_off)
            # Prevent division by zero
            avg_canny_weight = (canny_weight * steps_before_off) / max(steps, 1)
            control_weights_to_use = [depth_weight, avg_canny_weight]
            logger.info(
                "canny_step_off_fallback",
                avg_canny_weight=avg_canny_weight,
                steps_before_off=steps_before_off
            )
        
        try:
            # Try to use callback if available
            # Note: Some diffusers versions may not support callbacks modifying
            # controlnet_conditioning_scale, so we also set it as a parameter
            pipe_kwargs = {
                "prompt": prompt,
                "negative_prompt": self.negative_prompt,
                "image": control_pil,
                "controlnet_conditioning_scale": control_weights_to_use,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator
            }
            
            if callback is not None:
                pipe_kwargs["callback_on_step_end"] = callback
            
            result = self.pipe(**pipe_kwargs).images[0]
            
            # Convert to numpy
            result_array = np.array(result)
            
            logger.info("sdxl_generation_complete", output_shape=result_array.shape)
            
            return result_array
            
        except Exception as e:
            logger.error("sdxl_generation_failed", error=str(e), exc_info=True)
            raise

