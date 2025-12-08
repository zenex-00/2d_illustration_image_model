"""LoRA training for SDXL using diffusers"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from diffusers.utils import load_image
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from tqdm.auto import tqdm

from src.utils.logger import get_logger
from src.pipeline.config import get_config

logger = get_logger(__name__)


def train_lora(
    input_dir: str,
    target_dir: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    rank: int = 32,
    alpha: int = 16,
    validation_split: float = 0.2,
    seed: int = 42,
    save_steps: int = 500,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    control_images_dir: Optional[str] = None,
    controlnet_depth_weight: float = 0.6,
    controlnet_canny_weight: float = 0.4,
) -> str:
    """
    Train LoRA adapter for SDXL on input-target image pairs
    
    Args:
        input_dir: Directory containing input images
        target_dir: Directory containing target images (must match input filenames)
        output_dir: Directory to save trained LoRA weights
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        rank: LoRA rank (lower = smaller model, faster training)
        alpha: LoRA alpha (scaling factor)
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        save_steps: Save checkpoint every N steps
        progress_callback: Optional callback function to report progress
        control_images_dir: Optional directory containing control images (depth and edge maps)
        controlnet_depth_weight: Weight for depth ControlNet conditioning
        controlnet_canny_weight: Weight for canny ControlNet conditioning
        
    Returns:
        Path to saved LoRA weights file
    """
    config = get_config()
    hardware_config = config.get_hardware_config()
    device = hardware_config.get("device", "cuda")
    precision = hardware_config.get("precision", "float16")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info("training_start", input_dir=input_dir, target_dir=target_dir, num_epochs=num_epochs)
    
    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16" if precision == "float16" else "no",
        project_config=project_config,
    )
    
    # Load image pairs
    input_path = Path(input_dir)
    target_path = Path(target_dir)
    
    # Extract numeric prefix and match by index
    def extract_index(filename: str) -> Optional[int]:
        """Extract numeric prefix from filename (e.g., '0000_image.jpg' -> 0)"""
        try:
            idx_str = filename.split('_')[0]
            return int(idx_str)
        except (ValueError, IndexError):
            return None
    
    input_dict = {}
    for f in input_path.iterdir():
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            idx = extract_index(f.name)
            if idx is not None:
                input_dict[idx] = f
    
    target_dict = {}
    for f in target_path.iterdir():
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            idx = extract_index(f.name)
            if idx is not None:
                target_dict[idx] = f
    
    # Validate indices match
    if set(input_dict.keys()) != set(target_dict.keys()):
        missing_inputs = set(target_dict.keys()) - set(input_dict.keys())
        missing_targets = set(input_dict.keys()) - set(target_dict.keys())
        raise ValueError(f"Index mismatch: missing inputs for {missing_inputs}, missing targets for {missing_targets}")
    
    # Create matched pairs
    input_files = [input_dict[i] for i in sorted(input_dict.keys())]
    target_files = [target_dict[i] for i in sorted(target_dict.keys())]
    
    if len(input_files) < 10:
        raise ValueError(f"Need at least 10 image pairs, got {len(input_files)}")
    
    logger.info("image_pairs_loaded", count=len(input_files))
    
    # Create train/validation split
    indices = np.arange(len(input_files))
    np.random.shuffle(indices)
    split_idx = int(len(input_files) * (1 - validation_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_pairs = [(input_files[i], target_files[i]) for i in train_indices]
    val_pairs = [(input_files[i], target_files[i]) for i in val_indices]
    
    # Store index mapping for reliable control image lookup
    file_index_map = {path: idx for idx, path in enumerate(input_files)}
    
    logger.info("dataset_split", train=len(train_pairs), val=len(val_pairs))
    
    # Load base model
    base_model_id = config.get_phase_config("phase2").get("sdxl", {}).get("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
    
    logger.info("loading_base_model", model_id=base_model_id)
    
    # Load tokenizers and text encoders
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder_2")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    
    # Load UNet
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    
    # Load ControlNet models if control images directory is provided
    depth_controlnet = None
    canny_controlnet = None
    use_controlnet = control_images_dir is not None and Path(control_images_dir).exists()
    
    if use_controlnet:
        logger.info("loading_controlnet_models")
        try:
            depth_controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=torch.float16 if precision == "float16" else torch.float32
            )
            canny_controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16 if precision == "float16" else torch.float32
            )
            # Freeze ControlNet weights (they won't be trained)
            depth_controlnet.requires_grad_(False)
            canny_controlnet.requires_grad_(False)
            logger.info("controlnet_models_loaded")
        except Exception as e:
            logger.warning("controlnet_load_failed", error=str(e))
            logger.warning("training_without_controlnet")
            use_controlnet = False
    
    # Convert to appropriate precision
    if precision == "float16":
        text_encoder = text_encoder.to(dtype=torch.float16)
        text_encoder_2 = text_encoder_2.to(dtype=torch.float16)
        vae = vae.to(dtype=torch.float16)
        unet = unet.to(dtype=torch.float16)
        if depth_controlnet is not None:
            depth_controlnet = depth_controlnet.to(dtype=torch.float16)
        if canny_controlnet is not None:
            canny_controlnet = canny_controlnet.to(dtype=torch.float16)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Setup scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Prepare with accelerator
    unet, optimizer, train_pairs, val_pairs = accelerator.prepare(
        unet, optimizer, train_pairs, val_pairs
    )
    
    # Move models to device
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    vae.to(accelerator.device)
    if depth_controlnet is not None:
        depth_controlnet.to(accelerator.device)
    if canny_controlnet is not None:
        canny_controlnet.to(accelerator.device)
    
    # Freeze text encoders and VAE
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Training loop
    global_step = 0
    num_train_steps = len(train_pairs) * num_epochs // batch_size
    
    # Define transform for image preprocessing (used in both training and validation)
    from torchvision.transforms import ToTensor, Normalize, Compose
    # VAE expects images in [0, 1] range, then we normalize to [-1, 1]
    transform = Compose([
        ToTensor(),  # Converts to [0, 1]
        Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    logger.info("training_loop_start", total_steps=num_train_steps)
    
    for epoch in range(num_epochs):
        unet.train()
        train_loss = 0.0
        
        # Shuffle training pairs
        epoch_pairs = train_pairs.copy()
        np.random.shuffle(epoch_pairs)
        
        progress_bar = tqdm(
            range(0, len(epoch_pairs), batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step_idx in progress_bar:
            batch_start = step_idx
            batch_end = min(step_idx + batch_size, len(epoch_pairs))
            batch_pairs = epoch_pairs[batch_start:batch_end]
            
            # Load and preprocess images
            input_images = []
            target_images = []
            control_images_depth = []
            control_images_canny = []
            
            for batch_idx, (input_path, target_path) in enumerate(batch_pairs):
                input_img = load_image(str(input_path)).convert("RGB")
                target_img = load_image(str(target_path)).convert("RGB")
                
                # Resize to 1024x1024 for SDXL
                input_img = input_img.resize((1024, 1024), Image.LANCZOS)
                target_img = target_img.resize((1024, 1024), Image.LANCZOS)
                
                input_images.append(input_img)
                target_images.append(target_img)
                
                # Load control images if available
                if use_controlnet:
                    try:
                        input_filename = Path(input_path).name
                        # Try to extract leading number
                        try:
                            idx_str = input_filename.split("_")[0]
                            control_idx = int(idx_str)
                        except (ValueError, IndexError):
                            # Fallback: use file index from the sorted list
                            control_idx = file_index_map.get(input_path, batch_idx)
                            if control_idx == batch_idx:
                                # Last resort: try regex to find any number
                                import re
                                match = re.search(r'(\d+)', input_filename)
                                if match:
                                    control_idx = int(match.group(1))
                        
                        depth_path = Path(control_images_dir) / f"depth_{control_idx:04d}.png"
                        edge_path = Path(control_images_dir) / f"edge_{control_idx:04d}.png"
                        
                        if depth_path.exists() and edge_path.exists():
                            depth_img = load_image(str(depth_path)).convert("RGB")
                            edge_img = load_image(str(edge_path)).convert("RGB")
                            
                            # Resize to 1024x1024
                            depth_img = depth_img.resize((1024, 1024), Image.LANCZOS)
                            edge_img = edge_img.resize((1024, 1024), Image.LANCZOS)
                            
                            control_images_depth.append(depth_img)
                            control_images_canny.append(edge_img)
                        else:
                            # Create placeholder if control images missing
                            logger.warning("control_images_missing", depth_path=str(depth_path), edge_path=str(edge_path))
                            placeholder = Image.new("RGB", (1024, 1024), (0, 0, 0))
                            control_images_depth.append(placeholder)
                            control_images_canny.append(placeholder)
                    except Exception as e:
                        logger.warning("control_image_load_failed", error=str(e), batch_idx=batch_idx)
                        # Create placeholder on error
                        placeholder = Image.new("RGB", (1024, 1024), (0, 0, 0))
                        control_images_depth.append(placeholder)
                        control_images_canny.append(placeholder)
            
            # Convert to tensors and normalize for VAE
            input_tensors = torch.stack([transform(img) for img in input_images]).to(accelerator.device)
            target_tensors = torch.stack([transform(img) for img in target_images]).to(accelerator.device)
            
            # Prepare control images if using ControlNet
            control_depth_tensors = None
            control_canny_tensors = None
            if use_controlnet and control_images_depth and control_images_canny:
                control_depth_tensors = torch.stack([transform(img) for img in control_images_depth]).to(accelerator.device)
                control_canny_tensors = torch.stack([transform(img) for img in control_images_canny]).to(accelerator.device)
            
            # Encode target images to latents (these are our ground truth)
            with torch.no_grad():
                # Encode target images to latents
                target_latents = vae.encode(target_tensors).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (target_latents.shape[0],),
                device=accelerator.device
            ).long()
            
            # Add noise to target latents
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # Get text embeddings with trigger token for LoRA activation
            # Use same prompt format as inference to ensure consistency
            prompt = "flat vector illustration <flt_vctr_style>, minimalist style, clean lines, solid colors"
            with torch.no_grad():
                # Tokenize
                tokens = tokenizer(
                    [prompt] * len(batch_pairs),
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                
                tokens_2 = tokenizer_2(
                    [prompt] * len(batch_pairs),
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                
                # Encode
                text_embeds = text_encoder(tokens.input_ids).last_hidden_state
                text_encoder_2_output = text_encoder_2(tokens_2.input_ids, output_hidden_states=True)
                text_embeds_2 = text_encoder_2_output.last_hidden_state
                pooled_text_embeds = text_encoder_2_output.text_embeds
                
                # Concatenate for SDXL
                text_embeds = torch.cat([text_embeds, text_embeds_2], dim=-1)
            
            # Calculate time_ids for SDXL (original_size, crops_coords_top_left, target_size)
            # SDXL uses 1024x1024 by default
            original_size = torch.tensor([[1024, 1024]] * len(batch_pairs), device=accelerator.device, dtype=torch.long)
            crops_coords_top_left = torch.tensor([[0, 0]] * len(batch_pairs), device=accelerator.device, dtype=torch.long)
            target_size = torch.tensor([[1024, 1024]] * len(batch_pairs), device=accelerator.device, dtype=torch.long)
            
            # Concatenate to form time_ids (batch_size, 6)
            time_ids = torch.cat([original_size, crops_coords_top_left, target_size], dim=1)
            
            # Predict noise with ControlNet conditioning if available
            # Note: Training uses fixed ControlNet weights (depth_weight, canny_weight),
            # while inference uses a step-off schedule that reduces canny weight over time.
            # This is an intentional design difference - training focuses on learning from
            # consistent conditioning, while inference benefits from gradual canny fade-out.
            if use_controlnet and control_depth_tensors is not None and control_canny_tensors is not None:
                # Encode control images to latents
                with torch.no_grad():
                    depth_latents = vae.encode(control_depth_tensors).latent_dist.sample()
                    depth_latents = depth_latents * vae.config.scaling_factor
                    canny_latents = vae.encode(control_canny_tensors).latent_dist.sample()
                    canny_latents = canny_latents * vae.config.scaling_factor
                
                # Get ControlNet outputs
                with torch.no_grad():
                    depth_down_block_res_samples, depth_mid_block_res_sample = depth_controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        controlnet_cond=depth_latents,
                        added_cond_kwargs={
                            "text_embeds": pooled_text_embeds,
                            "time_ids": time_ids
                        },
                        return_dict=False,
                    )
                    
                    canny_down_block_res_samples, canny_mid_block_res_sample = canny_controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        controlnet_cond=canny_latents,
                        added_cond_kwargs={
                            "text_embeds": pooled_text_embeds,
                            "time_ids": time_ids
                        },
                        return_dict=False,
                    )
                
                # Combine ControlNet outputs with weights
                down_block_res_samples = []
                for depth_res, canny_res in zip(depth_down_block_res_samples, canny_down_block_res_samples):
                    combined = depth_res * controlnet_depth_weight + canny_res * controlnet_canny_weight
                    down_block_res_samples.append(combined)
                
                mid_block_res_sample = (
                    depth_mid_block_res_sample * controlnet_depth_weight +
                    canny_mid_block_res_sample * controlnet_canny_weight
                )
                
                # Predict noise with ControlNet conditioning
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_text_embeds,
                        "time_ids": time_ids
                    },
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                # Predict noise without ControlNet
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_text_embeds,
                        "time_ids": time_ids
                    },
                ).sample
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            global_step += 1
            
            # Update progress
            if progress_callback:
                progress_callback({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": loss.item(),
                    "progress": (global_step / num_train_steps) * 100,
                })
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save checkpoint
            if global_step % save_steps == 0:
                save_path = Path(output_dir) / f"checkpoint-{global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                unet.save_pretrained(save_path)
                logger.info("checkpoint_saved", path=str(save_path), step=global_step)
        
        avg_train_loss = train_loss / len(progress_bar)
        logger.info("epoch_complete", epoch=epoch + 1, avg_loss=avg_train_loss)
        
        # Validation (optional, can be simplified)
        if len(val_pairs) > 0 and (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            unet.eval()
            val_losses = []
            with torch.no_grad():
                for val_input_path, val_target_path in val_pairs[:10]:  # Sample 10 for speed
                    # Load and preprocess (similar to training)
                    val_input_img = load_image(str(val_input_path)).convert("RGB")
                    val_target_img = load_image(str(val_target_path)).convert("RGB")
                    val_input_img = val_input_img.resize((1024, 1024), Image.LANCZOS)
                    val_target_img = val_target_img.resize((1024, 1024), Image.LANCZOS)
                    
                    # Convert to tensors
                    val_input_tensor = transform(val_input_img).unsqueeze(0).to(accelerator.device)
                    val_target_tensor = transform(val_target_img).unsqueeze(0).to(accelerator.device)
                    
                    # Encode to latents
                    val_target_latents = vae.encode(val_target_tensor).latent_dist.sample()
                    val_target_latents = val_target_latents * vae.config.scaling_factor
                    
                    # Sample noise and timesteps
                    val_noise = torch.randn_like(val_target_latents)
                    val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=accelerator.device).long()
                    val_noisy_latents = noise_scheduler.add_noise(val_target_latents, val_noise, val_timesteps)
                    
                    # Get text embeddings
                    val_tokens = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(accelerator.device)
                    val_tokens_2 = tokenizer_2([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(accelerator.device)
                    val_text_embeds = text_encoder(val_tokens.input_ids).last_hidden_state
                    val_text_encoder_2_output = text_encoder_2(val_tokens_2.input_ids, output_hidden_states=True)
                    val_text_embeds_2 = val_text_encoder_2_output.last_hidden_state
                    val_pooled_text_embeds = val_text_encoder_2_output.text_embeds
                    val_text_embeds = torch.cat([val_text_embeds, val_text_embeds_2], dim=-1)
                    
                    # Time IDs
                    val_original_size = torch.tensor([[1024, 1024]], device=accelerator.device, dtype=torch.long)
                    val_crops_coords_top_left = torch.tensor([[0, 0]], device=accelerator.device, dtype=torch.long)
                    val_target_size = torch.tensor([[1024, 1024]], device=accelerator.device, dtype=torch.long)
                    val_time_ids = torch.cat([val_original_size, val_crops_coords_top_left, val_target_size], dim=1)
                    
                    # Forward pass (simplified - no ControlNet for validation)
                    val_model_pred = unet(
                        val_noisy_latents,
                        val_timesteps,
                        encoder_hidden_states=val_text_embeds,
                        added_cond_kwargs={
                            "text_embeds": val_pooled_text_embeds,
                            "time_ids": val_time_ids
                        },
                    ).sample
                    
                    # Calculate loss
                    val_loss = F.mse_loss(val_model_pred.float(), val_noise.float(), reduction="mean")
                    val_losses.append(val_loss.item())
            
            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                logger.info("validation_complete", epoch=epoch + 1, val_loss=avg_val_loss)
                if progress_callback:
                    progress_callback({
                        "epoch": epoch + 1,
                        "step": global_step,
                        "val_loss": avg_val_loss,
                        "progress": (global_step / num_train_steps) * 100,
                    })
    
    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    final_model_path = output_path / "vector_style_lora.safetensors"
    
    # Save LoRA weights using PEFT's save_pretrained (saves adapter_config.json and adapter_model.safetensors)
    unet.save_pretrained(str(output_path))
    
    # Also try to save a single safetensors file with all LoRA weights
    try:
        from safetensors.torch import save_file
        # Get the PEFT state dict (only LoRA weights)
        lora_state_dict = {}
        for name, param in unet.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_state_dict[name] = param.data.cpu()
        
        # Also check the state dict for LoRA weights
        full_state_dict = unet.state_dict()
        for k, v in full_state_dict.items():
            if "lora" in k.lower() and k not in lora_state_dict:
                lora_state_dict[k] = v.cpu()
        
        if lora_state_dict:
            save_file(lora_state_dict, str(final_model_path))
            logger.info("lora_saved_safetensors", path=str(final_model_path))
        else:
            logger.warning("no_lora_weights_found")
            # Fallback: use the adapter_model.safetensors from PEFT
            adapter_path = output_path / "adapter_model.safetensors"
            if adapter_path.exists():
                final_model_path = adapter_path
                logger.info("using_peft_adapter", path=str(final_model_path))
    except Exception as e:
        logger.warning("safetensors_save_failed", error=str(e))
        # Fallback: use the adapter_model.safetensors from PEFT
        adapter_path = output_path / "adapter_model.safetensors"
        if adapter_path.exists():
            final_model_path = adapter_path
            logger.info("using_peft_adapter_fallback", path=str(final_model_path))
    
    logger.info("training_complete", output_path=str(output_path))
    
    # Return the path to the weights file
    if final_model_path.exists():
        return str(final_model_path)
    else:
        # Return directory path as fallback
        return str(output_path)
