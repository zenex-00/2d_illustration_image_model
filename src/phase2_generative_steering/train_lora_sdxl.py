"""LoRA training script for SDXL style transfer"""

import os
import json
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any
from datetime import datetime
from PIL import Image
import numpy as np

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.utils import make_image_grid
import peft
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from accelerate import Accelerator
from accelerate.utils import set_seed
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.logger import get_logger, setup_logging
from src.pipeline.config import get_config

logger = get_logger(__name__)

# Style token
STYLE_TOKEN = "<flt_vctr_style>"


class LoRADataset:
    """Dataset for LoRA training with augmentation"""
    
    def __init__(
        self,
        image_pairs: List[Tuple[str, str]],  # (input_path, output_path)
        tokenizer,
        text_encoder,
        text_encoder_2,
        size: int = 1024,
        augmentation: bool = True
    ):
        self.image_pairs = image_pairs
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.size = size
        self.augmentation = augmentation
        
        # Augmentation pipeline
        if augmentation:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        input_path, output_path = self.image_pairs[idx]
        
        # Validate file existence
        import os
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output image not found: {output_path}")
        
        # Load images with context manager and error handling
        try:
            with Image.open(input_path) as img:
                input_img = img.convert("RGB").resize((self.size, self.size))
                input_array = np.array(input_img)
        except (IOError, OSError) as e:
            from PIL import UnidentifiedImageError
            if isinstance(e, UnidentifiedImageError):
                raise ValueError(f"Corrupted or unsupported image format: {input_path}") from e
            raise IOError(f"Failed to load image: {input_path}") from e
        
        try:
            with Image.open(output_path) as img:
                output_img = img.convert("RGB").resize((self.size, self.size))
                output_array = np.array(output_img)
        except (IOError, OSError) as e:
            from PIL import UnidentifiedImageError
            if isinstance(e, UnidentifiedImageError):
                raise ValueError(f"Corrupted or unsupported image format: {output_path}") from e
            raise IOError(f"Failed to load image: {output_path}") from e
        
        # Apply augmentation
        
        # Augment both images with same seed for consistency
        if self.augmentation:
            seed = np.random.randint(0, 2**32)
            A.set_seed(seed)
            transformed = self.transform(image=input_array)
            input_tensor = transformed["image"]
            
            A.set_seed(seed)
            transformed = self.transform(image=output_array)
            output_tensor = transformed["image"]
        else:
            transformed = self.transform(image=input_array)
            input_tensor = transformed["image"]
            transformed = self.transform(image=output_array)
            output_tensor = transformed["image"]
        
        # Generate caption
        caption = self._generate_caption(input_path)
        
        # Tokenize caption
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(caption)
        
        return {
            "pixel_values": output_tensor,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds
        }
    
    def _generate_caption(self, input_path: str) -> str:
        """Generate synthetic caption for image"""
        # Simple caption generation - can be enhanced with BLIP-2
        base_caption = "flat vector illustration, minimalist style, clean lines, solid colors, side view, white background"
        return f"{base_caption} {STYLE_TOKEN}"
    
    def _encode_prompt(self, prompt: str):
        """Encode prompt using both text encoders"""
        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode with text encoder 1
        with torch.no_grad():
            prompt_embeds = self.text_encoder(tokens.input_ids.to(self.text_encoder.device))[0]
        
        # Encode with text encoder 2
        tokens_2 = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            prompt_embeds_2 = self.text_encoder_2(tokens_2.input_ids.to(self.text_encoder_2.device))[0]
            pooled_prompt_embeds = self.text_encoder_2.get_pooled_output()
        
        # Concatenate embeddings
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds


def train_lora(
    input_dir: str,
    output_dir: str,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    rank: int = 32,
    alpha: int = 16,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    num_epochs: int = 10,
    validation_split: float = 0.2,
    save_steps: int = 100,
    output_path: str = "vector_style_v1.safetensors",
    callbacks: Optional[Any] = None
):
    """
    Train LoRA for style transfer.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory containing output (target) images
        base_model: Base SDXL model ID
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        validation_split: Validation split ratio
        save_steps: Save checkpoint every N steps
        output_path: Output path for trained LoRA
        callbacks: Optional callbacks object with methods: on_epoch_start, on_epoch_end, on_step
    """
    setup_logging()
    logger.info("lora_training_start", base_model=base_model, rank=rank, alpha=alpha)
    logger.info(f"peft_version: {peft.__version__}")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )
    
    set_seed(42)
    
    # Load base model
    logger.info("loading_base_model", model_id=base_model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Setup noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model,
        subfolder="scheduler"
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
        lora_dropout=0.1,  # Add dropout for regularization
    )
    
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    
    # Prepare dataset
    input_path = Path(input_dir)
    output_path_dir = Path(output_dir)
    
    # Find image pairs
    image_pairs = []
    for input_file in input_path.glob("*.png"):
        output_file = output_path_dir / input_file.name
        if output_file.exists():
            image_pairs.append((str(input_file), str(output_file)))
    
    if len(image_pairs) < 10:
        raise ValueError(f"Insufficient image pairs: {len(image_pairs)}. Need at least 10.")
    
    logger.info("image_pairs_found", count=len(image_pairs))
    
    # Split train/validation
    split_idx = int(len(image_pairs) * (1 - validation_split))
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    # Create datasets
    train_dataset = LoRADataset(
        train_pairs,
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.text_encoder_2,
        augmentation=True
    )
    
    val_dataset = LoRADataset(
        val_pairs,
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.text_encoder_2,
        augmentation=False
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Prepare with accelerator
    pipe.unet, pipe.vae, optimizer, train_loader, val_loader = accelerator.prepare(
        pipe.unet, pipe.vae, optimizer, train_loader, val_loader
    )
    
    # Set VAE to eval mode (we don't train VAE)
    pipe.vae.eval()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    total_steps = len(train_loader) * num_epochs
    
    for epoch in range(num_epochs):
        # Callback: epoch start
        if callbacks and hasattr(callbacks, 'on_epoch_start'):
            callbacks.on_epoch_start(epoch + 1, num_epochs)
        
        pipe.unet.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(pipe.unet):
                # 1. Encode images to latent space
                with torch.no_grad():
                    # Convert pixel values to proper format (batch, channels, height, width)
                    # batch["pixel_values"] should already be in [0, 1] range from normalization
                    pixel_values = batch["pixel_values"]
                    if pixel_values.min() < 0:
                        # Denormalize if needed (from [-1, 1] to [0, 1])
                        pixel_values = (pixel_values + 1.0) / 2.0
                    
                    # Encode to latents
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                
                # 2. Sample noise
                noise = torch.randn_like(latents)
                
                # 3. Sample timestep
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (bsz,), 
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # 4. Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 5. Predict noise with UNet
                # Prepare prompt embeddings
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                
                # Predict noise
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds}
                ).sample
                
                # 6. Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Callback: step update
            if callbacks and hasattr(callbacks, 'on_step'):
                callbacks.on_step(global_step, total_steps, loss.item())
            
            if global_step % save_steps == 0:
                # Save checkpoint
                checkpoint_path = f"checkpoint_step_{global_step}.safetensors"
                pipe.unet.save_pretrained(checkpoint_path)
                logger.info("checkpoint_saved", path=checkpoint_path, step=global_step)
        
        # Validation
        pipe.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # Same process as training but without gradient
                pixel_values = batch["pixel_values"]
                if pixel_values.min() < 0:
                    pixel_values = (pixel_values + 1.0) / 2.0
                
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds}
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                val_loss += loss.item()
        
        # Prevent division by zero
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = 0.0
        if len(train_loader) > 0:
            epoch_loss /= len(train_loader)
        else:
            epoch_loss = 0.0
        
        # Callback: epoch end
        if callbacks and hasattr(callbacks, 'on_epoch_end'):
            callbacks.on_epoch_end(epoch + 1, epoch_loss, val_loss)
        
        logger.info(
            "epoch_complete",
            epoch=epoch + 1,
            train_loss=epoch_loss,
            val_loss=val_loss,
            global_step=global_step
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            pipe.unet.save_pretrained(output_path)
            logger.info("best_model_saved", path=output_path, val_loss=val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("early_stopping", epoch=epoch + 1)
                break
    
    # Save final model
    pipe.unet.save_pretrained(output_path)
    logger.info("training_complete", output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA for SDXL style transfer")
    parser.add_argument("--input_dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output images directory")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--output", type=str, default="vector_style_v1.safetensors")
    
    args = parser.parse_args()
    
    train_lora(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_path=args.output
    )



