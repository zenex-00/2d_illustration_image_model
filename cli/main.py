"""CLI interface for Gemini 3 Pro pipeline"""

import click
import sys
from pathlib import Path
from typing import Optional
from src.pipeline.orchestrator import Gemini3Pipeline
from src.utils.logger import setup_logging, get_logger
from src.pipeline.config import get_config

logger = get_logger(__name__)


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Gemini 3 Pro Vehicle-to-Vector Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=log_level)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_svg', type=click.Path())
@click.option('--output-png', type=click.Path(), help='Optional PNG preview output')
@click.option('--palette', type=click.Path(exists=True), help='Custom palette YAML file')
@click.option('--save-intermediates', is_flag=True, help='Save intermediate phase outputs')
@click.option('--intermediate-dir', type=click.Path(), help='Directory for intermediate outputs')
@click.pass_context
def process(ctx, input_path, output_svg, output_png, palette, save_intermediates, intermediate_dir):
    """Run full pipeline on input image"""
    try:
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        
        palette_list = None
        if palette:
            import yaml
            with open(palette, 'r') as f:
                palette_config = yaml.safe_load(f)
                palette_list = palette_config.get('palette', [])
        
        svg_xml, metadata = pipeline.process_image(
            input_image_path=input_path,
            palette_hex_list=palette_list,
            output_svg_path=output_svg,
            output_png_path=output_png,
            save_intermediates=save_intermediates,
            intermediate_dir=intermediate_dir
        )
        
        click.echo(f"✓ Pipeline completed successfully")
        click.echo(f"  SVG saved to: {output_svg}")
        if output_png:
            click.echo(f"  PNG saved to: {output_png}")
        click.echo(f"  Total time: {metadata['total_processing_time_ms']:.2f}ms")
        
    except Exception as e:
        click.echo(f"✗ Pipeline failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.pass_context
def phase1(ctx, input_path, output_path):
    """Run Phase I only: Semantic Sanitization"""
    try:
        from src.utils.image_utils import load_image, save_image
        from src.phase1_semantic_sanitization.sanitizer import Phase1Sanitizer
        
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        image = load_image(input_path)
        
        clean_plate, metadata = pipeline.phase1.sanitize(image)
        save_image(clean_plate, output_path)
        
        click.echo(f"✓ Phase I completed: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Phase I failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.pass_context
def phase2(ctx, input_path, output_path):
    """Run Phase II only: Generative Steering"""
    try:
        from src.utils.image_utils import load_image, save_image
        from src.phase2_generative_steering.generator import Phase2Generator
        
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        image = load_image(input_path)
        
        vector_raster, metadata = pipeline.phase2.generate(image)
        save_image(vector_raster, output_path)
        
        click.echo(f"✓ Phase II completed: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Phase II failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.pass_context
def phase3(ctx, input_path, output_path):
    """Run Phase III only: Chromatic Enforcement"""
    try:
        from src.utils.image_utils import load_image, save_image
        from src.phase3_chromatic_enforcement.enforcer import Phase3Enforcer
        
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        image = load_image(input_path)
        
        quantized, metadata = pipeline.phase3.enforce(image)
        save_image(quantized, output_path)
        
        click.echo(f"✓ Phase III completed: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Phase III failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.pass_context
def phase4(ctx, input_path, output_path):
    """Run Phase IV only: Vector Reconstruction"""
    try:
        from src.utils.image_utils import load_image
        from src.phase4_vector_reconstruction.vectorizer import Phase4Vectorizer
        
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        image = load_image(input_path)
        
        svg_xml, metadata = pipeline.phase4.vectorize(image, output_path=output_path)
        
        click.echo(f"✓ Phase IV completed: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Phase IV failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path())
@click.pass_context
def batch(ctx, input_dir, output_dir):
    """Process batch of images"""
    try:
        pipeline = Gemini3Pipeline(config_path=ctx.obj.get('config'))
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        click.echo(f"Processing {len(image_files)} images...")
        
        for i, img_file in enumerate(image_files, 1):
            click.echo(f"[{i}/{len(image_files)}] Processing {img_file.name}...")
            
            output_svg = output_path / f"{img_file.stem}.svg"
            output_png = output_path / f"{img_file.stem}.png"
            
            try:
                svg_xml, metadata = pipeline.process_image(
                    input_image_path=str(img_file),
                    output_svg_path=str(output_svg),
                    output_png_path=str(output_png)
                )
                click.echo(f"  ✓ {img_file.name} completed")
            except Exception as e:
                click.echo(f"  ✗ {img_file.name} failed: {str(e)}", err=True)
        
        click.echo(f"\n✓ Batch processing completed")
        
    except Exception as e:
        click.echo(f"✗ Batch processing failed: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()






