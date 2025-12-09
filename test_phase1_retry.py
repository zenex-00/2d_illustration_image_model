#!/usr/bin/env python3
"""
Test script for Phase 1 sanitization with retry logic
"""

import numpy as np
import sys
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase1_semantic_sanitization.sanitizer import Phase1Sanitizer
from src.pipeline.config import get_config
from src.utils.image_utils import load_image, save_image


def test_phase1_retry_logic():
    """Test the Phase 1 sanitization with retry logic."""
    print("Testing Phase 1 sanitization with retry logic...")

    # Get configuration
    config = get_config()

    # Initialize sanitizer
    sanitizer = Phase1Sanitizer(config)

    # Create a simple test image with some content
    # For testing purposes, we'll use a real image if available or create a dummy one
    test_image_path = project_root / "test_input.jpg"

    if test_image_path.exists():
        print(f"Loading test image from {test_image_path}")
        test_image = load_image(str(test_image_path))
    else:
        # Create a dummy test image (white background with some colored shapes)
        print("Creating dummy test image...")
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White background

        # Add some colored shapes to simulate objects that might be detected
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
        cv2.circle(test_image, (300, 300), 50, (0, 255, 0), -1)  # Green circle
        cv2.putText(test_image, "TEST", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text

    print(f"Test image shape: {test_image.shape}")

    try:
        print("Starting Phase 1 sanitization...")
        result_image, metadata = sanitizer.sanitize(
            image=test_image,
            correlation_id="test-correlation-id"
        )

        print(f"Sanitization completed successfully!")
        print(f"Result shape: {result_image.shape}")
        print(f"Metadata: {metadata}")

        # Save the result for inspection
        output_path = project_root / "test_output_phase1.jpg"
        save_image(result_image, str(output_path))
        print(f"Result saved to {output_path}")

        # Check if the result is different from input (indicating processing occurred)
        if np.array_equal(test_image, result_image):
            print("WARNING: Output is identical to input - no sanitization may have occurred")
        else:
            print("SUCCESS: Output differs from input - sanitization occurred")

        return True

    except Exception as e:
        print(f"Error during sanitization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_access():
    """Test that the configuration is properly loaded."""
    print("\nTesting configuration access...")

    config = get_config()
    training_config = config.get("training", {})
    phase1_retry_config = training_config.get("phase1_retry", {})

    print(f"Phase 1 retry config: {phase1_retry_config}")

    expected_keys = ["max_retries", "retry_on_no_change", "skip_on_failure", "parameter_variations", "fallback_models"]
    missing_keys = [key for key in expected_keys if key not in phase1_retry_config]

    if missing_keys:
        print(f"ERROR: Missing keys in phase1_retry config: {missing_keys}")
        return False
    else:
        print("SUCCESS: All expected configuration keys are present")
        return True


if __name__ == "__main__":
    print("Phase 1 Sanitization Retry Logic Test")
    print("=" * 50)

    # Test configuration access
    config_ok = test_config_access()

    if not config_ok:
        print("Configuration test failed, exiting...")
        sys.exit(1)

    # Test the actual sanitization
    sanitization_ok = test_phase1_retry_logic()

    if sanitization_ok:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)