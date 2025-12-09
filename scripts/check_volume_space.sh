#!/bin/bash
# Script to check and manage RunPod volume space for model downloads

set -e

VOLUME_PATH="${1:-/models}"
echo "Checking space at $VOLUME_PATH..."

# Get the actual disk usage
df -h "$VOLUME_PATH" || {
    echo "Error: Cannot access $VOLUME_PATH"
    exit 1
}

echo ""
echo "Attempting to create a test file to check actual write permissions and quota..."
TEST_FILE="$VOLUME_PATH/test_quota_check.tmp"

# Try to create a file to see if we can actually write
if dd if=/dev/zero of="$TEST_FILE" bs=1M count=10 2>/dev/null; then
    echo "✓ Successfully created 10MB test file"
    rm -f "$TEST_FILE"
    echo "✓ Test file removed"
else
    echo "✗ Failed to create test file - quota may be exceeded or no write permissions"
    df -i "$VOLUME_PATH"  # Check inode usage too
    exit 1
fi

echo ""
echo "Space check completed successfully. You can now run the model setup."

# Suggest running the setup command
echo ""
echo "Run the setup with:"
echo "python scripts/setup_model_volume.py --volume-path $VOLUME_PATH"