#!/bin/bash
# Quick script to download SYNTH dataset subsets for testing

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SYNTH Dataset Subset Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default values
SAMPLES=${1:-100}
OUTPUT_DIR=${2:-data/demo}

echo -e "${GREEN}Configuration:${NC}"
echo "  Max samples: $SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if we're in the right directory
if [ ! -f "prepare_synth_data.py" ]; then
    echo -e "${YELLOW}⚠️  Warning: prepare_synth_data.py not found${NC}"
    echo "   Please run this script from the Baguettotron-321M root directory"
    exit 1
fi

# Run the download
echo -e "${GREEN}📥 Downloading $SAMPLES samples...${NC}"
python prepare_synth_data.py \
    --subset \
    --max-samples "$SAMPLES" \
    --tokenize \
    --output-dir "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}✅ Done! Dataset ready at: $OUTPUT_DIR${NC}"
echo ""
echo -e "${BLUE}Quick commands:${NC}"
echo "  # View the data"
echo "  ls -lh $OUTPUT_DIR/"
echo ""
echo "  # Check number of sequences"
echo "  python -c \"import json; print(len(json.load(open('$OUTPUT_DIR/train_tokens.json'))))\""
echo ""
echo "  # Start training with this subset"
echo "  python train.py --train-data $OUTPUT_DIR/train_tokens.json --max-epochs 1"
