#!/bin/bash
# Test script for interactive chat mode
# This creates a simple test that simulates interactive input

echo "==================================================================="
echo "Testing Chat Mode - Interactive Simulation"
echo "==================================================================="

# Create a test input file
cat > /tmp/chat_test_input.txt << 'EOF'
Bonjour
quit
EOF

echo ""
echo "Simulating interactive chat with pre-written input:"
echo "  Input 1: 'Bonjour'"
echo "  Input 2: 'quit'"
echo ""

# Run interactive mode with input from file
cd "$(dirname "$0")/.."
cat /tmp/chat_test_input.txt | ./baguettotron generate \
  --checkpoint outputs/tiny_correct/ckpt_50/model.pt \
  --chat-mode \
  --interactive \
  --config tiny \
  --system-prompt "Tu es un assistant IA amical." \
  --max-length 30 2>&1

echo ""
echo "==================================================================="
echo "Chat mode test completed!"
echo ""
echo "To use interactively with real input, run:"
echo ""
echo "  ./baguettotron generate \\"
echo "    --checkpoint outputs/tiny_correct/ckpt_50/model.pt \\"
echo "    --chat-mode \\"
echo "    --interactive \\"
echo "    --config tiny \\"
echo "    --system-prompt \"Tu es un assistant IA expert.\""
echo ""
echo "==================================================================="

# Cleanup
rm -f /tmp/chat_test_input.txt
