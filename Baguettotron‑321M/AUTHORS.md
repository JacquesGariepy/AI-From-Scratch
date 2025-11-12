# Authors & Contributors

## Author

**Jacques Gariépy**
- GitHub: [@JacquesGariepy](https://github.com/JacquesGariepy)
- Role: Sole author - Independent reverse engineering
- Contribution: From-scratch implementation through reverse engineering of publicly available model architecture

## ⚠️ Critical Disclaimer

**THIS PROJECT HAS ZERO AFFILIATION WITH PleIAs OR THE BAGUETTOTRON TEAM**

### What This Means:

❌ **NOT AFFILIATED**: This code has NO AFFILIATION with PleIAs or the Baguettotron team in ANY WAY
❌ **NOT official**: This code is NOT affiliated with, endorsed by, or approved by the Baguettotron team
❌ **ZERO collaboration**: The Baguettotron/PleIAs team has NOT shared, provided, assisted, or communicated with this project
❌ **NO inside access**: No proprietary code, documentation, or insider information was used
❌ **NO relationship**: No partnership, sponsorship, or relationship with PleIAs exists
❌ **NOT representing**: This project does NOT represent PleIAs or the Baguettotron team

✅ **Reverse engineered**: Architecture reconstructed from publicly available information ONLY
✅ **Educational purpose**: Written from scratch for learning and educational purposes ONLY
✅ **Public sources only**: Based solely on HuggingFace model structure and config.json
✅ **Independent work**: 100% independent implementation by Jacques Gariépy

## Analyzed Model

**PleIAs/Baguettotron-321M** (for reference only)
- Official Model: https://huggingface.co/PleIAs/Baguettotron
- Dataset: https://huggingface.co/datasets/PleIAs/SYNTH
- License: Apache 2.0
- Status: Open-weight (weights available, source code not released)

This implementation is a **reverse engineering attempt** to understand and recreate the architecture from publicly available information.

## Architecture Inspiration

The architecture follows **LlamaForCausalLM** design (Meta):
- Grouped Query Attention (GQA)
- SwiGLU activation
- RoPE positional embeddings
- RMSNorm pre-normalization

## Legal & Ethical Notes

- This implementation respects the Apache 2.0 license of the original model
- No proprietary code or confidential information was accessed or used
- All code written independently from publicly available architectural information
- This is an educational exercise in understanding transformer architectures
- For production use, please use the official model: https://huggingface.co/PleIAs/Baguettotron

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to:
https://github.com/JacquesGariepy/AI-From-Scratch

## License

Apache License 2.0 - See LICENSE file for details
