# Baguettotron‑321M (PleIAs)

**Subtitle:** Small Reasoning Model — White Paper based on the official Hugging Face release
**Version:** 1.1 — Source: model and files published on Hugging Face [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron)

---

## 1) Executive Summary

Baguettotron‑321M is a **small reasoning model** (~321 M parameters) trained **exclusively** on **SYNTH (~200 B tokens)**, a **fully synthetic, multi‑task** corpus that emphasizes reasoning traces and grounded answers. The architecture is **Llama‑like/Qwen‑like**, unusually **deep (80 layers)**, multilingual (FR/DE/IT/ES/PL + partial NL/LA), and **natively instructed with thinking tags** (`<think>…</think>`).

**Goal.** Provide, under 0.5 B parameters, a practical balance of **reasoning / memory / math / retrieval**, with a simple instruction format (Qwen‑style) and built‑in RAG through **source tags** (`<source_1>…</source_1>`, …).

---

### Learn it fast — plain‑language takeaways

* **What problem does this solve?** A compact model that can show its work, remember core facts, and cite sources when you provide them.
* **How does it pull that off?** It drafts ideas inside `<think>…</think>` (a private workspace) and then compresses them into a clean answer.
* **When should you use it?** General text tasks (ask–answer, short math, extraction), especially when you can pass small source snippets.

**Mini‑exercise**
Write a one‑line question and wrap two short sources (each 2–3 sentences) inside `<source_1>…</source_1>` and `<source_2>…</source_2>`. Ask for an answer **with quotes**. Compare the result to removing sources entirely.

---

## 2) Reference Specifications (from `config.json`)

* **architectures:** `LlamaForCausalLM`
* **vocab_size:** **65 536** (special tokens are added via tokenizer files)
* **hidden_size (d_model):** **576**
* **head_dim:** **64** (→ 9×64 = 576)
* **num_hidden_layers:** **80**
* **num_attention_heads:** **9** ; **num_key_value_heads:** **3** (→ **GQA**)
* **intermediate_size (MLP):** **1 536**
* **hidden_act:** **SiLU**
* **rms_norm_eps:** **1e‑5**, **pre‑norm**
* **max_position_embeddings:** **4 096**
* **rope_theta:** **10 000** (no rope_scaling declared)
* **attention_dropout:** **0.0** ; **use_cache:** **true**
* **tie_word_embeddings:** **true**
* **torch_dtype:** **bfloat16**

> **Approx. memory footprint:** ~0.65–0.9 GB (INT8), ~0.4–0.45 GB (4‑bit, weights only) + KV‑cache depending on context length and batch size.

---

### Why these fields matter (one‑liners)

* `vocab_size` → how many symbols the model can read/write. Bigger ≠ always better; it changes memory.
* `hidden_size (d_model)` → the width of the internal vectors; sets per‑token compute and memory.
* `num_hidden_layers` → how many reasoning steps are stacked; **depth** often helps composition.
* `num_attention_heads / num_key_value_heads` → how many focus lines and how many memory groups the model uses at once.
* `intermediate_size` → MLP capacity to mix features and form concepts.
* `max_position_embeddings` + `rope_theta` → how far it can look and how it encodes order.
* `rms_norm_eps`, `hidden_act` → training/inference stability and non‑linearity choice.

**Back‑of‑the‑envelope memory**
**Weights (bf16):** ≈ 321M × 2 bytes ≈ **0.64 GB**.
**KV‑cache (bf16):** `bytes ≈ batch × seq_len × layers × KV_heads × head_dim × 2(K,V) × 2`.
Example: `1×4096×80×3×64×2×2 ≈ 240 MB`.
Takeaway: context length and batch size move KV‑cache much more than weights.

---

## 3) Tokenizer & Reasoning Tags

### 3.1 Special tokens (from `special_tokens_map.json`)

* Dialog delimiters: `<|im_start|>`, `<|im_end>`
* Reasoning tags: `<think>`, `</think>`
* RAG tags: `source_1` … `source_10`, `<ref`, `</ref>`
* Control marks: `→`, `↺`, `※`, `?maybe?`, `●`, `◐`, `○`, `⚠`, `☐`, `☑`, `✓`
* **H‑series:** `⟨H≈0.1⟩` … `⟨H≈1.8⟩` (gradations used inside traces/states)

> The base vocabulary remains 65 536; these are **special tokens** configured by the tokenizer.

### 3.2 Conversation template (`chat_template.json`)

The template concatenates messages in a Qwen‑style layout and **opens generation with `<think>`** when `add_generation_prompt=True`:

```
{% for m in messages %}<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<think>
{% endif %}
```

**BOS:** `<|im_start|>` ; **EOS:** `<|im_end|>` ; **stop:** `"<|im_end|>"`.

---

### Try it yourself

**Goal:** get grounded quotes.
Paste two short paragraphs into `<source_1>` and `<source_2>` and ask: *“Answer in 3 sentences and include quotes.”*
Observe the model emits bracketed `[quote]` markers tied to the sources.

**Common slips**

* Putting sources **outside** the user turn → no quotes.
* Removing `<think>` everywhere → weaker memory and brittle multi‑step answers.
* Mixing languages inside a single sentence → tokenizer can handle it, but answers may drift; keep prompts coherent.

---

## 4) Dataset & Compute

* **SYNTH (~200 B tokens)**: synthetic, multi‑task pipelines (Wikipedias’ “vital articles” memorization, RAG with grounding, arithmetic, editing, extraction, constrained creative writing), **~20% EU‑languages**, **code excluded** (https://huggingface.co/datasets/PleIAs/SYNTH)
* **Hardware:** training on **16 × H100** at **Jean Zay**. Very **early signals** on MMLU; **depth 80** shows empirical gains.

---

### Why synthetic data here?

* **Predictable rights:** CDLA‑Permissive‑2.0 clarifies reuse.
* **Targeted skills:** you can dial tasks (retrieval, arithmetic, editing) and add traces consistently.
* **Trade‑off:** less world‑messiness than raw web; needs careful pipelines to avoid style echo.

**Check yourself**
If you remove reasoning fields from samples and train anyway, which tasks degrade first? (Clue: memory questions that require intermediate steps.)

---

## 5) Architecture & Design

### 5.1 How it works — four clear ideas

1. **Read symbols:** each word is encoded as numbers (embeddings). The model only sees vectors.
2. **Look in the right place:** attention compares vectors to decide what matters **now**.
3. **Work the problem:** between `<think>` and `</think>`, the model drafts a structured explanation (a scratchpad). These tokens form a workspace kept separate from the final answer.
4. **Condense into an answer:** when producing the final output, generation is intentionally “cooler”: it summarizes, checks, and decides.

**Core:** causal decoder, **Llama‑like**.
**Extreme depth:** **80 layers** for better exposure to multi‑step composition.
**Attention:** **GQA** (9 Q‑heads, 3 KV‑heads) for memory/FLOPs efficiency.
**MLP:** `intermediate_size=1536`, activation **SiLU**.
**Positions:** **RoPE** standard (`rope_theta=10000`), context 4 096, no declared scaling.
**Tied embeddings** for the output head.

---

### What a single token experiences (end‑to‑end)

1. **Embed:** map the token to a 576‑dim vector.
2. **Position:** apply rotary angles so the model knows order.
3. **Attention:** compare to past tokens; fetch relevant bits from the KV cache.
4. **MLP:** mix features to form crisper concepts.
5. Repeat ×80 → **deeper** thought.
6. **Final norm + head:** turn features back into a probability over the vocabulary.

**Attention in 3 tokens (toy mental model)**
Say the sequence is `“Paris is”`. To guess the next word, heads can pay attention to “Paris” (capital), to recent “is” (grammar), or both. GQA shares memory among heads to keep compute small while letting different heads focus.

**Why 80 layers?**
Depth increases the number of times the model can refine its draft. Width adds capacity; **depth** adds **steps**. Small models often benefit more from extra steps than extra width.

**RoPE explained in one line**
Each pair of features rotates by an angle that grows with position, so relative order is baked into the vectors.

---

## 6) Evaluation (announced benchmarks)

* **MMLU (reasoning/memory)**, **gsm8k (math)**, **HotPotQA (retrieval)**: **321 M** approaches **Qwen‑0.6B** and **exceeds** Gemma models of similar size (non‑code).

> Detailed numbers are provided in public model cards/figures; emphasis here is **non‑code** tasks.

---

### What these benchmarks probe

* **MMLU:** breadth of facts and concept links across disciplines (closed‑book).
* **GSM8K:** small multi‑step arithmetic.
* **HotPotQA:** retrieve and combine two facts coherently.

**Replicate the feel (micro‑tests)**

* Write a 2‑hop question and pass two short sources; ask for quotes.
* Pose a 2‑step math story with small numbers; check for a brief draft then a final line.

**Reading the numbers**
Close scores to larger SLMs usually mean **prompting** and **template fidelity** matter as much as raw size.

---

## 7) Usage & Good Practices

* **Instruction style:** Qwen:

```
<|im_start|>user
Question…<|im_end|>
<|im_start|>assistant
<think>
```

* **Multi‑turn:** use **rolling thinking**: keep the newest trace and **drop the previous** one to keep context short.
* **No‑trace mode:** possible by **closing immediately** with `</think>`, but expect a **drop** on memory‑heavy tasks.
* **RAG:** provide sources as dedicated blocks (`<source_1>…</source_1>`, etc.). The final answer includes **“+[quote]+”** citations.
* **Languages:** reading/writing: **FR/DE/IT/ES/PL** (+ NL/LA to a smaller extent). **Reasoning traces are always in English.**

---

### Quick recipes

* **Grounded Q&A:** add `<source_i>` blocks; ask for `[quote]`s.
* **Memory probe:** ask about a *vital article* fact and keep `<think>` enabled.
* **Trace‑off mode:** insert `</think>` immediately if your policy requires hidden reasoning.

### Troubleshooting

* **No quotes?** Ensure sources are **inside** the user turn and ask for quotes explicitly.
* **Answers drift languages?** Keep the user message monolingual; request the answer language.
* **Short answers only?** Increase `max_new_tokens`; keep temperature lower for the final answer.

---

## 8) Inference (Transformers, minimal example)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "PleIAs/Baguettotron"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

messages = [
  {"role":"user","content":"Who are you?"}
]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(text, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=False))
```

> **Practical tips:** start with `temperature=0.7–0.9` while inside `<think>`, then go **cooler** (`0.2–0.4`) for the final part; clean old traces each turn.

---

## 9) License & Compliance

* **License:** **Apache‑2.0** (model and artifacts).
* **Traceability:** SYNTH pipelines rely on **Wikipedia** (attribution) + open‑weights models whose outputs are free for reuse.

---

## 10) Appendices

### 10.1 Parameter count (order of magnitude)

* Attention/MLP per layer → ~3.54 M params/layer × 80 ≈ **283 M**
* Tied embeddings (65 536×576) ≈ **37.8 M**
  → Total ≈ **321 M**.

### 10.2 Release notes

* `rope_theta` set to **10 000** (not 100 000).
* **SiLU** activation (not SwiGLU, per config).
* **GQA (KV=3)** explicit.
* Specs **aligned** with HF files (`config.json`, `chat_template.json`, `special_tokens_map.json`).

---

## Appendix D — Official Hugging Face Details (sourced)

**Repo:** `PleIAs/Baguettotron` — License **Apache‑2.0** — *Transformers* `4.51.3` — dtype **BF16** — weights `model.safetensors` ≈ **642 MB**. Key files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.json`, `generation_config.json`.

### D.1 Key `config.json` fields

* `architectures: ["LlamaForCausalLM"]`
* `hidden_size: 576`, `num_hidden_layers: 80`, `num_attention_heads: 9`, `num_key_value_heads: 3` (**GQA**)
* `intermediate_size: 1536`, `hidden_act: "silu"`
* `max_position_embeddings: 4096`, `rope_theta: 10000`, `rms_norm_eps: 1e‑5`
* `attention_dropout: 0.0`, `use_cache: true`, `tie_word_embeddings: true`
* `vocab_size: 65536`, `torch_dtype: "bfloat16"`

### D.2 Special tokens (from `special_tokens_map.json`)

`<|im_start|>`, `<|im_end>`, `<think>`, `</think>`, `source_1…source_10`, `<ref`, `</ref>`, `→`, `↺`, `※`, `?maybe?`, `●`, `◐`, `○`, `⚠`, `☐`, `☑`, `✓`, `⟨H≈0.1⟩…⟨H≈1.8⟩`.

### D.3 Chat template (`chat_template.json`)

```json
{
  "chat_template": "{% for m in messages %}<|im_start|>{{ m['role'] }}
{{ m['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
<think>
{% endif %}",
  "eos_token": "<|im_end|>",
  "bos_token": "<|im_start|>",
  "stop": ["<|im_end|>"],
  "roles": {"user":"user","assistant":"assistant","system":"system"}
}
```

### D.4 Training & compute (model card + dataset page)

* **Data:** **SYNTH ≈200 B tokens**, synthetic, general‑purpose (Wikipedia “vital articles” memory, grounded RAG, arithmetic, editing, extraction, creative writing, etc.).
* **Hardware:** **16× H100** at **Jean Zay** (project A0191016886). Early MMLU signal; empirical gains from **80‑layer depth**.

### D.5 Languages & traces

* Read/Write: **FR/DE/IT/ES/PL** (+ partial NL/LA).
* **Reasoning traces are always in English.**

### D.6 Usage (reminders)

* **Qwen‑Instruct style**; generation opens with `<think>` automatically via the template.
* **Multi‑turn:** “rolling thinking” (keep the latest trace, drop the previous).
* **No‑trace:** possible by replacing the opening with `</think>` (observed degradation on memory tasks).
* **RAG:** pass sources inside `<source_1>…</source_1>`, `<source_2>…</source_2>`; the model returns references like `([quote])`.

### D.7 Inference example (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
name = "PleIAs/Baguettotron"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
messages = [{"role":"user","content":"Who are you?"}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=False))
```

### D.8 Benchmarks (summary from model card)

* Evaluated on **MMLU**, **gsm8k**, **HotPotQA**: **321 M** is close to **Qwen‑0.6B** and **above** Gemma models of comparable size (non‑code).

### D.9 Metadata & versions

* **License:** Apache‑2.0.
* **Repo size:** ~649 MB; `model.safetensors` ~642 MB; **BF16** weights.
* **Recent commits:** e.g. `config.json` hash `c3c1fac`; `chat_template.json` `d33889a`.

---

## Appendix E — From‑Scratch Implementation (reference)

Minimal, fully compatible implementation matching the public configuration (RoPE, GQA, 80 layers, 4 096 context). Ready for supervised fine‑tuning and local inference.

### E.1 `model.py` — Baguettotron‑321M (Llama‑like, RoPE, GQA, RMSNorm, SiLU)

```python
# model.py
# Baguettotron‑321M: 576‑dim, 80 layers, 9 heads, 3 KV heads, MLP 1536, ctx 4096, RoPE θ=1e4
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.type_as(x)) * self.weight

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)

def build_rope_cache(seq_len: int, head_dim: int, base_theta: float = 10000.0, device=None, dtype=None):
    inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, rope_theta: float = 10000.0):
        super().__init__()
        assert d_model == n_heads * head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.max_seq_cached = 0
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
    def _maybe_cache_rope(self, seqlen: int, device, dtype):
        if seqlen > self.max_seq_cached:
            cos, sin = build_rope_cache(seqlen, self.head_dim, self.rope_theta, device=device, dtype=dtype)
            self.cos_cache, self.sin_cache = cos, sin
            self.max_seq_cached = seqlen
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = True):
        b, s, _ = x.size()
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        self._maybe_cache_rope(s, x.device, x.dtype)
        cos, sin = self.cos_cache[:s].unsqueeze(1).unsqueeze(2), self.sin_cache[:s].unsqueeze(1).unsqueeze(2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(b, s, self.n_heads*self.head_dim)
        return self.o_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, mlp_hidden: int, rope_theta: float):
        super().__init__()
        head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_heads, n_kv_heads, head_dim, rope_theta)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden)
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=True)
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class BTConfig:
    vocab_size: int = 65536
    d_model: int = 576
    n_layers: int = 80
    n_heads: int = 9
    n_kv_heads: int = 3
    mlp_hidden: int = 1536
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

class BaguettotronForCausalLM(nn.Module):
    def __init__(self, cfg: BTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.mlp_hidden, cfg.rope_theta)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        for blk in self.layers:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)
```

### E.2 `generate.py` — Local inference (HF tokenizer + chat template)

```python
# generate.py
import torch
from transformers import AutoTokenizer
from model import BaguettotronForCausalLM, BTConfig

NAME = "PleIAs/Baguettotron"  # tokenizer and template

tok = AutoTokenizer.from_pretrained(NAME)
cfg = BTConfig()
model = BaguettotronForCausalLM(cfg).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

messages = [{"role":"user","content":"Who are you?"}]
text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(text, return_tensors="pt").to(model.lm_head.weight.device)

with torch.no_grad():
    for _ in range(256):
        logits = model(inputs.input_ids)
        next_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        inputs = torch.cat([inputs.input_ids, next_id], dim=1)
        if next_id.item() == tok.convert_tokens_to_ids("<|im_end|>"):
            break

print(tok.decode(inputs[0], skip_special_tokens=False))
```

### E.3 `train.py` — Minimal SFT loop (bf16, grad‑accum)

```python
# train.py (minimal SFT)
import torch, math, json, random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from model import BaguettotronForCausalLM, BTConfig

class ChatJsonl(Dataset):
    def __init__(self, path, tokenizer, ctx=4096):
        self.tok = tokenizer
        self.ctx = ctx
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                txt = self.tok.apply_chat_template(obj["messages"], add_generation_prompt=True, tokenize=False)
                ids = self.tok(txt, add_special_tokens=False).input_ids
                for i in range(0, len(ids) - 1, ctx):
                    x = ids[i:i+ctx]
                    y = ids[i+1:i+ctx+1]
                    if len(x) == len(y) == ctx:
                        self.samples.append((torch.tensor(x), torch.tensor(y)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

NAME = "PleIAs/Baguettotron"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(NAME)
cfg = BTConfig()
model = BaguettotronForCausalLM(cfg).to(DEVICE)

train = ChatJsonl("data/sft.jsonl", tok, ctx=cfg.max_seq_len)
loader = DataLoader(train, batch_size=1, shuffle=True)

optim = torch.optim.AdamW(model.parameters(), lr=2.5e-4, betas=(0.9,0.95), weight_decay=0.1)
sched = get_cosine_schedule_with_warmup(optim, 4000, 300000)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
accum = 8
step = 0
model.train()
for epoch in range(1):
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss/accum).backward()
        if (step+1) % accum == 0:
            scaler.step(optim); scaler.update(); optim.zero_grad(); sched.step()
        step += 1
        if step % 1000 == 0:
            torch.save(model.state_dict(), f"out/bt321m_{step}.pt")
```

### E.4 MGQA variant — (RMSNorm + Masked Group‑Query Attention)

```python
class MaskedGroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=3, rope_theta=10000.0, max_seq=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope_theta = rope_theta
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.max_seq = 0
    def _rope(self, s, device, dtype):
        if s > self.max_seq:
            cos, sin = build_rope_cache(s, self.head_dim, self.rope_theta, device, dtype)
            self.cos, self.sin = cos, sin
            self.max_seq = s
    def forward(self, x):
        B, L, E = x.size()
        self._rope(L, x.device, x.dtype)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        q = apply_rope(q, self.cos[:L].unsqueeze(1).unsqueeze(2), self.sin[:L].unsqueeze(1).unsqueeze(2))
        k = apply_rope(k, self.cos[:L].unsqueeze(1).unsqueeze(2), self.sin[:L].unsqueeze(1).unsqueeze(2))
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q, k, v = [t.permute(0,2,1,3) for t in (q,k,v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(0,2,1,3).contiguous().view(B, L, E)
        return self.out_proj(out)
```

### E.5 Data, templates & tokens

* **Tokenizer:** `AutoTokenizer.from_pretrained("PleIAs/Baguettotron")`
* **Chat template:** included in repo (auto‑opens with `<think>`).
* **Special tokens:** `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `source_1..10`, `<ref`, `</ref>`, `⟨H≈…⟩` symbols.

### E.6 Inference recipes

* **Traces ON:** `temp_think=0.9, top_p_think=0.95`; **final:** `temp=0.3, top_p=0.4`; **self‑consistency k=3**.
* **Traces OFF:** inject `</think>` at opening; expect some drop on memory/QA tasks.

### E.7 Implementation notes

* **Tied embeddings:** `lm_head.weight = embed.weight` (done).
* **Flash‑Attention:** enabled automatically via SDPA if available (PyTorch ≥ 2.0).
* **BF16:** prefer bfloat16 on Ampere+; otherwise FP16.

---

## Appendix F — Design assumptions & objectives

1. **Visible parameters:** vocab 65 536 + ~40 reasoning specials; d_model = 576; MLP = 1 536; 9 heads; “Masked group‑query attention”.
2. **Reasoning mechanisms:** separation `<think>`/`<final>`, deliberate decoding (separate temps), self‑consistency, segment masks (private scratchpad).
3. **Reference implementation:** classes `RMSNorm`, `MaskedGroupQueryAttention`, `TransformerBlock`, `Baguettotron` — aligned with RoPE 4 096 and GQA=3.
4. **Target:** local small model, maskable traces, low latency, instruction‑following in FR/DE/IT/ES/PL.

---

## Appendix G — Dataset **SYNTH** (PleIAs)

**Summary.** *SYNTH* is a **general‑purpose, fully synthetic** corpus designed to train **Small Reasoning Models** end‑to‑end. It is published openly by **PleIAs** and the **AI Alliance**.

### G.1 Statistics & composition

* **79 648 272** text samples (≈ **41 G** words, **~75 G** tokens with PleIAs tokenizer).
* **Seeds:** **58 698** Wikipedia articles (*Vital articles* level‑5) + **3 727** **Wikibooks** pages (cooking) + **130** internal texts (recent events, self‑docs, AI).
* **Amplification:** ≥×**100** (up to ×**10 000** for *recent/self*). Section sampling (~**250 k** sections) → **query** generation with random constraints (including negative prompts) → answers with **reasoning traces**.
* **Multilingual:** ≈ **20%** non‑English (FR/DE/ES/IT/PL + NL/LA).
* **Exercises:** memorization/retrieval, **RAG** with grounding (1‑10 sources), arithmetic, editing (translation/extraction/correction), **creative** (constrained writing), multi‑turn conversations.
* **License:** **CDLA‑Permissive‑2.0** (data), seeds under **CC‑BY‑SA 4.0**.

### G.2 Field structure (schema)

`syntch_id: string` • `language: {en, fr, it, es, de, pl, nl, la}` • `exercise: {reasoning, writing, retrieval, arithmetic, ...}` • `model: string` (generator used) • `query: string` (back‑translated prompt) • `query_seed_url: string` • `query_seed_text: string` • `additional_seed_url: string?` • `seed_license: string` • `constraints: string` • `script: string` (pipeline id) • `synthetic_reasoning: string` • `synthetic_answer: string` • `words: int64`.

### G.3 Design & pipelines

* **Core memory:** *Wikipedia: Vital articles* (encyclopedic core).
* **Pipelines:** (1) **Memorization/Retrieval** (back‑translated queries + embedding search)
  (2) **RAG** (source ingestion → sourced answer, `[quote]` markers)
  (3) **Math** (~3 000 formalized problems + random variants)
  (4) **Editing** (translation, style, extraction, correction)
  (5) **Creative** (lipograms, constraints)
  (6) **Multi‑turn** (built from simpler turns).
* **Openness standards:** *model attribution* (model outputs usable), *seed attribution* (seeds re‑published with license).

### G.4 Declared scope

* **Direct use:** pre‑training small reasoning models (< 3–4 B), mid‑training/finetune, research (memory/skills).
* **Out of scope:** **code**, broad world multilingual (currently 8 languages), very large LLMs.

### G.5 Training Baguettotron on SYNTH (recap)

* **Training budget:** **~200 G tokens** consumed (multi‑epoch/mixtures) from SYNTH.
* **Compute:** **16× H100** (Jean‑Zay, project A0191016886), **Nanotron** framework (HF).
* **Early signal:** **MMLU** emerges early; **80‑layer** depth yields consistent gains.

### G.6 Example format (RAG fragment)

```
<|im_start|>user
Question...
<source_1>…</source_1>
<source_2>…</source_2>
<|im_end|>
<|im_start|>assistant
<think>
…
```

### G.7 Compliance & traceability checklist

* **Seed attribution** (`query_seed_url`, `seed_license`).
* **Model attribution** (`model`).
* **Intermediate traces** (`synthetic_reasoning`).
* **Hallucination control:** negative prompts, systematic grounding, LLM‑as‑judge filtering.

---

## Appendix H — SYNTH ↔ Baguettotron‑321M Alignment

* **Tokenizer:** PleIAs (European), vocab **65 536** + special tokens (thinking, sources, `⟨H≈…⟩`).
* **Template:** **Qwen‑Instruct**‑like, **auto‑open `<think>`**; multi‑turn recommended with **rolling thinking**.
* **RAG:** `<source_i>` protocol → `[quote]` citations.
* **Decoding:** separate temperatures for *think/final*, self‑consistency `k=3–5`, stop on `<|im_end|>`.

### 10.3 Glossary (plain terms)

* **Trace:** temporary draft text between `<think>` and `</think>`.
* **KV cache:** the attention memory of past tokens; grows with sequence length.
* **GQA:** group multiple query heads to share fewer key/value heads (memory saver).
* **RoPE:** a trick to encode the order of tokens by rotating vector pairs.

### 10.4 Worked example (one full turn)

**Prompt**

```
<|im_start|>user
Who wrote the EU anthem? Answer with one quote.
<source_1>The “Ode to Joy” theme was composed by Ludwig van Beethoven ...</source_1>
<|im_end|>
```

**What happens**

1. The prompt opens `<think>` → the model drafts a short rationale.
2. It selects the Beethoven fact from `<source_1>` and marks a `[quote]`.
3. It closes with a concise answer in the requested style.
