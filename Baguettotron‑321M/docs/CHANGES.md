# Baguettotron-321M - Modifications pour correspondre à LlamaForCausalLM

## Résumé

Le code a été mis à jour pour correspondre exactement à l'architecture **LlamaForCausalLM** utilisée par le modèle officiel sur Hugging Face (https://huggingface.co/PleIAs/Baguettotron).

## Architecture finale: ✓ 320.96M paramètres (~321M)

### Modifications principales

#### 1. **tests/test_all.py** - Correction import MGQA
- **Ligne 26**: Suppression de la ligne qui écrasait l'import correct de `MaskedGroupQueryAttention`
- **Résultat**: Test `test_equivalence_with_mgqa_when_weights_match` fonctionne maintenant

#### 2. **mgqa.py** - Alignement format RoPE
- Import de `_apply_rope_inplace` au lieu de `apply_rope`
- Conversion au format **interleaved** (T, Hd) pour correspondre à `model.py`
- Correction signature `_rope_make_cache` (ajout paramètre `dtype`)
- **Résultat**: Compatible avec l'implémentation RoPE de LlamaForCausalLM

#### 3. **model.py** - Architecture complète LlamaForCausalLM

##### a) BTConfig - Nouveaux paramètres
```python
@dataclass
class BTConfig:
    vocab_size: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    mlp_hidden: int = 128
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    tie_embeddings: bool = False      # ← NOUVEAU
    rms_norm_eps: float = 1e-6        # ← NOUVEAU (1e-5 pour l'officiel)
```

##### b) MLP → SwiGLU (CHANGEMENT MAJEUR)
**Avant:**
```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
```

**Après (SwiGLU comme LlamaForCausalLM):**
```python
class FeedForward(nn.Module):
    """
    SwiGLU MLP as used in LlamaForCausalLM.
    Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, d_model: int, hidden: int):
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
```

**Impact:** +70M paramètres (250M → 321M) grâce à la projection supplémentaire

##### c) Embedding tying
```python
class BaguettotronForCausalLM(nn.Module):
    def __init__(self, cfg: BTConfig):
        ...
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie embeddings if requested (weight sharing)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight  # ← NOUVEAU
```

##### d) RMSNorm epsilon configurable
```python
# Dans Block.__init__
self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)  # ← eps ajouté
self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)  # ← eps ajouté

# Dans BaguettotronForCausalLM.__init__
self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)  # ← eps ajouté
```

#### 4. **train.py** - Support nouveaux paramètres
```python
base = {
    ...
    "tie_embeddings": bool(mcfg.get("tie_embeddings", False)),  # ← NOUVEAU
    "rms_norm_eps": float(mcfg.get("rms_norm_eps", 1e-6)),      # ← NOUVEAU
}
```

#### 5. **configs/model_official_baguettotron_321m.yaml** - Configuration officielle
**Correction pour correspondre exactement au config.json HuggingFace:**
```yaml
model:
  vocab_size: 65536          # ← Corrigé (était 151552)
  d_model: 576               # ← Corrigé (était 768)
  n_layers: 80               # ✓ Correct
  n_heads: 9                 # ← Corrigé (était 12)
  n_kv_heads: 3              # ← Corrigé (était 4)
  mlp_hidden: 1536           # ← Corrigé (était 2048)
  max_seq_len: 4096          # ← Corrigé (était 8192)
  rope_theta: 10000.0        # ✓ Correct
  rms_norm_eps: 1.0e-05      # ✓ Correct
  tie_embeddings: true       # ← AJOUTÉ
  sliding_window: 0          # ← Corrigé (était 4096)
```

## Comparaison avec HuggingFace config.json

| Paramètre HF | Notre Config | Valeur | ✓ |
|--------------|--------------|--------|---|
| `architectures` | `BaguettotronForCausalLM` | LlamaForCausalLM compatible | ✓ |
| `vocab_size` | `vocab_size` | 65536 | ✓ |
| `hidden_size` | `d_model` | 576 | ✓ |
| `num_hidden_layers` | `n_layers` | 80 | ✓ |
| `num_attention_heads` | `n_heads` | 9 | ✓ |
| `num_key_value_heads` | `n_kv_heads` | 3 | ✓ |
| `intermediate_size` | `mlp_hidden` | 1536 | ✓ |
| `max_position_embeddings` | `max_seq_len` | 4096 | ✓ |
| `rope_theta` | `rope_theta` | 10000.0 | ✓ |
| `rms_norm_eps` | `rms_norm_eps` | 1e-05 | ✓ |
| `tie_word_embeddings` | `tie_embeddings` | true | ✓ |
| `hidden_act` | SiLU (in SwiGLU) | silu | ✓ |

## Structure architecturale finale

```
BaguettotronForCausalLM (320.96M params)
├── embed: Embedding(65536, 576)                    [37.75M params]
├── layers: ModuleList [×80]
│   └── Block
│       ├── norm1: RMSNorm(576, eps=1e-05)          [576 params]
│       ├── attn: GQAttention (GQA with RoPE)
│       │   ├── q_proj: Linear(576, 576)            [331,776 params]
│       │   ├── k_proj: Linear(576, 192)            [110,592 params]
│       │   ├── v_proj: Linear(576, 192)            [110,592 params]
│       │   └── o_proj: Linear(576, 576)            [331,776 params]
│       ├── norm2: RMSNorm(576, eps=1e-05)          [576 params]
│       └── ffn: FeedForward (SwiGLU)
│           ├── gate_proj: Linear(576, 1536)        [884,736 params]
│           ├── up_proj: Linear(576, 1536)          [884,736 params]
│           └── down_proj: Linear(1536, 576)        [884,736 params]
├── final_norm: RMSNorm(576, eps=1e-05)             [576 params]
└── lm_head: Linear(576, 65536) [tied with embed]  [0 params - shared]
```

### Calcul détaillé par couche (×80):
- **Attention**: 331,776 + 110,592 + 110,592 + 331,776 = 884,736 params
- **SwiGLU MLP**: 884,736 + 884,736 + 884,736 = 2,654,208 params
- **Norms**: 576 + 576 = 1,152 params
- **Total par couche**: 3,540,096 params
- **80 couches**: 283,207,680 params

### Total général:
- Embeddings: 37,748,736 params
- 80 couches: 283,207,680 params
- Final norm: 576 params
- **TOTAL: 320,956,992 params (~321M)** ✓

## Tests

**13/13 tests réussis ✓**

```bash
$ pytest -v
tests/test_all.py::test_shapes_and_head_dims PASSED
tests/test_all.py::test_rope_cache_grows_monotonically PASSED
tests/test_all.py::test_equivalence_with_mgqa_when_weights_match PASSED
tests/test_all.py::test_causal_mask_no_future_leak PASSED
tests/test_all.py::test_forward_backward_gradients PASSED
tests/test_all.py::test_save_load_roundtrip_determinism PASSED
tests/test_all.py::test_generation_loop_grows_sequence PASSED
tests/test_all.py::test_rmsnorm_basic_properties PASSED
tests/test_all.py::test_quick_train_step_runs PASSED
tests/test_all.py::test_train_from_yaml_offline PASSED
tests/test_all.py::test_model_moves_to_cuda_if_available PASSED
tests/test_all.py::test_chat_prompt_renderer_minimal PASSED
tests/test_all.py::test_greedy_generate_ids_offline_behavior PASSED
```

## Validation

✓ Configuration correspond exactement à `config.json` HuggingFace
✓ Architecture compatible avec `LlamaForCausalLM`
✓ MLP utilise SwiGLU (gate_proj, up_proj, down_proj)
✓ Pre-normalization (RMSNorm avant attention/MLP)
✓ Embeddings partagés (tie_word_embeddings=true)
✓ RoPE avec theta=10000
✓ Grouped-Query Attention (9 heads, 3 KV heads)
✓ Nombre de paramètres: **320.96M (~321M)** ✓
✓ Tous les tests passent

## Fichiers créés pour validation

- `test_config.py` - Vérifie la config officielle
- `count_params.py` - Compte détaillé des paramètres
- `verify_llama_compatibility.py` - Comparaison complète avec LlamaForCausalLM

## Utilisation

```bash
# Test avec config officielle
python test_config.py

# Vérification compatibilité LlamaForCausalLM
python verify_llama_compatibility.py

# Tests complets
pytest -v
```

## Références

- HuggingFace: https://huggingface.co/PleIAs/Baguettotron
- Config: https://huggingface.co/PleIAs/Baguettotron/blob/main/config.json
- Paper: whitepaper.md
