# Windows Port Notes — 3XS Edge RX

**Machine:** 3XS Edge RX · AMD Ryzen 5 9600X · 32GB DDR5 · AMD RX 9070 XT (16GB) · 2TB M.2 · Windows 11  
**Date noted:** 2026-06-24

---

## The AMD GPU situation

PyTorch's GPU backends: CUDA (NVIDIA), ROCm (AMD — Linux only), MPS (Apple Silicon only).  
The RX 9070 XT on Windows has three paths:

| Path | GPU acceleration | Effort |
|------|-----------------|--------|
| WSL2 + Ubuntu + ROCm 6.x | Full, native | High — install WSL2, Ubuntu, ROCm, PyTorch ROCm build |
| Windows + torch-directml | Partial, via DirectX 12 | Low — `pip install torch-directml` |
| Windows CPU only | None | Trivial |

**Recommendation:** Start with CPU-only or DirectML. WSL2 + ROCm is the best long-term path to actually use that GPU, but meaningful setup overhead. The 9600X is fast enough for this project scale — MCTS is the bottleneck, not raw tensor ops. The 16GB VRAM is excellent headroom if/when we scale the model.

---

## Code changes required

### 1. Device detection — `train_chess.py` lines 65–68, `eval_chess.py` (same pattern)

**Current (macOS only):**
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**Cross-platform replacement:**
```python
if torch.cuda.is_available():           # covers CUDA and ROCm (ROCm exposes as "cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

If using DirectML, replace the whole block with:
```python
import torch_directml
device = torch_directml.device()
```

### 2. Run command — no caffeinate on Windows

```bash
# macOS
caffeinate -dims venv/bin/python3 train_chess.py

# Windows (set power plan to High Performance instead)
venv\Scripts\python train_chess.py
```

### 3. Stockfish binary

Download the Windows build from stockfishchess.org. Update the Stockfish path in `eval_chess.py` to point to the `.exe`.

### 4. venv activation

```bash
# macOS
source venv/bin/activate
venv/bin/python3 eval_watcher.py

# Windows
venv\Scripts\activate
venv\Scripts\python eval_watcher.py
```

---

## Files to change

| File | Change needed |
|------|--------------|
| `train_chess.py` | Device detection block (3 lines) |
| `eval_chess.py` | Device detection block + Stockfish path |
| `chessai/` package | No changes — device is passed in, not hardcoded |
| `eval_watcher.py` | No changes — pure Python |

---

## Verification

Once set up, confirm the right device is detected:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# Should print True + ROCm version if WSL2+ROCm is working
```

First line of training output should read `Device: cuda` (ROCm) or `Device: cpu`.
