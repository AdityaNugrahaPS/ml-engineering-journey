# Rencana Implementasi: Fase 0 (Refresh) + Project 1 IndoGPT-mini

> **Untuk pelaksana:** rencana ini dikerjakan oleh **kamu sendiri** sebagai latihan, dengan Claude sebagai mentor/reviewer. Centang setiap langkah (`- [ ]`) saat selesai. Spec induk: `docs/superpowers/specs/2026-10-16-ml-engineer-roadmap-design.md`.

**Goal:** Dalam 10 minggu (1 Nov 2026 – 9 Jan 2027), menyegarkan fondasi ML dan menyelesaikan repo `indogpt-mini`, berisi autograd NumPy + MLP, tokenizer BPE Bahasa Indonesia, dan GPT kecil yang dilatih di RTX 3050. Paralel dengan itu, menyelesaikan 60 soal LeetCode tahap L1.

**Architecture:** Satu repo Python (layout `src/`) dengan lima modul yang berdiri sendiri: `autograd` (NumPy murni), `tokenizer` (BPE byte-level), `model` (GPT di PyTorch), `data` (persiapan korpus + loader memmap), dan `train` (loop training + jadwal LR). Setiap modul dikerjakan dengan TDD: salin test dulu, pastikan gagal, implementasikan, pastikan lulus, lalu commit.

**Tech Stack:** Python 3.12, NumPy, PyTorch (CUDA), `regex`, Hugging Face `datasets` & `huggingface_hub`, Weights & Biases, Gradio, pytest, ruff, GitHub Actions, Docker.

---

## Cara memakai rencana ini

- **Ikon:** 📘 Belajar · 🛠️ Project · 🧩 LeetCode · ✅ Checkpoint
- **🧠 = latihan inti.** Test lengkap dan kerangka kode disediakan, tapi **implementasinya kamu tulis sendiri**. Kunci jawaban sengaja tidak disertakan, karena portofolio harus hasil tanganmu sendiri: pewawancara akan menanyakan detailnya. Yang disediakan: stub, petunjuk matematis, dan referensi.
- **Aturan macet:** setelah 45 menit, baca petunjuk atau referensi. Setelah 90 menit, tanya Claude dan minta **petunjuk, bukan solusi**.
- Kode infrastruktur (config, CI, skrip CLI, aplikasi demo) diberikan lengkap.
- **Anggaran per minggu:** ±14 jam belajar + project, ±6 jam LeetCode (1 jam/hari Senin–Jumat + 1 jam review hari Minggu). Mulai Januari 2027 kamu punya 25–30 jam/minggu; kelebihannya adalah **jam cadangan** (lihat bagian berikut), bukan untuk menambah task.
- **Shell:** PowerShell. Semua perintah dijalankan dari folder `C:\AI or ML Engineering\indogpt-mini`.
- **`python` di rencana ini = `.venv\Scripts\python`.** Kamu bisa mengaktifkan venv dengan `.venv\Scripts\Activate.ps1`. Kalau PowerShell menolak karena *execution policy*, tulis saja path lengkapnya, misal `.venv\Scripts\python -m pytest`.

### Kalau tertinggal jadwal

Prioritas yang **tidak boleh** dikorbankan, dari yang paling penting:
1. Test modul inti lulus
2. Soal LeetCode non-Hard
3. Materi 📘 inti

Yang **boleh ditunda:** soal bertanda ⭐ (Hard), Task 12 (baseline bigram), dan Task 9 (Docker, boleh digeser ke minggu 10).

**Jam cadangan:** mulai Januari 2027 kamu punya 5–10 jam ekstra per minggu. Pakai untuk menyelesaikan task yang tertunda. Minggu 8–9 (20 Des – 2 Jan) bertepatan dengan Natal dan Tahun Baru; kalau jam belajar terpotong, sisa task-nya boleh dibawa ke jam cadangan Januari, asalkan P1 selesai sebelum Fase 2 dimulai (10 Jan 2027).

---

## Ringkasan 10 minggu

| Mgg | Tanggal | 📘 Fokus belajar | 🛠️ Task | 🧩 LeetCode |
|---|---|---|---|---|
| 1 | 1–7 Nov | Shell, Git, NumPy, pandas | 1–4: setup, repo, CI, cek GPU | Arrays & Hashing |
| 2 | 8–14 Nov | Aljabar linear, kalkulus, micrograd | 5: autograd skalar 🧠 | A&H, Two Pointers |
| 3 | 15–21 Nov | Probabilitas, statistik, ML klasik | 6: autograd tensor 🧠 | Two Pointers, Sliding Window |
| 4 | 22–28 Nov | Python engineering, Docker | 7–10: MLP, digits, Docker, README 1a | Sliding Window, Stack |
| 5 | 29 Nov – 5 Des | PyTorch basics, makemore 1–2 | 11–12: korpus, baseline bigram | Stack, Binary Search |
| 6 | 6–12 Des | makemore 3–4, tokenizer | 13–15: BPE 🧠 + optimasi | Binary Search, Linked List |
| 7 | 13–19 Des | makemore 5, *Let's build GPT* | 16: model GPT 🧠 | Linked List |
| 8 | 20–26 Des | Mixed precision, AdamW, LR schedule | 17–20: encode, loader, training loop 🧠 | Linked List, Trees |
| 9 | 27 Des – 2 Jan | Resep training, scaling laws | 21: ablation 3 ukuran model | Trees |
| 10 | 3–9 Jan 2027 | HF Hub, Gradio | 22–26: sampling, HF, demo, README | Trees |

---

## Struktur file (target akhir)

```
indogpt-mini/
├── .github/workflows/ci.yml        # CI: ruff + pytest
├── .gitignore
├── Dockerfile                      # menjalankan test di container
├── LICENSE
├── MODEL_CARD.md                   # diunggah sebagai README model di HF Hub
├── README.md
├── pyproject.toml
├── requirements-lock.txt           # versi dependensi terkunci (Task 25)
├── artifacts/tokenizer.json        # tokenizer terlatih (di-commit, kecil)
├── scripts/
│   ├── check_gpu.py                # cek CUDA + benchmark matmul
│   ├── train_digits.py             # MLP NumPy pada dataset digits
│   ├── baseline_bigram.py          # bits-per-byte baseline
│   ├── train_tokenizer.py          # latih BPE + laporan kompresi
│   ├── sample.py                   # generate teks dari checkpoint
│   ├── push_to_hub.py              # unggah ke Hugging Face Hub
│   └── reproduce.sh                # seluruh pipeline, satu perintah
├── space/
│   ├── app.py                      # demo Gradio (HF Spaces)
│   └── requirements.txt
├── src/indogpt/
│   ├── __init__.py
│   ├── baseline.py                 # bigram byte-level 🧠
│   ├── autograd/
│   │   ├── __init__.py
│   │   ├── scalar.py               # Value (autograd skalar) 🧠
│   │   ├── tensor.py               # Tensor (autograd NumPy) 🧠
│   │   └── nn.py                   # Linear, MLP, SGD 🧠
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── bpe.py                  # BPETokenizer 🧠
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py               # GPTConfig + PRESETS
│   │   └── gpt.py                  # GPT 🧠
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_text.py         # unduh + bersihkan Wikipedia ID
│   │   ├── encode.py               # teks -> train.bin / val.bin
│   │   └── loader.py               # get_batch 🧠
│   └── train/
│       ├── __init__.py
│       ├── lr.py                   # get_lr 🧠
│       └── train.py                # loop training 🧠
└── tests/
    ├── test_smoke.py
    ├── test_scalar.py
    ├── test_tensor.py
    ├── test_nn.py
    ├── test_prepare_text.py
    ├── test_baseline.py
    ├── test_bpe.py
    ├── test_gpt.py
    ├── test_loader.py
    ├── test_lr.py
    └── test_train.py
```

---

## Daftar LeetCode L1 (60 soal, basis NeetCode 150)

Tanda ⭐ = Hard, **opsional**. Kalau waktu tidak cukup, tunda ke tahap L4 (Jun–Agu 2027).

| Mgg | Soal |
|---|---|
| 1 | 217 Contains Duplicate · 242 Valid Anagram · 1 Two Sum · 49 Group Anagrams · 347 Top K Frequent Elements · 238 Product of Array Except Self |
| 2 | 271 Encode and Decode Strings (premium di LeetCode; kerjakan versi gratis di neetcode.io) · 36 Valid Sudoku · 128 Longest Consecutive Sequence · 125 Valid Palindrome · 167 Two Sum II · 15 3Sum |
| 3 | 11 Container With Most Water · ⭐42 Trapping Rain Water · 121 Best Time to Buy and Sell Stock · 3 Longest Substring Without Repeating Characters · 424 Longest Repeating Character Replacement · 567 Permutation in String |
| 4 | ⭐76 Minimum Window Substring · ⭐239 Sliding Window Maximum · 20 Valid Parentheses · 155 Min Stack · 150 Evaluate Reverse Polish Notation · 22 Generate Parentheses |
| 5 | 739 Daily Temperatures · 853 Car Fleet · ⭐84 Largest Rectangle in Histogram · 704 Binary Search · 74 Search a 2D Matrix · 875 Koko Eating Bananas |
| 6 | 153 Find Minimum in Rotated Sorted Array · 33 Search in Rotated Sorted Array · 981 Time Based Key-Value Store · ⭐4 Median of Two Sorted Arrays · 206 Reverse Linked List · 21 Merge Two Sorted Lists |
| 7 | 143 Reorder List · 19 Remove Nth Node From End of List · 138 Copy List with Random Pointer · 2 Add Two Numbers · 141 Linked List Cycle · 287 Find the Duplicate Number |
| 8 | 146 LRU Cache · ⭐23 Merge k Sorted Lists · ⭐25 Reverse Nodes in k-Group · 226 Invert Binary Tree · 104 Maximum Depth of Binary Tree · 543 Diameter of Binary Tree |
| 9 | 110 Balanced Binary Tree · 100 Same Tree · 572 Subtree of Another Tree · 235 Lowest Common Ancestor of a BST · 102 Binary Tree Level Order Traversal · 199 Binary Tree Right Side View |
| 10 | 1448 Count Good Nodes in Binary Tree · 98 Validate Binary Search Tree · 230 Kth Smallest Element in a BST · 105 Construct Binary Tree from Preorder and Inorder · ⭐124 Binary Tree Maximum Path Sum · ⭐297 Serialize and Deserialize Binary Tree |

**Metode per soal:**
1. Timebox 25–30 menit. Kalau buntu, baca hint lalu solusi.
2. Kerjakan ulang **tanpa melihat** setelah 1, 3, dan 7 hari.
3. Catat di spreadsheet dengan kolom: `No | Soal | Pola | Kesalahanku | Review berikutnya`.
4. Selalu tulis kompleksitas waktu & ruang, dan jelaskan solusinya dengan suara keras.

---

# MINGGU 1 (1–7 Nov): Setup & refresh NumPy/pandas

### 📘 Belajar (±6 jam)
- [ ] MIT *Missing Semester* (missing.csail.mit.edu/2020): Lecture 1 *Course overview + the shell*, Lecture 5 *Command-line Environment*, Lecture 6 *Version Control (Git)*
- [ ] Dokumentasi NumPy: *NumPy: the absolute basics for beginners* + halaman *Broadcasting*
- [ ] Latihan NumPy di notebook coretan, tanpa loop Python:
  1. matriks jarak Euclidean berpasangan `(N, D) x (M, D) -> (N, M)`
  2. softmax per baris yang stabil numerik
  3. one-hot encoding dengan fancy indexing

  Cocokkan hasilnya dengan versi loop.
- [ ] *Python for Data Analysis* edisi 3 (gratis di wesmckinney.com/book): bab 5 (pandas), 8 (join/merge), 10 (groupby)

### 🛠️ Task 1: Instal tools

**Files:** tidak ada (setup mesin)

- [ ] **Step 1: Instal Git, VS Code, dan uv**

```powershell
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudioCode -e
winget install --id astral-sh.uv -e
```

Tutup lalu buka ulang terminal. Kemudian konfigurasi Git:

```powershell
git config --global user.name "Nama Lengkap Kamu"
git config --global user.email "email-github-kamu@contoh.com"
```

- [ ] **Step 2: Cek driver NVIDIA**

Run: `nvidia-smi`
Expected: tabel yang menampilkan `NVIDIA GeForce RTX 3050 6GB Laptop GPU` dan versi CUDA driver. Kalau perintah tidak dikenal, instal driver terbaru dari nvidia.com, lalu restart.

### 🛠️ Task 2: Kerangka repo + smoke test

**Files:**
- Create: `indogpt-mini/pyproject.toml`
- Create: `indogpt-mini/.gitignore`
- Create: `indogpt-mini/LICENSE`
- Create: `indogpt-mini/README.md`
- Create: `indogpt-mini/src/indogpt/__init__.py`
- Test: `indogpt-mini/tests/test_smoke.py`

- [ ] **Step 1: Buat folder dan venv**

```powershell
mkdir "C:\AI or ML Engineering\indogpt-mini"
cd "C:\AI or ML Engineering\indogpt-mini"
uv venv --python 3.12
```

- [ ] **Step 2: Tulis `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "indogpt-mini"
version = "0.1.0"
description = "GPT kecil Bahasa Indonesia yang dibangun dari nol: autograd, tokenizer BPE, Transformer."
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "regex>=2024.4"]

[project.optional-dependencies]
train = ["torch>=2.4", "datasets>=2.20", "wandb>=0.17"]
demo = ["gradio>=4.0", "huggingface_hub>=0.34"]
dev = ["pytest>=8.0", "ruff>=0.6", "scikit-learn>=1.4"]

[tool.hatch.build.targets.wheel]
packages = ["src/indogpt"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
# Dikunci eksplisit: versi ruff baru memakai rule default yang berbeda, dan tanpa ini CI bisa tiba-tiba merah.
select = ["E4", "E7", "E9", "F"]
```

- [ ] **Step 3: Tulis `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
data/
runs/
wandb/
*.pt
```

- [ ] **Step 4: Tulis `README.md` sementara**

```markdown
# IndoGPT-mini

GPT kecil untuk Bahasa Indonesia yang dibangun dari nol. 🚧 Sedang dikerjakan.
```

- [ ] **Step 5: Tulis `LICENSE`.** Salin teks lisensi MIT dari choosealicense.com/licenses/mit, lalu isi tahun `2026` dan nama lengkapmu.

- [ ] **Step 6: Tulis test yang gagal, `tests/test_smoke.py`**

```python
import indogpt


def test_package_imports():
    assert indogpt.__version__ == "0.1.0"
```

- [ ] **Step 7: Buat file paket kosong lalu instal dependensi.** Paket harus sudah ada sebelum instalasi editable. Instal PyTorch CUDA lebih dulu, baru paket lainnya.

```powershell
New-Item -ItemType Directory -Force src\indogpt
New-Item -ItemType File src\indogpt\__init__.py
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[train,demo,dev]"
```

Catatan: cek perintah instalasi terbaru di pytorch.org/get-started/locally (pilih Windows, Pip, CUDA). Kalau tag CUDA di sana berbeda dari `cu128`, pakai tag dari situs itu.

- [ ] **Step 8: Jalankan test untuk memastikan gagal**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL dengan `AttributeError: module 'indogpt' has no attribute '__version__'`

- [ ] **Step 9: Isi `src/indogpt/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 10: Jalankan test.** Instalasi editable langsung membaca perubahan, jadi tidak perlu instal ulang.

Run: `python -m pytest -v`
Expected: `1 passed`

- [ ] **Step 11: Commit pertama dan push ke GitHub**

```powershell
git init -b main
git add .
git commit -m "chore: kerangka repo indogpt-mini"
```

Buat repo **public** kosong bernama `indogpt-mini` di github.com/new (jangan centang README atau license). Lalu:

```powershell
git remote add origin https://github.com/USERNAME_GITHUB_KAMU/indogpt-mini.git
git push -u origin main
```

### 🛠️ Task 3: CI GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Tulis workflow**

```yaml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q
```

- [ ] **Step 2: Cek ruff secara lokal**

Run: `python -m ruff check .`
Expected: `All checks passed!`

- [ ] **Step 3: Commit + push, lalu cek tab Actions di GitHub**

```powershell
git add .github/workflows/ci.yml
git commit -m "ci: ruff + pytest di GitHub Actions"
git push
```

Expected: run `ci` berwarna hijau ✅ di tab Actions.

### 🛠️ Task 4: Skrip cek GPU

**Files:**
- Create: `scripts/check_gpu.py`

- [ ] **Step 1: Tulis skrip**

```python
"""Cek apakah PyTorch melihat GPU, lalu bandingkan kecepatan matmul CPU vs GPU."""
import time

import torch


def bench(device: str, n: int = 4096, reps: int = 10) -> float:
    """Kembalikan throughput matmul fp32 dalam TFLOPS."""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(2):  # pemanasan
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return 2 * n**3 / dt / 1e12


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    print(f"CUDA tersedia: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 didukung: {torch.cuda.is_bf16_supported()}")
        print(f"GPU fp32: {bench('cuda'):.2f} TFLOPS")
    print(f"CPU fp32: {bench('cpu', n=2048, reps=3):.3f} TFLOPS")
```

- [ ] **Step 2: Jalankan**

Run: `python scripts/check_gpu.py`
Expected: `CUDA tersedia: True`, nama GPU RTX 3050, `bf16 didukung: True`, dan throughput GPU jauh di atas CPU (biasanya lebih dari 10x). Kalau CUDA `False`, kemungkinan torch versi CPU yang terinstal: ulangi Task 2 Step 7.

- [ ] **Step 3: Commit**

```powershell
git add scripts/check_gpu.py
git commit -m "feat: skrip cek GPU dan benchmark matmul"
```

### ✅ Checkpoint Minggu 1
- [ ] `torch.cuda.is_available()` = True
- [ ] CI hijau di GitHub
- [ ] 6 soal LeetCode minggu 1 selesai dan tercatat di spreadsheet

---

# MINGGU 2 (8–14 Nov): Aljabar linear, kalkulus, autograd skalar

### 📘 Belajar (±8 jam)
- [ ] 3Blue1Brown *Essence of Linear Algebra*: episode 1–9 dan 13–14 (change of basis, eigen)
- [ ] *Mathematics for Machine Learning* (mml-book.github.io, PDF gratis), Bab 5 *Vector Calculus*, subbab 5.1–5.6 (5.6 = backpropagation & automatic differentiation)
- [ ] Karpathy, *The spelled-out intro to neural networks and backpropagation: building micrograd*
- [ ] Latihan di kertas, simpan fotonya di catatan:
  1. turunan sigmoid dan tanh
  2. gradien MSE terhadap `w` dan `b` untuk regresi linear
  3. gradien softmax + cross-entropy terhadap logits; tunjukkan hasilnya `p − y`

### 🛠️ Task 5: 🧠 Autograd skalar (`Value`)

**Files:**
- Create: `src/indogpt/autograd/__init__.py` (kosong)
- Create: `src/indogpt/autograd/scalar.py`
- Test: `tests/test_scalar.py`

- [ ] **Step 1: Tulis test, `tests/test_scalar.py`**

```python
import math

import pytest

from indogpt.autograd.scalar import Value

torch = pytest.importorskip("torch")


def test_forward_arithmetic():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a - b / 2 + a**2
    expected = 2.0 * -3.0 + 2.0 - (-3.0 / 2) + 2.0**2
    assert c.data == pytest.approx(expected)


def test_simple_gradients():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a  # dc/da = b + 1 = -2, dc/db = a = 2
    c.backward()
    assert a.grad == pytest.approx(-2.0)
    assert b.grad == pytest.approx(2.0)


def test_node_used_twice_accumulates_grad():
    a = Value(3.0)
    b = a + a  # db/da = 2
    b.backward()
    assert a.grad == pytest.approx(2.0)


def test_tanh_and_exp_gradients():
    x = Value(0.5)
    y = x.tanh() * x.exp()
    y.backward()
    t, e = math.tanh(0.5), math.exp(0.5)
    assert y.data == pytest.approx(t * e)
    assert x.grad == pytest.approx((1 - t**2) * e + t * e)


def test_matches_pytorch_on_complex_expression():
    def expr(a, b, relu):
        c = a * b + b**3
        c = c + c * 2 + relu(b - a)
        d = (a + b) * c - a / 2
        e = relu(d) + d**2
        return e / 4.0 + 1.0 / (e + 10.0)

    a, b = Value(-2.0), Value(1.5)
    out = expr(a, b, lambda v: v.relu())
    out.backward()

    at = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    out_t = expr(at, bt, torch.relu)
    out_t.backward()

    assert out.data == pytest.approx(out_t.item())
    assert a.grad == pytest.approx(at.grad.item())
    assert b.grad == pytest.approx(bt.grad.item())
```

- [ ] **Step 2: Tulis stub, `src/indogpt/autograd/scalar.py`**

```python
"""Autograd skalar ala micrograd. Latihan: implementasikan method yang masih NotImplementedError."""
from __future__ import annotations


def _noop() -> None:
    pass


class Value:
    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = _noop
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # ---- operasi dasar: LATIHAN ----
    def __add__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __mul__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __pow__(self, k: float) -> Value:
        raise NotImplementedError

    def relu(self) -> Value:
        raise NotImplementedError

    def tanh(self) -> Value:
        raise NotImplementedError

    def exp(self) -> Value:
        raise NotImplementedError

    def backward(self) -> None:
        raise NotImplementedError

    # ---- operasi turunan: sudah lengkap, dibangun dari operasi dasar ----
    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: float) -> Value:
        return other * self**-1
```

- [ ] **Step 3: Jalankan test untuk memastikan gagal**

Run: `python -m pytest tests/test_scalar.py -v`
Expected: 5 FAILED dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan.** Petunjuk:
  - Bungkus angka biasa dulu: `other = other if isinstance(other, Value) else Value(other)`.
  - Setiap operasi membuat `out = Value(hasil, (self, other), "op")`, lalu memasang fungsi `_backward` milik `out` yang **menambahkan** (`+=`) gradien lokal × `out.grad` ke anak-anaknya. Kenapa `+=` dan bukan `=`? Lihat `test_node_used_twice_accumulates_grad`.
  - Tabel turunan lokal: `a+b → (1, 1)` · `a*b → (b, a)` · `a**k → k·a**(k-1)` · `relu → 1 jika out > 0, selain itu 0` · `tanh → 1 − tanh²` · `exp → exp(a)` (pakai `math.tanh` / `math.exp`, jadi tambahkan `import math`).
  - `backward()`: buat urutan topologis dengan DFS lewat `_prev`, set `self.grad = 1.0`, lalu panggil `_backward()` setiap node dalam urutan **terbalik**.
  - Referensi: video micrograd, bagian *manual backprop* sampai *topological sort*.

- [ ] **Step 5: Jalankan test**

Run: `python -m pytest tests/test_scalar.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```powershell
git add src/indogpt/autograd tests/test_scalar.py
git commit -m "feat: autograd skalar Value dengan verifikasi gradien vs PyTorch"
```

### ✅ Checkpoint Minggu 2
- [ ] Bisa menjelaskan tanpa catatan kenapa backprop butuh urutan topologis
- [ ] Tiga turunan latihan kertas sudah benar (cocokkan dengan MML Bab 5)

---

# MINGGU 3 (15–21 Nov): Probabilitas, statistik, ML klasik, autograd tensor

### 📘 Belajar (±7 jam)
- [ ] MML Bab 6 subbab 6.1–6.5 (probabilitas, aturan Bayes, statistik ringkasan, Gaussian) dan subbab 8.3 *Parameter Estimation* (MLE)
- [ ] StatQuest: *Maximum Likelihood*, *Bayes' Theorem*, *p-values*, *Bias and Variance*, *Regularization (Ridge, Lasso)*, *Cross Validation*, *ROC and AUC*
- [ ] Géron *Hands-On Machine Learning*: Bab 2–4 (baca cepat sebagai refresh)
- [ ] Tulis masing-masing 1 paragraf di catatan:
  1. kenapa meminimalkan MSE sama dengan MLE di bawah noise Gaussian
  2. kenapa cross-entropy sama dengan MLE untuk distribusi kategorikal

### 🛠️ Task 6: 🧠 Autograd tensor NumPy

**Files:**
- Create: `src/indogpt/autograd/tensor.py`
- Test: `tests/test_tensor.py`

- [ ] **Step 1: Tulis test, `tests/test_tensor.py`**

```python
import numpy as np
import pytest

from indogpt.autograd.tensor import Tensor, cross_entropy, unbroadcast

torch = pytest.importorskip("torch")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _pair(rng, *shape):
    x = rng.normal(size=shape)
    return Tensor(x), torch.tensor(x, requires_grad=True)


def test_unbroadcast_reduces_to_original_shape():
    g = np.ones((4, 3))
    np.testing.assert_allclose(unbroadcast(g, (3,)), np.full(3, 4.0))
    np.testing.assert_allclose(unbroadcast(g, (4, 1)), np.full((4, 1), 3.0))
    np.testing.assert_allclose(unbroadcast(g, (4, 3)), g)


def test_broadcast_add_and_mul(rng):
    X, Xt = _pair(rng, 4, 3)
    B, Bt = _pair(rng, 3)
    W, Wt = _pair(rng, 4, 1)
    ((X + B) * W).sum().backward()
    ((Xt + Bt) * Wt).sum().backward()
    for ours, theirs in [(X, Xt), (B, Bt), (W, Wt)]:
        np.testing.assert_allclose(ours.grad, theirs.grad.numpy())


def test_matmul(rng):
    A, At = _pair(rng, 5, 4)
    B, Bt = _pair(rng, 4, 3)
    (A @ B).sum().backward()
    (At @ Bt).sum().backward()
    np.testing.assert_allclose(A.grad, At.grad.numpy())
    np.testing.assert_allclose(B.grad, Bt.grad.numpy())


def test_elementwise_chain(rng):
    X, Xt = _pair(rng, 3, 4)
    y = ((X.tanh() + X.relu()) * X.exp() + (X**2 + 1.0).log()).mean()
    yt = ((Xt.tanh() + Xt.relu()) * Xt.exp() + (Xt**2 + 1.0).log()).mean()
    y.backward()
    yt.backward()
    assert float(y.data) == pytest.approx(yt.item())
    np.testing.assert_allclose(X.grad, Xt.grad.numpy(), rtol=1e-8)


def test_sum_with_axis_and_keepdims(rng):
    X, Xt = _pair(rng, 2, 3, 4)
    W, Wt = _pair(rng, 2, 1, 4)
    (X.sum(axis=1, keepdims=True) * W).sum().backward()
    (Xt.sum(dim=1, keepdim=True) * Wt).sum().backward()
    np.testing.assert_allclose(X.grad, Xt.grad.numpy())
    np.testing.assert_allclose(W.grad, Wt.grad.numpy())


def test_reused_tensor_accumulates_grad(rng):
    X, Xt = _pair(rng, 3)
    (X * X + X).sum().backward()
    (Xt * Xt + Xt).sum().backward()
    np.testing.assert_allclose(X.grad, Xt.grad.numpy())


def test_cross_entropy_matches_torch(rng):
    L, Lt = _pair(rng, 6, 5)
    targets = np.array([0, 4, 2, 2, 1, 3])
    loss = cross_entropy(L, targets)
    loss_t = torch.nn.functional.cross_entropy(Lt, torch.from_numpy(targets).long())
    loss.backward()
    loss_t.backward()
    assert float(loss.data) == pytest.approx(loss_t.item())
    np.testing.assert_allclose(L.grad, Lt.grad.numpy(), rtol=1e-8)


def test_cross_entropy_is_stable_for_large_logits():
    logits = Tensor(np.array([[1000.0, 0.0], [0.0, 1000.0]]))
    loss = cross_entropy(logits, np.array([0, 1]))
    assert np.isfinite(loss.data)
    assert float(loss.data) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Tulis stub, `src/indogpt/autograd/tensor.py`**

```python
"""Autograd tensor berbasis NumPy (versi vektor dari autograd skalar).

Latihan: implementasikan semua fungsi/method yang masih NotImplementedError.
"""
from __future__ import annotations

import numpy as np


def _noop() -> None:
    pass


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Jumlahkan `grad` sepanjang dimensi yang di-broadcast agar bentuknya kembali `shape`."""
    raise NotImplementedError


class Tensor:
    def __init__(self, data, _children: tuple[Tensor, ...] = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = _noop
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    # ---- operasi dasar: LATIHAN ----
    def __add__(self, other: Tensor | float) -> Tensor:
        raise NotImplementedError

    def __mul__(self, other: Tensor | float) -> Tensor:
        raise NotImplementedError

    def __matmul__(self, other: Tensor) -> Tensor:
        raise NotImplementedError

    def __pow__(self, k: float) -> Tensor:
        raise NotImplementedError

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        raise NotImplementedError

    def relu(self) -> Tensor:
        raise NotImplementedError

    def tanh(self) -> Tensor:
        raise NotImplementedError

    def exp(self) -> Tensor:
        raise NotImplementedError

    def log(self) -> Tensor:
        raise NotImplementedError

    def backward(self) -> None:
        raise NotImplementedError

    # ---- operasi turunan: sudah lengkap ----
    def __neg__(self) -> Tensor:
        return self * -1.0

    def __radd__(self, other: float) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self + (-other)

    def __rsub__(self, other: float) -> Tensor:
        return other + (-self)

    def __rmul__(self, other: float) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | float) -> Tensor:
        return self * other**-1

    def __rtruediv__(self, other: float) -> Tensor:
        return other * self**-1

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        s = self.sum(axis=axis, keepdims=keepdims)
        return s * (s.data.size / self.data.size)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Rata-rata cross-entropy. logits: (N, C), targets: (N,) indeks kelas. Harus stabil numerik."""
    raise NotImplementedError
```

- [ ] **Step 3: Jalankan test untuk memastikan gagal**

Run: `python -m pytest tests/test_tensor.py -v`
Expected: 8 FAILED dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan.** Petunjuk:
  - Polanya sama dengan `Value`, tapi `grad` berupa array dengan bentuk yang sama dengan `data`.
  - **`unbroadcast`:** pertama, jumlahkan sepanjang sumbu-sumbu depan yang tidak ada di `shape` (`grad.ndim - len(shape)` sumbu). Kedua, untuk setiap sumbu yang di `shape` bernilai 1, jumlahkan dengan `keepdims=True`.
  - **`+` dan `*`:** di backward, bungkus gradien setiap anak dengan `unbroadcast(..., anak.shape)`. Inilah yang membuat bias `(3,)` menerima gradien yang benar.
  - **`@` (2D):** `dA = dOut @ B.T`, `dB = A.T @ dOut`.
  - **`sum`:** gradien disebar kembali ke bentuk asal. Kalau `keepdims=False` dan `axis` bukan `None`, gunakan `np.expand_dims(out.grad, axis)` dulu, lalu `np.broadcast_to(..., self.shape)`.
  - **Turunan lokal:** `log → 1/x` · `relu → (x > 0)` · `tanh → 1 − tanh²` · `exp → exp(x)` · `x**k → k·x**(k-1)`.
  - **`backward()`:** urutan topologis seperti `Value`, dengan gradien awal `np.ones_like(self.data)`.
  - **`cross_entropy`** sebagai satu operasi gabungan (fused):
    - Forward: `z = logits − max(logits, axis=1, keepdims=True)`, `lse = log(sum(exp(z), axis=1))`, `loss = mean(lse − z[arange(N), targets])`.
    - Backward: `p = softmax(z)`, kurangi 1 pada `p[arange(N), targets]`, lalu `logits.grad += p / N * out.grad`.
    - Hasil inilah yang kamu turunkan di kertas minggu 2 (`p − y`).

- [ ] **Step 5: Jalankan semua test**

Run: `python -m pytest -v`
Expected: semua test lulus (smoke, scalar, tensor)

- [ ] **Step 6: Commit**

```powershell
git add src/indogpt/autograd/tensor.py tests/test_tensor.py
git commit -m "feat: autograd tensor NumPy dengan broadcasting dan cross-entropy stabil"
```

### ✅ Checkpoint Minggu 3
- [ ] Bisa menjelaskan kenapa `cross_entropy` mengurangi `max` dulu (stabilitas `exp`), dan kenapa gradiennya tidak berubah karenanya
- [ ] 2 paragraf MSE/cross-entropy ⇔ MLE sudah ditulis

---

# MINGGU 4 (22–28 Nov): Python engineering, Docker, MLP, README 1a

### 📘 Belajar (±5 jam)
- [ ] Python Packaging User Guide: tutorial *Packaging Python Projects*
- [ ] Dokumentasi pytest: *How to use fixtures* dan *How to parametrize fixtures and test functions*
- [ ] Dokumentasi Docker *Get started*: konsep dasar + workshop membangun dan menjalankan image. Instal Docker Desktop (butuh WSL2).
- [ ] Dokumentasi Python `dataclasses` dan dasar type hints (`typing`)

### 🛠️ Task 7: 🧠 Lapisan NN: `Linear`, `MLP`, `SGD`

**Files:**
- Create: `src/indogpt/autograd/nn.py`
- Test: `tests/test_nn.py`

- [ ] **Step 1: Tulis test, `tests/test_nn.py`**

```python
import numpy as np
import pytest

from indogpt.autograd.nn import MLP, SGD, Linear
from indogpt.autograd.tensor import Tensor, cross_entropy


def test_linear_shapes_and_init():
    rng = np.random.default_rng(0)
    layer = Linear(64, 32, rng)
    assert layer.weight.data.shape == (64, 32)
    assert layer.bias.data.shape == (32,)
    assert np.all(layer.bias.data == 0)
    assert layer.weight.data.std() == pytest.approx(np.sqrt(2 / 64), rel=0.1)
    out = layer(Tensor(rng.normal(size=(8, 64))))
    assert out.data.shape == (8, 32)


def test_mlp_parameter_count():
    mlp = MLP([2, 16, 16, 3], np.random.default_rng(0))
    n = sum(p.data.size for p in mlp.parameters())
    assert n == (2 * 16 + 16) + (16 * 16 + 16) + (16 * 3 + 3)


def test_zero_grad_resets_gradients():
    rng = np.random.default_rng(0)
    mlp = MLP([4, 8, 2], rng)
    x = Tensor(rng.normal(size=(5, 4)))
    cross_entropy(mlp(x), np.array([0, 1, 0, 1, 1])).backward()
    assert any(np.any(p.grad != 0) for p in mlp.parameters())
    mlp.zero_grad()
    assert all(np.all(p.grad == 0) for p in mlp.parameters())


def test_mlp_learns_two_blobs():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(-2, 0.5, size=(50, 2)), rng.normal(2, 0.5, size=(50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    mlp = MLP([2, 16, 2], rng)
    opt = SGD(mlp.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = cross_entropy(mlp(Tensor(x)), y)
        loss.backward()
        opt.step()
    acc = (mlp(Tensor(x)).data.argmax(axis=1) == y).mean()
    assert acc > 0.95
```

- [ ] **Step 2: Tulis stub, `src/indogpt/autograd/nn.py`**

```python
"""Lapisan neural network di atas Tensor autograd NumPy."""
from __future__ import annotations

import numpy as np

from indogpt.autograd.tensor import Tensor


class Module:
    def parameters(self) -> list[Tensor]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


class Linear(Module):
    """y = x @ W + b. W ~ N(0, sqrt(2 / n_in)) (inisialisasi He), b = 0."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        raise NotImplementedError


class MLP(Module):
    """Tumpukan Linear dengan ReLU di antaranya (tanpa ReLU setelah lapisan terakhir)."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        raise NotImplementedError


class SGD:
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 3: Jalankan test untuk memastikan gagal**

Run: `python -m pytest tests/test_nn.py -v`
Expected: 4 FAILED dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan.** Petunjuk:
  - `Linear` menyimpan atribut `self.weight` (bentuk `(n_in, n_out)`) dan `self.bias` (bentuk `(n_out,)`), keduanya `Tensor`. Gunakan `rng.normal(0, np.sqrt(2 / n_in), size=...)`.
  - `MLP` menyimpan daftar `Linear` dan memanggil `.relu()` setelah setiap lapisan **kecuali** yang terakhir, karena lapisan terakhir menghasilkan logits.
  - `SGD.step`: `p.data -= self.lr * p.grad` untuk setiap parameter.
  - Pertanyaan untuk dipikirkan: kenapa inisialisasi He memakai `sqrt(2 / n_in)` untuk ReLU? (Géron Bab 11, atau makemore part 3 minggu 6.)

- [ ] **Step 5: Jalankan semua test**

Run: `python -m pytest -v`
Expected: semua lulus

- [ ] **Step 6: Commit**

```powershell
git add src/indogpt/autograd/nn.py tests/test_nn.py
git commit -m "feat: Linear, MLP, dan SGD di atas autograd NumPy"
```

### 🛠️ Task 8: Latih MLP pada dataset digits

**Files:**
- Create: `scripts/train_digits.py`

- [ ] **Step 1: Tulis skrip**

```python
"""Latih MLP autograd NumPy buatan sendiri pada dataset digits (8x8) dari scikit-learn."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from indogpt.autograd.nn import MLP, SGD
from indogpt.autograd.tensor import Tensor, cross_entropy


def main() -> None:
    rng = np.random.default_rng(0)
    x, y = load_digits(return_X_y=True)
    x = x / 16.0  # skala piksel 0..16 -> 0..1
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    mlp = MLP([64, 64, 10], rng)
    opt = SGD(mlp.parameters(), lr=0.2)
    batch_size = 64
    for epoch in range(40):
        idx = rng.permutation(len(x_tr))
        for i in range(0, len(idx), batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            loss = cross_entropy(mlp(Tensor(x_tr[b])), y_tr[b])
            loss.backward()
            opt.step()
        acc = (mlp(Tensor(x_te)).data.argmax(axis=1) == y_te).mean()
        print(f"epoch {epoch + 1:2d}  loss {float(loss.data):.4f}  test acc {acc:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Jalankan**

Run: `python scripts/train_digits.py`
Expected: loss turun, dan `test acc` di epoch 40 **≥ 0.95** (implementasi referensi mendapat 0,967–0,972 di 5 seed). Kalau di bawah 0,9, hampir pasti ada bug di gradien. Ulangi `pytest tests/test_tensor.py`. Catat angka akhirnya untuk README (Task 10).

- [ ] **Step 3: Commit**

```powershell
git add scripts/train_digits.py
git commit -m "feat: latih MLP NumPy pada dataset digits"
```

### 🛠️ Task 9: Dockerfile untuk test

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Tulis Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[dev]"
COPY tests ./tests
CMD ["pytest", "-q"]
```

- [ ] **Step 2: Build dan jalankan**

```powershell
docker build -t indogpt-mini-test .
docker run --rm indogpt-mini-test
```

Expected: semua test lulus di dalam container. Pertanyaan untuk dipikirkan: kenapa `COPY tests` diletakkan **setelah** `pip install`? (Petunjuk: layer cache.)

- [ ] **Step 3: Commit**

```powershell
git add Dockerfile
git commit -m "build: Dockerfile untuk menjalankan test"
```

### 🛠️ Task 10: README tahap 1a + tag rilis

**Files:**
- Modify: `README.md` (ganti seluruh isi)

- [ ] **Step 1: Tulis README.** Ganti `<akurasi>` dengan angka dari Task 8, dan `USERNAME` dengan username GitHub-mu.

````markdown
# IndoGPT-mini

![ci](https://github.com/USERNAME/indogpt-mini/actions/workflows/ci.yml/badge.svg)

GPT kecil untuk Bahasa Indonesia yang dibangun **dari nol**: mesin autograd NumPy, tokenizer BPE, dan Transformer di PyTorch.

> Status: tahap 1a selesai (autograd NumPy + MLP). Tokenizer dan GPT sedang dikerjakan.

## Tahap 1a — Autograd NumPy

- `indogpt.autograd.scalar`: autograd skalar untuk memahami backpropagation.
- `indogpt.autograd.tensor`: autograd tensor dengan broadcasting, matmul, dan cross-entropy yang stabil numerik.
- Semua gradien diverifikasi terhadap PyTorch (`tests/test_tensor.py`).
- MLP 64-64-10 pada dataset digits: **akurasi test <akurasi>**.

## Menjalankan

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
python scripts/train_digits.py
```
````

- [ ] **Step 2: Commit, tag, push**

```powershell
git add README.md
git commit -m "docs: README tahap 1a"
git tag v0.1-autograd
git push --tags
git push
```

### ✅ Checkpoint Fase 0 (akhir Minggu 4)
- [ ] PyTorch berjalan di GPU lokal
- [ ] Bisa menurunkan gradien MSE dan cross-entropy di atas kertas **tanpa melihat catatan**
- [ ] CI hijau. Tahap 1a (autograd + MLP) lulus semua test dan ada di GitHub dengan README.
- [ ] 24 soal LeetCode (minggu 1–4) selesai, minimal 20 non-Hard

---

# MINGGU 5 (29 Nov – 5 Des): PyTorch fundamentals, makemore 1–2, korpus

### 📘 Belajar (±8 jam)
- [ ] Tutorial PyTorch *Learn the Basics* (pytorch.org/tutorials), semua bagian: tensors, datasets & dataloaders, transforms, build model, autograd, optimization, save & load
- [ ] Karpathy *makemore* part 1 (bigram) dan part 2 (MLP)
- [ ] Baca cepat Bengio et al. 2003, *A Neural Probabilistic Language Model*

### 🛠️ Task 11: Unduh dan bersihkan Wikipedia Bahasa Indonesia

**Files:**
- Create: `src/indogpt/data/__init__.py` (kosong)
- Create: `src/indogpt/data/prepare_text.py`
- Test: `tests/test_prepare_text.py`

- [ ] **Step 1: Tulis test, `tests/test_prepare_text.py`**

```python
from indogpt.data.prepare_text import clean_text


def test_clean_text_collapses_whitespace():
    raw = "  Halo   dunia\t!\n\n\n\nBaris  baru "
    assert clean_text(raw) == "Halo dunia !\n\nBaris baru"
```

- [ ] **Step 2: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_prepare_text.py -v`
Expected: ERROR `ModuleNotFoundError: No module named 'indogpt.data.prepare_text'`

- [ ] **Step 3: Tulis `src/indogpt/data/prepare_text.py`**

```python
"""Unduh Wikipedia Bahasa Indonesia, bersihkan, lalu simpan sebagai satu file teks."""
import argparse
import re
from pathlib import Path

DOC_SEP = "\n<|endoftext|>\n"
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def main() -> None:
    from datasets import load_dataset  # impor di sini agar test tidak butuh `datasets`

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/idwiki.txt")
    p.add_argument("--max-chars", type=int, default=200_000_000)
    p.add_argument("--min-chars", type=int, default=200)
    args = p.parse_args()

    ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = n_docs = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = clean_text(row["text"])
            if len(text) < args.min_chars:
                continue
            f.write(text + DOC_SEP)
            total += len(text)
            n_docs += 1
            if total >= args.max_chars:
                break
    print(f"{n_docs} dokumen, {total / 1e6:.1f} juta karakter -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Jalankan test**

Run: `python -m pytest tests/test_prepare_text.py -v`
Expected: `1 passed`

- [ ] **Step 5: Unduh korpus**

Run: `python -m indogpt.data.prepare_text`
Expected: muncul ringkasan seperti `... dokumen, 200.0 juta karakter -> data\raw\idwiki.txt`. Unduhan butuh beberapa menit sampai puluhan menit, tergantung koneksi. Buka file itu dan baca beberapa dokumen untuk memeriksa kualitas datanya.

- [ ] **Step 6: Commit** (file data otomatis diabaikan oleh `.gitignore`)

```powershell
git add src/indogpt/data tests/test_prepare_text.py
git commit -m "feat: unduh dan bersihkan Wikipedia Bahasa Indonesia"
```

### 🛠️ Task 12: 🧠 Baseline bigram byte-level (bits-per-byte)

Kenapa memakai **bits-per-byte (BPB)**? Perplexity tidak bisa dibandingkan antar-tokenizer yang berbeda, sedangkan BPB bisa. Baseline ini akan menjadi baris pertama tabel hasil di README.

**Files:**
- Create: `src/indogpt/baseline.py`
- Create: `scripts/baseline_bigram.py`
- Test: `tests/test_baseline.py`

- [ ] **Step 1: Tulis test, `tests/test_baseline.py`**

```python
import numpy as np

from indogpt.baseline import bits_per_byte, fit_bigram


def test_rows_are_normalized_distributions():
    logp = fit_bigram(b"halo dunia, halo indonesia")
    assert logp.shape == (256, 256)
    np.testing.assert_allclose(np.exp(logp).sum(axis=1), np.ones(256))


def test_repetitive_text_is_cheap():
    data = b"ab" * 1000
    assert bits_per_byte(fit_bigram(data), data) < 0.1


def test_random_bytes_cost_about_eight_bits():
    data = np.random.default_rng(0).integers(0, 256, size=100_000, dtype=np.uint8).tobytes()
    assert 7.0 < bits_per_byte(fit_bigram(data), data) <= 8.0
```

- [ ] **Step 2: Tulis stub, `src/indogpt/baseline.py`**

```python
"""Baseline bigram tingkat byte: pembanding paling sederhana untuk GPT."""
import numpy as np


def fit_bigram(data: bytes, alpha: float = 0.01) -> np.ndarray:
    """Matriks (256, 256) berisi log p(byte_berikut | byte_sebelum), dengan smoothing `alpha`."""
    raise NotImplementedError


def bits_per_byte(logp: np.ndarray, data: bytes) -> float:
    """Rata-rata -log2 p(data[i+1] | data[i]) untuk semua pasangan berurutan."""
    raise NotImplementedError
```

- [ ] **Step 3: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: 3 FAILED dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan.** Petunjuk:
  - `a = np.frombuffer(data, dtype=np.uint8).astype(np.int64)`
  - Hitung kemunculan pasangan dengan `np.bincount(a[:-1] * 256 + a[1:], minlength=65536).reshape(256, 256)`, jauh lebih cepat daripada loop.
  - Tambahkan `alpha`, normalisasi per baris, lalu ambil `np.log`.
  - `bits_per_byte`: `-logp[a[:-1], a[1:]].mean() / np.log(2)`.
  - Ini persis model bigram di makemore part 1, hanya saja per byte, bukan per huruf.

- [ ] **Step 5: Jalankan test**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: `3 passed`

- [ ] **Step 6: Tulis `scripts/baseline_bigram.py`**

```python
"""Baseline bigram byte-level. Hasilnya (bits-per-byte) menjadi pembanding GPT di README."""
import argparse
from pathlib import Path

from indogpt.baseline import bits_per_byte, fit_bigram

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="data/raw/idwiki.txt")
    p.add_argument("--max-bytes", type=int, default=50_000_000)
    args = p.parse_args()

    data = Path(args.text).read_bytes()[: args.max_bytes]
    split = int(0.9 * len(data))
    logp = fit_bigram(data[:split])
    print(f"bigram val bits-per-byte: {bits_per_byte(logp, data[split:]):.3f}")
```

- [ ] **Step 7: Jalankan dan catat hasilnya**

Run: `python scripts/baseline_bigram.py`
Expected: satu angka BPB, biasanya sekitar 3–4 (uji coba referensi pada 7 juta karakter awal: 3,48). Catat untuk tabel hasil di Task 21.

- [ ] **Step 8: Commit**

```powershell
git add src/indogpt/baseline.py scripts/baseline_bigram.py tests/test_baseline.py
git commit -m "feat: baseline bigram byte-level dengan metrik bits-per-byte"
```

---

# MINGGU 6 (6–12 Des): makemore 3–4 + tokenizer BPE

### 📘 Belajar (±7 jam)
- [ ] Karpathy *makemore* part 3 (*Activations & Gradients, BatchNorm*) dan part 4 (*Becoming a Backprop Ninja*)
- [ ] Karpathy *Let's build the GPT Tokenizer*
- [ ] Baca cepat Sennrich et al. 2016, *Neural Machine Translation of Rare Words with Subword Units*, bagian 3

### 🛠️ Task 13: 🧠 Tokenizer BPE (versi naif)

**Files:**
- Create: `src/indogpt/tokenizer/__init__.py` (kosong)
- Create: `src/indogpt/tokenizer/bpe.py`
- Test: `tests/test_bpe.py`

- [ ] **Step 1: Tulis test, `tests/test_bpe.py`**

```python
import pytest

from indogpt.tokenizer.bpe import EOT, BPETokenizer

CORPUS = (
    "Indonesia adalah negara kepulauan terbesar di dunia. "
    "Ibu kota Indonesia adalah Jakarta. Pekanbaru adalah ibu kota Provinsi Riau. "
) * 20


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    return t


def test_vocab_size(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256 - 1


def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer()
    t.train("aaabdaaabac", vocab_size=258)
    assert list(t.merges.items()) == [((97, 97), 256)]


def test_ties_broken_by_smallest_pair():
    t = BPETokenizer()
    t.train("abcd", vocab_size=258)  # 1 merge; (a,b), (b,c), (c,d) sama-sama muncul 1 kali
    assert list(t.merges.items()) == [((97, 98), 256)]


@pytest.mark.parametrize(
    "text",
    ["", "Halo, dunia!", "Pekanbaru 2026 — kota bertuah 🚀", "    spasi   ganda\n\nbaris baru", "ñandú ✓"],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_compresses_training_text(tok):
    text = "Ibu kota Indonesia adalah Jakarta."
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "Pekanbaru adalah ibu kota Provinsi Riau."
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.vocab_size == tok.vocab_size


def test_eot_token(tok):
    assert tok.eot_id == tok.vocab_size - 1
    assert tok.decode([tok.eot_id]) == EOT
```

- [ ] **Step 2: Tulis stub, `src/indogpt/tokenizer/bpe.py`**

```python
"""Tokenizer BPE byte-level (ala GPT-2). Latihan: implementasikan method yang NotImplementedError."""
from __future__ import annotations

import json
from pathlib import Path

import regex

# Pra-tokenisasi: kata (opsional diawali spasi), angka, tanda baca, spasi.
SPLIT_PATTERN = r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}  # (id_a, id_b) -> id_baru, urut sesuai merge
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._pattern = regex.compile(SPLIT_PATTERN)
        self._cache: dict[str, list[int]] = {}

    @property
    def eot_id(self) -> int:
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + 1  # +1 untuk token EOT

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        data = {"merges": [list(pair) for pair in self.merges]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        tok = cls()
        merges = json.loads(Path(path).read_text(encoding="utf-8"))["merges"]
        for i, (a, b) in enumerate(merges):
            new_id = 256 + i
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok
```

- [ ] **Step 3: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: semua FAILED/ERROR dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan.** Petunjuk:
  - **`train`:**
    1. Pecah teks dengan `self._pattern.findall(text)`, lalu hitung frekuensi setiap potongan unik (`collections.Counter`).
    2. Ubah setiap potongan unik menjadi `list(chunk.encode("utf-8"))`.
    3. Ulangi `vocab_size − 257` kali: hitung frekuensi semua pasangan bersebelahan (dibobot frekuensi potongan), lalu pilih pasangan terbaik dengan **aturan tie-break wajib** `best = min(stats, key=lambda p: (-stats[p], p))`, yaitu frekuensi terbesar, dan kalau seri, pasangan terkecil. Beri id baru `256 + i`, ganti pasangan itu di semua potongan, dan perbarui `self.merges` serta `self.vocab`.
    4. Merge **tidak pernah** melewati batas potongan.
  - **`encode`:** untuk setiap potongan (pakai `self._cache`), mulai dari byte-nya. Selama ada pasangan yang ada di `merges`, gabungkan pasangan dengan **id merge terkecil** (`min(pairs, key=lambda p: self.merges.get(p, float("inf")))`).
  - **`decode`:** gabungkan `self.vocab[i]` (atau `EOT.encode()` untuk `eot_id`), lalu `.decode("utf-8", errors="replace")`.
  - Referensi: video tokenizer Karpathy, bagian *training the tokenizer* dan *encoding/decoding*.

- [ ] **Step 5: Jalankan test**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: semua lulus

- [ ] **Step 6: Commit**

```powershell
git add src/indogpt/tokenizer tests/test_bpe.py
git commit -m "feat: tokenizer BPE byte-level (training naif)"
```

### 🛠️ Task 14: 🧠 Optimasi training BPE (incremental)

Versi naif menghitung ulang **semua** pasangan di setiap merge. Untuk 5 juta karakter dan vocab 8192, waktunya bisa berjam-jam. Ini masalah engineering nyata, dan bagus untuk diceritakan saat wawancara.

**Files:**
- Modify: `src/indogpt/tokenizer/bpe.py` (ganti nama `train` lama menjadi `train_naive`, tulis `train` baru)
- Test: `tests/test_bpe.py` (tambah 1 test)

- [ ] **Step 1: Tambahkan test di akhir `tests/test_bpe.py`**

```python
def test_fast_training_matches_naive():
    text = CORPUS + "Sungai Siak mengalir melewati kota Pekanbaru. " * 10
    fast, naive = BPETokenizer(), BPETokenizer()
    fast.train(text, vocab_size=320)
    naive.train_naive(text, vocab_size=320)
    assert list(fast.merges.items()) == list(naive.merges.items())
```

- [ ] **Step 2: Ganti nama `train` menjadi `train_naive`.** Lalu tambahkan `train` baru yang sementara berisi `raise NotImplementedError`.

Run: `python -m pytest tests/test_bpe.py -v`
Expected: test yang memanggil `train` FAILED

- [ ] **Step 3: 🧠 Implementasikan `train` incremental.** Petunjuk:
  - Simpan `chunks: list[list[int]]`, `freqs: list[int]`, counter global `stats[pair]`, dan indeks `where[pair] -> set(indeks_chunk)`.
  - Di setiap merge, ambil `best` dengan aturan tie-break yang **sama persis** dengan versi naif.
  - Untuk setiap `i` di `where.pop(best)`: (1) kurangi `stats` dengan semua pasangan di `chunks[i]` × `freqs[i]`; (2) terapkan merge ke `chunks[i]`; (3) tambahkan lagi pasangan barunya ke `stats` × `freqs[i]`, dan daftarkan `i` ke `where` untuk setiap pasangan baru. Lewati `i` kalau `best` ternyata sudah tidak ada di chunk itu (indeks basi).
  - Hapus entri `stats` yang nilainya ≤ 0.
  - Kenapa hasilnya identik dengan versi naif? Karena `stats` selalu sama persis dengan hitungan ulang penuh. Hanya chunk yang berubah yang dihitung ulang.

- [ ] **Step 4: Jalankan test**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: semua lulus, termasuk `test_fast_training_matches_naive`

- [ ] **Step 5: Commit**

```powershell
git add src/indogpt/tokenizer/bpe.py tests/test_bpe.py
git commit -m "perf: training BPE incremental, hasil identik dengan versi naif"
```

### 🛠️ Task 15: Latih tokenizer pada korpus

**Files:**
- Create: `scripts/train_tokenizer.py`
- Create: `artifacts/tokenizer.json` (hasil, di-commit)

- [ ] **Step 1: Tulis skrip**

```python
"""Latih tokenizer BPE pada potongan awal korpus, lalu laporkan rasio kompresi."""
import argparse
import time
from pathlib import Path

from indogpt.tokenizer.bpe import BPETokenizer

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="data/raw/idwiki.txt")
    p.add_argument("--sample-chars", type=int, default=5_000_000)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--out", default="artifacts/tokenizer.json")
    args = p.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    sample = text[: args.sample_chars]
    held_out = text[args.sample_chars : args.sample_chars + 1_000_000]

    t0 = time.perf_counter()
    tok = BPETokenizer()
    tok.train(sample, vocab_size=args.vocab_size, verbose=True)
    print(f"training tokenizer: {time.perf_counter() - t0:.0f} detik")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)

    n_bytes = len(held_out.encode("utf-8"))
    n_tokens = len(tok.encode(held_out))
    print(f"kompresi (held-out): {n_bytes / n_tokens:.2f} byte/token")
    for word in ["Pekanbaru", " pemerintahan", " kepulauan", " dikembangkan"]:
        print(repr(word), "->", [tok.decode([i]) for i in tok.encode(word)])
```

- [ ] **Step 2: Jalankan**

Run: `python scripts/train_tokenizer.py`
Expected:
- Waktu training **kurang dari 10 menit**. Implementasi referensi butuh ±106 detik di laptop ini. Kalau punyamu jauh lebih lama, profil dulu dengan `python -m cProfile -s cumtime scripts/train_tokenizer.py --sample-chars 500000`.
- Kompresi sekitar 3–4,5 byte/token (referensi: 4,10).
- Kata-kata umum seperti `" pemerintahan"` menjadi 1–3 token. Referensi: `" pemerintahan"`, `" kepulauan"`, `" dikembangkan"` masing-masing 1 token, dan `"Pekanbaru"` → `P | ekan | baru`.

Catat waktu training dan rasio kompresi untuk README.

- [ ] **Step 3: Commit tokenizer**

```powershell
git add scripts/train_tokenizer.py artifacts/tokenizer.json
git commit -m "feat: tokenizer BPE 8192 terlatih pada Wikipedia ID"
```

### ✅ Checkpoint Minggu 6
- [ ] Bisa menjelaskan kenapa GPT-2 memakai byte-level BPE (tidak ada token "unknown")
- [ ] Bisa menjelaskan perbedaan kompleksitas training BPE versi naif dan incremental

---

# MINGGU 7 (13–19 Des): Transformer & model GPT

### 📘 Belajar (±7 jam)
- [ ] Karpathy *makemore* part 5 (*Building a WaveNet*)
- [ ] Karpathy *Let's build GPT: from scratch, in code, spelled out*
- [ ] Vaswani et al. 2017, *Attention Is All You Need*, bagian 3 (arsitektur)
- [ ] Jay Alammar, *The Illustrated Transformer*

### 🛠️ Task 16: 🧠 Model GPT

**Files:**
- Create: `src/indogpt/model/__init__.py` (kosong)
- Create: `src/indogpt/model/config.py`
- Create: `src/indogpt/model/gpt.py`
- Test: `tests/test_gpt.py`

- [ ] **Step 1: Tulis `src/indogpt/model/config.py`** (lengkap)

```python
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0


# Tiga ukuran untuk eksperimen ablation (perkiraan jumlah parameter dengan vocab 8192).
PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"n_layer": 4, "n_head": 4, "n_embd": 128},  # ~1,9 juta
    "small": {"n_layer": 6, "n_head": 4, "n_embd": 256},  # ~6,9 juta
    "medium": {"n_layer": 8, "n_head": 6, "n_embd": 384},  # ~17 juta
}
```

- [ ] **Step 2: Tulis test, `tests/test_gpt.py`**

```python
import math

import pytest

torch = pytest.importorskip("torch")

from indogpt.model.config import GPTConfig  # noqa: E402
from indogpt.model.gpt import GPT  # noqa: E402

CFG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG)


def test_forward_shapes(model):
    idx = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss is None


def test_initial_loss_close_to_uniform(model):
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    _, loss = model(idx, targets=idx)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.5


def test_causal_masking(model):
    model.eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 12))
    changed = idx.clone()
    changed[0, 8] = (idx[0, 8] + 1) % CFG.vocab_size
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])


def test_weight_tying(model):
    assert model.lm_head.weight is model.wte.weight


def test_rejects_too_long_sequence(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, CFG.block_size + 1, dtype=torch.long))


def test_can_overfit_single_batch(model):
    torch.manual_seed(0)
    idx = torch.randint(0, CFG.vocab_size, (2, 9))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_greedy_is_deterministic(model):
    model.eval()
    prompt = torch.randint(0, CFG.vocab_size, (1, 4))
    out1 = model.generate(prompt, max_new_tokens=20, top_k=1)
    out2 = model.generate(prompt, max_new_tokens=20, top_k=1)
    assert out1.shape == (1, 24)  # lebih panjang dari block_size: konteks harus dipotong
    assert torch.equal(out1, out2)
```

- [ ] **Step 3: Tulis stub, `src/indogpt/model/gpt.py`**

```python
"""GPT kecil (Transformer decoder-only). Latihan: implementasikan semua kelas di bawah."""
from __future__ import annotations

import torch
import torch.nn as nn

from indogpt.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
```

- [ ] **Step 4: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_gpt.py -v`
Expected: 7 ERROR/FAILED dengan `NotImplementedError`

- [ ] **Step 5: 🧠 Implementasikan.** Petunjuk:
  - **`CausalSelfAttention`:**
    1. `qkv = nn.Linear(d, 3d)`, `proj = nn.Linear(d, d)`.
    2. Pecah ke head dengan bentuk `(B, n_head, T, d/n_head)`.
    3. `att = q @ k.transpose(-2, -1) / sqrt(head_dim)`.
    4. Isi bagian atas segitiga dengan `-inf` memakai buffer `torch.tril(torch.ones(block_size, block_size))` via `register_buffer`.
    5. Lalu `softmax`, dropout, `att @ v`, gabungkan head, `proj`, dropout.
    - Setelah test lulus, coba ganti ke `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. Test harus tetap lulus, dan bandingkan kecepatannya.
  - **`MLP`:** `Linear(d, 4d) → GELU → Linear(4d, d) → Dropout`.
  - **`Block` (pre-LN):** `x = x + attn(ln_1(x))`, lalu `x = x + mlp(ln_2(x))`.
  - **`GPT` wajib punya atribut** `wte` (Embedding vocab×d), `wpe` (Embedding block_size×d), `drop`, `blocks` (`nn.ModuleList`), `ln_f`, dan `lm_head` (`Linear(d, vocab, bias=False)`).
    - Weight tying: `self.lm_head.weight = self.wte.weight`.
    - **Inisialisasi:** `self.apply(...)` yang menginisialisasi semua `Linear`/`Embedding` dengan `normal(0, 0.02)` dan bias nol. Tanpa ini, `test_initial_loss_close_to_uniform` gagal. Coba pikirkan kenapa.
  - **`forward`:**
    1. Lempar `ValueError` jika `T > block_size`.
    2. `x = drop(wte(idx) + wpe(arange(T)))` → blocks → `ln_f` → `logits = lm_head(x)`.
    3. Loss: `F.cross_entropy(logits.view(-1, V), targets.view(-1))` (tambahkan `from torch.nn import functional as F`).
  - **`generate`:**
    1. Ulangi `max_new_tokens` kali: potong konteks ke `idx[:, -block_size:]`, ambil `logits[:, -1, :] / temperature`.
    2. Kalau `top_k` dipakai: `v, _ = torch.topk(logits, top_k)`, lalu `logits[logits < v[:, [-1]]] = -inf`.
    3. `softmax` → `torch.multinomial` → `torch.cat`.

- [ ] **Step 6: Jalankan test**

Run: `python -m pytest tests/test_gpt.py -v`
Expected: `7 passed`

- [ ] **Step 7: Cek jumlah parameter preset**

```powershell
python -c "from indogpt.model.config import GPTConfig, PRESETS; from indogpt.model.gpt import GPT; [print(k, GPT(GPTConfig(**v)).num_params()) for k, v in PRESETS.items()]"
```

Expected: kira-kira tiny ~1,9 juta, small ~6,9 juta, medium ~17 juta

- [ ] **Step 8: Commit**

```powershell
git add src/indogpt/model tests/test_gpt.py
git commit -m "feat: model GPT (pre-LN, weight tying, causal attention, generate)"
```

---

# MINGGU 8 (20–26 Des): Infrastruktur training

### 📘 Belajar (±5 jam)
- [ ] Karpathy *Let's reproduce GPT-2 (124M)*: bagian mixed precision, AdamW + weight decay, gradient clipping, LR schedule, dan gradient accumulation (kira-kira 2 jam pertama)
- [ ] Dokumentasi PyTorch *Automatic Mixed Precision*
- [ ] d2l.ai Bab 12: subbab Adam dan *Learning Rate Scheduling*

### 🛠️ Task 17: Tokenisasi korpus → `train.bin` / `val.bin`

**Files:**
- Create: `src/indogpt/data/encode.py`

- [ ] **Step 1: Tulis skrip**

```python
"""Tokenisasi idwiki.txt -> train.bin / val.bin (uint16) + meta.json."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from indogpt.data.prepare_text import DOC_SEP
from indogpt.tokenizer.bpe import BPETokenizer

_tok: BPETokenizer | None = None


def _init(tok_path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(tok_path)


def _encode_doc(doc: str) -> list[int]:
    return _tok.encode(doc) + [_tok.eot_id]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="data/raw/idwiki.txt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--out-dir", default="data/tokens")
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    docs = [d for d in Path(args.text).read_text(encoding="utf-8").split(DOC_SEP) if d]
    n_val = max(1, int(len(docs) * args.val_frac))
    splits = {"val": docs[:n_val], "train": docs[n_val:]}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[str, float] = {}
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for name, split_docs in splits.items():
            encoded = pool.imap(_encode_doc, split_docs, chunksize=256)
            ids = np.fromiter((t for doc_ids in encoded for t in doc_ids), dtype=np.uint16)
            ids.tofile(out / f"{name}.bin")
            n_bytes = sum(len(d.encode("utf-8")) for d in split_docs)
            meta[f"{name}_tokens"] = int(ids.size)
            meta[f"{name}_bytes"] = n_bytes
            print(f"{name}: {ids.size:,} token dari {n_bytes:,} byte")

    meta["vocab_size"] = BPETokenizer.load(args.tokenizer).vocab_size
    meta["tokens_per_byte"] = meta["val_tokens"] / meta["val_bytes"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Jalankan**

Run: `python -m indogpt.data.encode`
Expected:
- Muncul baris `val: ... token` dan `train: ... token` (train kira-kira puluhan juta token).
- `data/tokens/meta.json` berisi `vocab_size: 8192`.
- Proses butuh beberapa menit. Kalau CPU-mu punya lebih dari 4 core, coba `--workers 8`.

- [ ] **Step 3: Commit**

```powershell
git add src/indogpt/data/encode.py
git commit -m "feat: tokenisasi korpus paralel ke train.bin/val.bin"
```

### 🛠️ Task 18: 🧠 Loader batch + jadwal learning rate

**Files:**
- Create: `src/indogpt/data/loader.py`
- Create: `src/indogpt/train/__init__.py` (kosong)
- Create: `src/indogpt/train/lr.py`
- Test: `tests/test_loader.py`
- Test: `tests/test_lr.py`

- [ ] **Step 1: Tulis `tests/test_loader.py`**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from indogpt.data.loader import get_batch  # noqa: E402


def test_get_batch_shapes_and_shift():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=8, batch_size=4, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y - x, torch.ones_like(x))  # karena data = 0, 1, 2, ...


def test_get_batch_never_reads_past_end():
    data = np.arange(20, dtype=np.uint16)
    for seed in range(50):
        _, y = get_batch(data, block_size=8, batch_size=16, rng=np.random.default_rng(seed))
        assert int(y.max()) <= 19
```

- [ ] **Step 2: Tulis `tests/test_lr.py`**

```python
import pytest

from indogpt.train.lr import get_lr

KW = {"max_lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 100, "max_steps": 1000}


def test_linear_warmup():
    assert get_lr(0, **KW) == pytest.approx(1e-3 / 100)
    assert get_lr(49, **KW) == pytest.approx(1e-3 * 50 / 100)
    assert get_lr(99, **KW) == pytest.approx(1e-3)


def test_cosine_decay_endpoints_and_midpoint():
    assert get_lr(100, **KW) == pytest.approx(1e-3)
    assert get_lr(550, **KW) == pytest.approx((1e-3 + 1e-4) / 2)
    assert get_lr(1000, **KW) == pytest.approx(1e-4)
    assert get_lr(5000, **KW) == pytest.approx(1e-4)


def test_monotonic_decay_after_warmup():
    lrs = [get_lr(s, **KW) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
```

- [ ] **Step 3: Tulis stub**

`src/indogpt/data/loader.py`:

```python
import numpy as np
import torch


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ambil `batch_size` potongan acak sepanjang `block_size`. y = x digeser satu token."""
    raise NotImplementedError
```

`src/indogpt/train/lr.py`:

```python
def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Warmup linear selama `warmup_steps`, lalu cosine decay ke `min_lr` di `max_steps`."""
    raise NotImplementedError
```

- [ ] **Step 4: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_loader.py tests/test_lr.py -v`
Expected: semua FAILED dengan `NotImplementedError`

- [ ] **Step 5: 🧠 Implementasikan.** Petunjuk:
  - **`get_batch`:**
    1. Ambil indeks awal `ix = rng.integers(0, len(data) - block_size, size=batch_size)`.
    2. `x` = tumpukan `data[i : i + block_size]`, dan `y` = tumpukan `data[i + 1 : i + 1 + block_size]`.
    3. Konversi ke `int64` (`torch.from_numpy(arr.astype(np.int64))`), lalu `.to(device)`.
  - **`get_lr`:**
    - Jika `step < warmup_steps`: `max_lr * (step + 1) / warmup_steps`.
    - Jika `step >= max_steps`: `min_lr`.
    - Selain itu: `ratio = (step − warmup) / (max_steps − warmup)`, `coeff = 0.5 * (1 + cos(π · ratio))`, dan hasilnya `min_lr + coeff * (max_lr − min_lr)`.

- [ ] **Step 6: Jalankan test**

Run: `python -m pytest tests/test_loader.py tests/test_lr.py -v`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```powershell
git add src/indogpt/data/loader.py src/indogpt/train tests/test_loader.py tests/test_lr.py
git commit -m "feat: loader batch memmap dan jadwal LR warmup + cosine"
```

### 🛠️ Task 19: 🧠 Loop training

**Files:**
- Create: `src/indogpt/train/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Tulis smoke test, `tests/test_train.py`**

```python
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from indogpt.train.train import TrainConfig, train  # noqa: E402


def test_train_smoke(tmp_path):
    data_dir = tmp_path / "tokens"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 64, size=5000, dtype=np.uint16).tofile(data_dir / f"{split}.bin")
    (data_dir / "meta.json").write_text(json.dumps({"vocab_size": 64, "tokens_per_byte": 0.25}))

    cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(tmp_path / "run"),
        preset="tiny",
        block_size=16,
        batch_size=4,
        grad_accum=2,
        max_steps=6,
        warmup_steps=2,
        eval_interval=3,
        eval_batches=2,
        device="cpu",
    )
    result = train(cfg)

    assert {"best_val_loss", "best_val_bpb", "tokens_per_sec", "num_params", "peak_vram_mb"} <= (
        result.keys()
    )
    assert np.isfinite(result["best_val_loss"])
    assert result["best_val_bpb"] == pytest.approx(result["best_val_loss"] / np.log(2) * 0.25)
    ckpt = torch.load(tmp_path / "run" / "ckpt.pt", map_location="cpu")
    assert {"model", "model_config", "train_config", "step", "val_loss"} <= ckpt.keys()
```

- [ ] **Step 2: Tulis stub + CLI lengkap, `src/indogpt/train/train.py`**

```python
"""Loop training IndoGPT-mini. Latihan: implementasikan `train()`. Config dan CLI sudah lengkap."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields


@dataclass
class TrainConfig:
    data_dir: str = "data/tokens"
    out_dir: str = "runs/default"
    preset: str = "small"
    block_size: int = 256
    batch_size: int = 16
    grad_accum: int = 4
    max_steps: int = 5000
    warmup_steps: int = 200
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 50
    device: str = "auto"
    seed: int = 0
    wandb: bool = False


def train(cfg: TrainConfig) -> dict:
    """Latih model, simpan checkpoint terbaik, dan kembalikan ringkasan hasil (lihat Task 19)."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih IndoGPT-mini")
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)
    result = train(TrainConfig(**vars(parser.parse_args())))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Jalankan untuk memastikan gagal**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAILED dengan `NotImplementedError`

- [ ] **Step 4: 🧠 Implementasikan `train()`.** Perilaku wajib:
  1. **Reproducibility:** `torch.manual_seed(cfg.seed)` dan `rng = np.random.default_rng(cfg.seed)`.
  2. **Device:** `"auto"` → `"cuda"` kalau tersedia, selain itu `"cpu"`.
  3. **Data:** baca `meta.json`. `np.memmap(path, dtype=np.uint16, mode="r")` untuk `train.bin` dan `val.bin`.
  4. **Model:** `GPTConfig(vocab_size=meta["vocab_size"], block_size=cfg.block_size, **PRESETS[cfg.preset])` → `GPT(...).to(device)`.
  5. **Optimizer:** `AdamW` dengan dua grup parameter. Parameter dengan `p.dim() >= 2` memakai `weight_decay=cfg.weight_decay`, sisanya (bias, LayerNorm) memakai `0.0`. Gunakan `betas=(0.9, 0.95)`.
  6. **Mixed precision:** di CUDA, bungkus forward dengan `torch.autocast("cuda", dtype=torch.bfloat16)`; di CPU pakai `contextlib.nullcontext()`. bf16 tidak butuh GradScaler.
  7. **Setiap step:**
     - Set `lr = get_lr(...)` ke semua `param_groups`.
     - Ulangi `grad_accum` kali: `get_batch` → forward → `(loss / grad_accum).backward()`.
     - `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)`.
     - `opt.step()`, lalu `opt.zero_grad(set_to_none=True)`.
  8. **Evaluasi** (saat `step % eval_interval == 0` dan di step terakhir):
     - `model.eval()` + `torch.no_grad()`, rata-rata loss dari `eval_batches` batch val.
     - `val_bpb = val_loss / ln(2) * meta["tokens_per_byte"]`.
     - Print step, train loss, val loss, val bpb, lr, token/detik.
     - Kalau `val_loss` terbaik sejauh ini, simpan `out_dir/ckpt.pt` berisi `{"model": state_dict, "model_config": asdict(gpt_cfg), "train_config": asdict(cfg), "step": step, "val_loss": val_loss}`.
     - Kembalikan ke `model.train()`.
  9. **Throughput:** `token_per_step = batch_size * block_size * grad_accum`. Ukur dengan `time.perf_counter()`, dan panggil `torch.cuda.synchronize()` sebelum membaca waktu di CUDA. Ukur **hanya waktu step training**: mulai timer setelah evaluasi, supaya angka token/detik tidak ikut turun saat ada eval.
  10. **W&B** (kalau `cfg.wandb`): `import wandb` di dalam fungsi, `wandb.init(project="indogpt-mini", name=Path(cfg.out_dir).name, config=asdict(cfg))`, `wandb.log({...})` setiap evaluasi, dan `wandb.finish()` di akhir.
  11. **Return:** `{"best_val_loss", "best_val_bpb", "tokens_per_sec", "num_params", "peak_vram_mb"}`. Nilai `peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20` di CUDA, `0.0` di CPU.

- [ ] **Step 5: Jalankan test**

Run: `python -m pytest -v`
Expected: semua test di repo lulus

- [ ] **Step 6: Commit**

```powershell
git add src/indogpt/train/train.py tests/test_train.py
git commit -m "feat: loop training bf16 + grad accumulation + cosine LR + checkpoint"
```

### 🛠️ Task 20: Sanity run di GPU

- [ ] **Step 1: Jalankan training singkat**

Run: `python -m indogpt.train.train --preset tiny --max-steps 300 --eval-interval 100 --out-dir runs/sanity`
Expected:
- Val loss di step 0 sekitar 9,0 (= ln 8192), lalu turun jelas di step 300 (misalnya ke bawah 7).
- Tidak ada `nan`.
- `peak_vram_mb` jauh di bawah 6000.

- [ ] **Step 2: Kalau loss tidak turun, debug** dengan urutan dari *A Recipe for Training Neural Networks* (minggu 9):
  1. Overfit satu batch (sudah ada di test GPT).
  2. Cek bahwa `lr` benar-benar berubah setiap step.
  3. Cek bahwa `x`/`y` hasil `get_batch` bisa di-decode menjadi teks Indonesia yang masuk akal.

### ✅ Checkpoint Minggu 8
- [ ] Bisa menjelaskan kenapa gradient accumulation membagi loss dengan `grad_accum`
- [ ] Bisa menjelaskan kenapa bias dan LayerNorm tidak diberi weight decay

---

# MINGGU 9 (27 Des 2026 – 2 Jan 2027): Eksperimen ablation

### 📘 Belajar (±3 jam; sisanya untuk menjalankan eksperimen)
- [ ] Karpathy, *A Recipe for Training Neural Networks* (blog, 2019)
- [ ] Hoffmann et al. 2022 (*Chinchilla*), bagian 1–3: hubungan ukuran model dan jumlah token
- [ ] W&B *Quickstart*. Jalankan `wandb login` sekali (masukkan API key-mu sendiri).

### 🛠️ Task 21: Latih 3 ukuran model dengan anggaran token yang sama

Setiap run memakai default `5000 step × 16 × 4 × 256 ≈ 82 juta token`.

- [ ] **Step 1: Jalankan tiny, small, dan medium** (satu per satu; boleh dibiarkan berjalan semalaman)

```powershell
python -m indogpt.train.train --preset tiny --out-dir runs/tiny --wandb
python -m indogpt.train.train --preset small --out-dir runs/small --wandb
python -m indogpt.train.train --preset medium --out-dir runs/medium --wandb
```

Kalau medium kehabisan memori (`CUDA out of memory`), pakai `--batch-size 8 --grad-accum 8`. Batch efektifnya tetap sama.

Expected: val loss medium < small < tiny, dan semuanya jauh di bawah baseline bigram dalam satuan BPB.

- [ ] **Step 2: Isi tabel hasil di catatanmu** (nanti dipindah ke README)

| Model | Param | Token dilihat | Val loss | Perplexity (`exp(loss)`) | Val BPB | Token/detik | Peak VRAM | Waktu |
|---|---|---|---|---|---|---|---|---|
| Bigram byte (baseline) | 65K | – | – | – | dari Task 12 | – | – | – |
| GPT belum dilatih (step 0) | – | 0 | ±9,01 (= ln 8192) | 8192 | dari log step 0 | – | – | – |
| tiny | | | | | | | | |
| small | | | | | | | | |
| medium | | | | | | | | |

- [ ] **Step 3: Ekspor grafik val loss ketiga run dari W&B** (satu grafik gabungan) sebagai `docs/loss_curves.png` di repo.

- [ ] **Step 4: Tulis 3–5 kalimat analisis**, jawab dua pertanyaan berikut:
  1. Apakah model lebih besar selalu lebih baik pada anggaran token yang sama? Apa kata Chinchilla tentang ini?
  2. Kenapa GPT yang **belum dilatih** (step 0) sudah punya BPB lebih rendah daripada bigram byte? (Uji coba referensi: ±2,7 vs ±3,5.) Petunjuk: hitung `log2(8192) / (byte per token)`. Kamus tokenizer sudah melakukan kompresi sendiri, jadi pembanding yang jujur untuk GPT terlatih adalah baris step 0, bukan hanya bigram.

- [ ] **Step 5: Commit**

```powershell
git add docs/loss_curves.png
git commit -m "docs: kurva loss ablation tiny/small/medium"
```

---

# MINGGU 10 (3–9 Jan 2027): Sampling, Hugging Face, demo, README final

### 📘 Belajar (±2 jam)
- [ ] Dokumentasi Hugging Face Hub: *Uploading models* dan *Model Cards*
- [ ] Gradio *Quickstart* dan dokumentasi HF *Spaces Overview*

### 🛠️ Task 22: Skrip sampling

**Files:**
- Create: `scripts/sample.py`

- [ ] **Step 1: Tulis skrip**

```python
"""Bangkitkan teks dari checkpoint."""
import argparse

import torch

from indogpt.model.config import GPTConfig
from indogpt.model.gpt import GPT
from indogpt.tokenizer.bpe import BPETokenizer


def load_model(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/medium/ckpt.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--prompt", default="Pekanbaru adalah")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    idx = torch.tensor([tok.encode(args.prompt) or [tok.eot_id]], device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))
```

- [ ] **Step 2: Jalankan dengan beberapa prompt dan temperature**

Run: `python scripts/sample.py --prompt "Sungai Siak" --temperature 0.7`
Expected: teks Bahasa Indonesia yang tata bahasanya cukup lancar, walaupun faktanya sering ngawur (wajar untuk model ~17 juta parameter). Simpan 3 contoh terbaik dan 1 contoh lucu untuk README.

- [ ] **Step 3: Commit**

```powershell
git add scripts/sample.py
git commit -m "feat: skrip sampling dari checkpoint"
```

### 🛠️ Task 23: Unggah ke Hugging Face Hub + model card

**Files:**
- Create: `scripts/push_to_hub.py`
- Create: `MODEL_CARD.md`

- [ ] **Step 1: Login** (dilakukan sendiri, masukkan token HF milikmu)

Run: `hf auth login`

- [ ] **Step 2: Tulis `scripts/push_to_hub.py`**

```python
"""Unggah checkpoint, tokenizer, dan model card ke Hugging Face Hub."""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="contoh: username-hf/indogpt-mini")
    p.add_argument("--ckpt", default="runs/medium/ckpt.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--card", default="MODEL_CARD.md")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(args.ckpt, Path(tmp) / "ckpt.pt")
        shutil.copy(args.tokenizer, Path(tmp) / "tokenizer.json")
        shutil.copy(args.card, Path(tmp) / "README.md")
        api = HfApi()
        api.create_repo(args.repo_id, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id)
    print(f"https://huggingface.co/{args.repo_id}")
```

- [ ] **Step 3: Tulis `MODEL_CARD.md`.** Isi bagian bertanda `«...»` dengan hasil dari Task 21 dan 22.

```markdown
---
language: id
license: mit
tags:
  - gpt
  - from-scratch
  - indonesian
datasets:
  - wikimedia/wikipedia
---

# IndoGPT-mini

GPT kecil (~17 juta parameter) yang dilatih **dari nol** pada Wikipedia Bahasa Indonesia,
dengan tokenizer BPE byte-level buatan sendiri (vocab 8192). Proyek belajar; kode lengkap:
https://github.com/«username-github»/indogpt-mini

## Data
Subset `20231101.id` dari `wikimedia/wikipedia`, ±200 juta karakter setelah dibersihkan.

## Training
Decoder-only Transformer pre-LN (8 layer, 6 head, d=384, konteks 256), AdamW, warmup + cosine LR,
bf16, gradient accumulation, pada satu RTX 3050 Laptop 6GB. ±82 juta token.

## Hasil
| Model | Val loss | Val bits-per-byte |
|---|---|---|
| Bigram byte (baseline) | – | «hasil Task 12» |
| IndoGPT-mini (medium) | «hasil Task 21» | «hasil Task 21» |

## Keterbatasan
Model sangat kecil: sering menghasilkan fakta yang salah (halusinasi), mewarisi bias Wikipedia,
dan tidak cocok untuk penggunaan produksi.

## Cara pakai
Lihat `scripts/sample.py` di repo GitHub. Checkpoint: `ckpt.pt` (PyTorch state dict).
```

- [ ] **Step 4: Unggah**

Run: `python scripts/push_to_hub.py --repo-id USERNAME_HF/indogpt-mini`
Expected: URL model tercetak, dan halaman model di HF menampilkan model card.

- [ ] **Step 5: Commit**

```powershell
git add scripts/push_to_hub.py MODEL_CARD.md
git commit -m "feat: unggah model dan model card ke Hugging Face Hub"
```

### 🛠️ Task 24: Demo Gradio di Hugging Face Spaces

**Files:**
- Create: `space/app.py`
- Create: `space/requirements.txt`

- [ ] **Step 1: Tulis `space/app.py`**

```python
"""Demo Gradio IndoGPT-mini (untuk Hugging Face Spaces)."""
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from indogpt.model.config import GPTConfig
from indogpt.model.gpt import GPT
from indogpt.tokenizer.bpe import BPETokenizer

MODEL_REPO = os.environ.get("MODEL_REPO", "USERNAME_HF/indogpt-mini")


def _load() -> tuple[GPT, BPETokenizer]:
    ckpt = torch.load(hf_hub_download(MODEL_REPO, "ckpt.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), BPETokenizer.load(hf_hub_download(MODEL_REPO, "tokenizer.json"))


MODEL, TOK = _load()


def complete(prompt: str, max_new_tokens: float, temperature: float, top_k: float) -> str:
    idx = torch.tensor([TOK.encode(prompt) or [TOK.eot_id]])
    out = MODEL.generate(idx, int(max_new_tokens), temperature=temperature, top_k=int(top_k))
    return TOK.decode(out[0].tolist())


demo = gr.Interface(
    fn=complete,
    inputs=[
        gr.Textbox(label="Prompt", value="Pekanbaru adalah"),
        gr.Slider(10, 200, value=80, step=10, label="Jumlah token baru"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(1, 200, value=50, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Hasil"),
    title="IndoGPT-mini",
    description="GPT kecil (~17 juta parameter) yang dilatih dari nol pada Wikipedia Bahasa Indonesia.",
)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Tulis `space/requirements.txt`** (ganti `USERNAME` dengan username GitHub-mu)

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
numpy
regex
huggingface_hub
indogpt-mini @ git+https://github.com/USERNAME/indogpt-mini.git
```

- [ ] **Step 3: Uji secara lokal**

Run: `$env:MODEL_REPO = "USERNAME_HF/indogpt-mini"; python space/app.py`
Expected: buka `http://127.0.0.1:7860`, ketik prompt, dan teks muncul.

- [ ] **Step 4: Buat Space dan unggah**
  1. Buat Space di huggingface.co/new-space (SDK: Gradio, hardware: CPU basic gratis), beri nama `indogpt-mini-demo`.
  2. Di *Settings → Variables*, tambahkan `MODEL_REPO`.
  3. Unggah:

```powershell
hf upload USERNAME_HF/indogpt-mini-demo space . --repo-type space
```

Expected: Space selesai build, demo bisa dibuka publik. Simpan URL-nya untuk README.

- [ ] **Step 5: Commit**

```powershell
git add space
git commit -m "feat: demo Gradio untuk Hugging Face Spaces"
```

### 🛠️ Task 25: README final, kunci dependensi, skrip reproduksi

**Files:**
- Modify: `README.md` (ganti seluruh isi)
- Create: `requirements-lock.txt`
- Create: `scripts/reproduce.sh`

- [ ] **Step 1: Kunci versi dependensi**

Run: `uv pip freeze > requirements-lock.txt`

- [ ] **Step 2: Tulis `scripts/reproduce.sh`** (untuk Linux/WSL/Git Bash)

```bash
#!/usr/bin/env bash
# Reproduksi seluruh pipeline IndoGPT-mini dari nol.
set -euo pipefail
python -m indogpt.data.prepare_text
python scripts/train_tokenizer.py
python -m indogpt.data.encode
for preset in tiny small medium; do
  python -m indogpt.train.train --preset "$preset" --out-dir "runs/$preset"
done
python scripts/sample.py --ckpt runs/medium/ckpt.pt
```

- [ ] **Step 3: Tulis `README.md` final.** Isi bagian `«...»` dengan hasilmu. Ringkasan di atas wajib maksimal 200 kata.

````markdown
# IndoGPT-mini

![ci](https://github.com/«username-github»/indogpt-mini/actions/workflows/ci.yml/badge.svg)

**GPT Bahasa Indonesia yang dibangun dari nol**, mulai dari mesin autograd NumPy, tokenizer BPE, sampai
Transformer yang dilatih di satu GPU laptop (RTX 3050 6GB).

🤗 Model: «link HF model» · 🚀 Demo: «link HF Space»

## Ringkasan
- **Masalah:** memahami model bahasa sampai ke tingkat gradien, bukan sekadar memanggil library.
- **Pendekatan:** membangun setiap komponen sendiri dan memverifikasinya dengan test terhadap PyTorch.
  Lalu melatih GPT ~17 juta parameter pada ±82 juta token Wikipedia Indonesia.
- **Hasil:** val bits-per-byte «X», turun dari «Z» saat belum dilatih dan dari «Y» untuk baseline bigram
  (lebih rendah lebih baik). Training BPE dipercepat
  dari «A» menjadi «B» lewat algoritma incremental yang hasilnya identik.

## Hasil
«tabel dari Task 21»

![kurva loss](docs/loss_curves.png)

Contoh keluaran (prompt: "Pekanbaru adalah"):
> «contoh dari Task 22»

## Arsitektur
```mermaid
flowchart LR
    A[Wikipedia ID] --> B[prepare_text<br/>pembersihan]
    B --> C[Tokenizer BPE<br/>buatan sendiri, vocab 8192]
    C --> D[encode<br/>train.bin / val.bin]
    D --> E[train<br/>GPT, bf16, AdamW, cosine LR]
    E --> F[ckpt.pt]
    F --> G[Hugging Face Hub]
    G --> H[Demo Gradio]
```

## Apa yang dibangun dari nol
| Komponen | File | Diverifikasi dengan |
|---|---|---|
| Autograd skalar & tensor (NumPy) | `src/indogpt/autograd/` | gradien vs PyTorch |
| Tokenizer BPE byte-level + training incremental | `src/indogpt/tokenizer/bpe.py` | roundtrip, kesamaan dengan versi naif |
| GPT (attention kausal, pre-LN, weight tying) | `src/indogpt/model/gpt.py` | test masking kausal, overfit 1 batch |
| Loop training bf16 + grad accumulation | `src/indogpt/train/train.py` | smoke test |

## Reproduksi
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-lock.txt && pip install -e .
bash scripts/reproduce.sh
```
Test: `pytest -q` · Container: `docker build -t indogpt-mini-test . && docker run --rm indogpt-mini-test`

## Pelajaran
- «3 hal yang paling kamu pelajari, misal bug inisialisasi yang membuat loss awal salah»

## Keterbatasan
Model kecil: sering berhalusinasi, mewarisi bias Wikipedia, dan bukan untuk produksi.
````

- [ ] **Step 4: Minta review.** Kirim link repo ke Claude dan minta review README dan kode **sebelum** rilis.

- [ ] **Step 5: Commit, tag, push**

```powershell
git add README.md requirements-lock.txt scripts/reproduce.sh
git commit -m "docs: README final dengan hasil, arsitektur, dan reproduksi"
git tag v1.0
git push --tags
git push
```

### 🛠️ Task 26: Uji diri akhir Fase 1

- [ ] **Step 1:** Di file kosong, tanpa melihat referensi, tulis `CausalSelfAttention` dalam 20 menit. Bandingkan dengan milikmu.
- [ ] **Step 2:** Jelaskan dengan suara keras (rekam, maks. 3 menit) alur satu token dari teks mentah sampai logits. Ini latihan cerita project untuk wawancara.
- [ ] **Step 3:** Posting write-up singkat P1 di LinkedIn: apa yang dibangun, satu angka hasil, dan link repo + demo.

### ✅ Checkpoint Fase 1 / Definition of Done P1 (sesuai spec)
- [ ] Kurva loss + perplexity validasi ada di README
- [ ] Tabel ablation 3 ukuran model + baseline bigram
- [ ] Model dipublikasikan di Hugging Face dengan model card
- [ ] Demo Gradio publik di HF Spaces
- [ ] Standar repo: ringkasan ≤ 200 kata, diagram arsitektur, Dockerfile, dependensi terkunci, skrip reproduksi satu perintah, CI hijau, tabel metrik, demo, commit rapi
- [ ] Bisa menulis self-attention dari nol tanpa referensi
- [ ] 60 soal LeetCode L1 tercatat (minimal 51 non-Hard selesai)
