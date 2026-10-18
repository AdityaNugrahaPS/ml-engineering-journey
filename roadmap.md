# Roadmap ML Engineer → Big Tech (Nov 2026 – Agu 2027)

**Dibuat:** 16 Oktober 2026 · **Revisi:** mulai 1 November 2026; waktu 2027 lebih longgar (25–30 jam/minggu)
**Untuk:** Mahasiswa semester 7, Universitas Riau, lulus sekitar pertengahan 2027
**Target peran:** Machine Learning Engineer (new grad / entry-level)
**Target perusahaan:** Google, NVIDIA, AWS/Amazon, TikTok/ByteDance, dan big tech sejenis (Indonesia + luar negeri)

---

## 1. Ringkasan keputusan

| Aspek | Keputusan |
|---|---|
| Level awal | Menengah tapi berkarat. Pernah membangun project ML; perlu *refresh*, bukan mulai dari nol |
| Strategi | **Pendekatan B: bangun portofolio sampai matang dulu, lalu melamar mulai awal Juni 2027** |
| Waktu | **2026:** ±20 jam/minggu (**14 jam belajar + project**, **6 jam LeetCode**). **2027:** 25–30 jam/minggu, dengan pembagian yang sama ditambah **5–10 jam cadangan**. LeetCode berjalan paralel sejak minggu 1 |
| Skripsi | Terpisah dari portofolio. Waktu 2027 lebih longgar dan fleksibel; kelebihan jamnya dipakai sebagai **cadangan** untuk mengejar keterlambatan, bukan untuk menambah scope project |
| Lokasi | Indonesia + luar negeri (Singapura dan lainnya) |
| Flagship | **Computer Vision real-time dengan optimasi GPU** (TensorRT + Triton), cocok dengan laptop RTX 3050 6GB |
| Hardware | Laptop dengan NVIDIA RTX 3050 6GB Laptop GPU dan RAM ±20GB, ditambah GPU gratis Kaggle/Colab bila perlu |

### Konteks rekrutmen (hasil riset Okt 2026)

- **Magang big tech umumnya mensyaratkan masih kuliah setelah magang selesai** (contoh: lowongan magang NVIDIA Spring 2027 mensyaratkan terdaftar di program S1/S2 selama 8–12 bulan magang). Karena kamu lulus pertengahan 2027, targetnya adalah **posisi new grad / entry-level**, bukan magang musim panas 2027.
- **TikTok/ByteDance Singapura** membuka lowongan *"Machine Learning Engineer Graduate – 2027 Start"* (Search, Recommendation, E-Commerce). Lamaran diproses bergulir dan dibatasi maksimal 2 posisi per pelamar.
- **Google** membuka *Software Engineer, University Graduate*. Lowongan di Singapura memprioritaskan pelamar yang sudah punya izin kerja di sana. Wawancaranya sangat berat di DSA.
- **AWS**: lowongan SDE early-career di Singapura yang ditemukan mensyaratkan izin kerja Singapura. Program graduate 2027 ada di Sydney.
- **Konsekuensi pendekatan B:** sebagian kuota "2027 Start" bisa terisi sebelum Juni 2027. Mitigasinya ada di bagian 7.
- **Format wawancara MLE big tech:** coding DSA, ML system design, ML fundamentals mendalam, dan behavioral. Untuk new grad, porsi DSA paling besar.
- **Portofolio yang dihargai:** 3–5 project yang mendalam, **di-deploy**, terdokumentasi, dan hasilnya terukur. Project klise (Titanic, notebook sentimen dasar) tidak membedakan kandidat.

---

## 2. Timeline & fase

| Fase | Periode | Fokus | Output |
|---|---|---|---|
| **0. Refresh** | 1 – 28 Nov 2026 (4 mgg) | Python engineering, NumPy/pandas, matematika ML, ML klasik, Git/Linux/Docker | P1a selesai |
| **1. Deep Learning** | 29 Nov 2026 – 9 Jan 2027 (6 mgg) | PyTorch mendalam, Transformer, training di GPU | **P1 selesai** |
| **2. MLOps & Recsys** | 10 Jan – 6 Mar 2027 (8 mgg) | Serving, CI/CD, tracking, orkestrasi, monitoring, cloud | **P2 selesai** |
| **3. Flagship CV** | 7 Mar – 1 Mei 2027 (8 mgg) | Deteksi + tracking, TensorRT, Triton, benchmark | **P3 selesai** |
| **4. LLM Systems** | 2 Mei – 5 Jun 2027 (5 mgg) | Serving LLM, kuantisasi, RAG + evaluasi | **P4 selesai** |
| **5. Interview & Apply** | 6 Jun – Agu 2027 | ML system design, ML fundamentals, mock, melamar | CV final + lamaran |

### Template minggu (±20 jam, Nov–Des 2026)

| Hari | Aktivitas |
|---|---|
| Senin–Jumat | 1 jam LeetCode (pagi) + 1.5–2 jam belajar/project (malam) |
| Sabtu | 4–5 jam blok project (pekerjaan terberat) |
| Minggu | 1 jam review LeetCode (spaced repetition) + 1 jam menulis README/catatan. **Sisanya libur** |

**Mulai Januari 2027 (25–30 jam/minggu):** pembagian di atas tetap, ditambah **5–10 jam cadangan** per minggu (misalnya 1 jam ekstra tiap malam atau setengah hari Minggu). Cadangan dipakai berurutan untuk: (1) mengejar task yang tertinggal, (2) soal ⭐ Hard yang ditunda, (3) bonus kontribusi open source. Kalau semuanya sesuai jadwal, jam cadangan boleh dipakai istirahat.

---

## 3. Kurikulum per fase

### Fase 0: Refresh (1 – 28 Nov 2026)

| Minggu | Belajar | Project | LeetCode |
|---|---|---|---|
| 1 (1–7 Nov) | Setup: WSL2, Python env (uv/conda), PyTorch + CUDA terverifikasi di RTX 3050, Git/GitHub, MIT *Missing Semester* (shell, Git). Refresh NumPy/pandas | Buat repo `indogpt-mini`, struktur folder, CI kosong | Arrays & Hashing |
| 2 (8–14 Nov) | Aljabar linear (vektor, perkalian matriks, eigen, SVD) + kalkulus (chain rule, gradien). Tonton Karpathy *micrograd* | P1a: mesin autograd (scalar) | Arrays & Hashing, Two Pointers |
| 3 (15–21 Nov) | Probabilitas & statistik (distribusi, MLE, Bayes, uji hipotesis). Refresh ML klasik: bias-variance, regularisasi, cross-validation, metrik | P1a: MLP di atas autograd, gradient check vs PyTorch dengan pytest, CI GitHub Actions | Sliding Window |
| 4 (22–28 Nov) | Python engineering: OOP, type hints, `pyproject.toml`, pytest. Docker dasar | P1a: README + selesai | Stack |

**Sumber:** *Mathematics for Machine Learning* (Deisenroth et al., PDF gratis), *Hands-On Machine Learning* (Géron), MIT *Missing Semester*, 3Blue1Brown *Essence of Linear Algebra*, StatQuest.

**Checkpoint Fase 0:**
- [ ] PyTorch berjalan di GPU lokal (`torch.cuda.is_available()` = True)
- [ ] Bisa menurunkan gradien MSE dan cross-entropy di atas kertas
- [ ] P1a lulus semua test di CI

### Fase 1: Deep Learning (29 Nov 2026 – 9 Jan 2027)

- Karpathy *Neural Networks: Zero to Hero*: makemore (bigram → MLP → BatchNorm → backprop manual → WaveNet), *Let's build GPT*, *Let's build the GPT Tokenizer*
- Konsep wajib paham: inisialisasi bobot, normalisasi, residual, attention, optimizer (SGD/Adam/AdamW), learning rate schedule, mixed precision, gradient accumulation, overfitting & regularisasi
- **Sumber pendukung:** d2l.ai, catatan CS231n

**Checkpoint Fase 1:**
- [ ] Bisa menulis self-attention dari nol tanpa melihat referensi
- [ ] P1 selesai sesuai *definition of done*

### Fase 2: MLOps & Recsys (10 Jan – 6 Mar 2027)

- Serving (FastAPI), containerisasi (Docker Compose), CI/CD, experiment tracking & model registry (MLflow), orkestrasi (Prefect), validasi data, monitoring (Prometheus + Grafana), drift (Evidently), deploy ke cloud (GCP Cloud Run atau AWS free tier)
- Recsys: retrieval vs ranking, matrix factorization, two-tower, ANN search (FAISS), metrik offline (Recall@K, NDCG@K), cold start
- **Sumber utama:** Made With ML, *Designing Machine Learning Systems* (Chip Huyen)
- **Pendukung:** DataTalksClub MLOps Zoomcamp, kursus *Recommendation Systems* dari Google (developers.google.com), blog Eugene Yan

**Checkpoint Fase 2:**
- [ ] Bisa menjelaskan arsitektur recsys dua tahap (retrieval → ranking) di papan tulis dalam 10 menit
- [ ] P2 ter-deploy dan dapat diakses publik

### Fase 3: Flagship Computer Vision (7 Mar – 1 Mei 2027)

- Deteksi objek (arsitektur one-stage vs DETR), mAP, augmentasi data, labeling, multi-object tracking (ByteTrack)
- Optimasi inference: ONNX, TensorRT (FP16, INT8 + kalibrasi), profiling dengan Nsight Systems, batching
- Serving: NVIDIA Triton Inference Server (dynamic batching, ensemble model, gRPC)
- **Sumber utama:** repo *triton-inference-server/tutorials* di GitHub, dokumentasi TensorRT, CS231n
- **Pendukung:** NVIDIA Deep Learning Institute, paper ByteTrack, library `supervision`

**Checkpoint Fase 3:**
- [ ] Tabel benchmark FP32 vs FP16 vs INT8 lengkap dengan angka nyata
- [ ] P3 selesai, write-up terpublikasi

### Fase 4: LLM Systems (2 Mei – 5 Jun 2027)

- Mekanisme inference LLM: KV cache, batching, paged attention, kuantisasi (AWQ, GPTQ, GGUF)
- RAG: chunking, embedding, retrieval, reranking, evaluasi (recall retrieval, faithfulness)
- **Sumber utama:** dokumentasi vLLM, *AI Engineering* (Chip Huyen)
- **Pendukung:** Hugging Face LLM Course

### Fase 5: Interview & Apply (6 Jun – Agu 2027)

- ML system design: latihan 8–10 kasus (feed ranking, rekomendasi video, search, deteksi fraud, ads CTR, deteksi konten berbahaya, ETA prediction, visual search)
- ML fundamentals: bias-variance, regularisasi, metrik, loss function, trade-off model, debugging training
- Behavioral: 6–8 cerita format STAR yang diambil dari P1–P4
- Mock interview: dengan teman, komunitas, atau platform mock
- **Sumber:** *Machine Learning System Design Interview* (Ali Aminian & Alex Xu), *Introduction to ML Interviews Book* (Chip Huyen, gratis), Tech Interview Handbook

---

## 4. Portofolio: 4 project bertahap

### Standar industri untuk SEMUA repo

- [ ] README: masalah, pendekatan, dan hasil (dengan angka) tertulis dalam 200 kata pertama
- [ ] Diagram arsitektur
- [ ] Bisa dijalankan ulang: Dockerfile / `docker compose`, dependensi dikunci versinya, `Makefile` atau skrip satu perintah
- [ ] Test (pytest) + CI GitHub Actions hijau
- [ ] Tabel metrik / benchmark
- [ ] Demo: video, GIF, atau Hugging Face Space
- [ ] Riwayat commit rapi (pesan commit bermakna)

### P1 · IndoGPT-mini: Deep Learning dari Nol (Fase 0–1)

| Tahap | Isi |
|---|---|
| 1a | Mesin autograd NumPy → MLP. Gradien diverifikasi terhadap PyTorch lewat pytest, CI di GitHub Actions |
| 1b | Tokenizer BPE yang dilatih sendiri pada korpus Wikipedia Bahasa Indonesia |
| 1c | Transformer bergaya GPT di PyTorch (±10–30 juta parameter), dilatih di RTX 3050: bf16, gradient accumulation, cosine LR, checkpoint, logging W&B |

**Definition of done:**
- [ ] Kurva loss + perplexity validasi
- [ ] Tabel ablation (minimal: 3 ukuran model vs perplexity)
- [ ] Model dipublikasikan di Hugging Face dengan model card
- [ ] Demo Gradio

**Nilai jual:** bekal untuk wawancara ML depth (backprop, attention), dan bukti bahwa kamu paham isi model, bukan sekadar memanggil library.

### P2 · Recommender System + Pipeline MLOps (Fase 2)

| Komponen | Isi |
|---|---|
| Data | MovieLens |
| Model (bertingkat) | Baseline popularitas → matrix factorization → **two-tower retrieval** + indeks FAISS → re-ranker LightGBM |
| Pipeline | Validasi data → training → MLflow (tracking + registry) → orkestrasi Prefect → retraining terjadwal |
| Serving | FastAPI + cache Redis, Docker Compose, deploy ke cloud |
| Operasional | CI/CD, Prometheus + Grafana, deteksi drift Evidently, load test Locust |

**Definition of done:**
- [ ] `docker compose up` menjalankan seluruh sistem
- [ ] Tabel Recall@K & NDCG@K: setiap model dibanding baseline
- [ ] Laporan load test p50/p95/p99 (target p95 < 50 ms)
- [ ] Endpoint publik + dashboard monitoring

**Nilai jual:** TikTok, Netflix, Amazon, Shopee. Project ini menunjukkan kematangan operasional, yaitu kemampuan yang membedakan MLE dari data scientist.

### P3 ⭐ FLAGSHIP · Analitik Lalu Lintas Real-Time di GPU (Fase 3)

| Komponen | Isi |
|---|---|
| Data | Rekam sendiri video jalan raya di Pekanbaru (dataset unik) + label ±1.500 frame dengan CVAT/Label Studio. Ditambah dataset publik (UA-DETRAC / BDD100K). Kelas: motor, mobil, bus, truk, sepeda, pejalan kaki |
| Model | Fine-tuning detektor modern (RT-DETR atau keluarga YOLO; **cek lisensi**, misal Ultralytics berlisensi AGPL-3.0), tracking ByteTrack, penghitungan via garis virtual |
| Privasi | Blur otomatis wajah & pelat nomor |
| Optimasi (inti) | PyTorch → ONNX → **TensorRT FP32/FP16/INT8** (kalibrasi INT8), profiling Nsight Systems |
| Serving | **NVIDIA Triton Inference Server**: dynamic batching, ensemble (pre-process → model → post-process), multi-stream kamera, dashboard live |
| Data flywheel | Frame dengan confidence rendah dicatat → dilabel ulang → model dilatih ulang |

**Definition of done:**
- [ ] Tabel benchmark: **mAP vs latensi vs FPS vs VRAM** untuk tiap presisi di RTX 3050
- [ ] Write-up teknis bergaya *"Dari X FPS ke Y FPS: …"* (blog / Medium / situs pribadi)
- [ ] Video demo multi-stream
- [ ] Dockerfile yang bisa dijalankan ulang

**Nilai jual:** NVIDIA (TensorRT dan Triton adalah stack mereka sendiri), plus Google dan AWS (CV di produksi). Konteks smart city Indonesia membuat ceritanya menarik saat wawancara.

### P4 · LLM Inference & RAG Lab (Fase 4, sengaja dibuat kecil)

| Komponen | Isi |
|---|---|
| Serving | LLM open-weights kecil: 1–3B FP16 atau 7–8B 4-bit agar muat di 6GB VRAM. vLLM (di WSL2) atau llama.cpp |
| Benchmark | FP16 vs AWQ/GPTQ vs GGUF: token/detik, time-to-first-token, memori, penurunan kualitas |
| RAG | Dokumen berbahasa Indonesia (misal peraturan akademik) |
| Evaluasi | ±100 pertanyaan berlabel; recall retrieval + faithfulness jawaban |

**Definition of done:**
- [ ] Laporan benchmark dengan grafik
- [ ] API RAG + skrip evaluasi yang bisa dijalankan ulang

**Nilai jual:** AWS, Google, NVIDIA, karena topik LLM serving paling banyak dicari di 2026.

### Bonus opsional (Fase 4–5)

- 1–2 pull request ke library open source yang dipakai (Evidently, supervision, vLLM, Triton, dsb.)

---

## 5. LeetCode terkurasi (±130 soal + ±30 ML coding)

Dasarnya **NeetCode 150**, tapi **tidak semua dikerjakan**. Target: menguasai pola, bukan mengejar jumlah. Ritme: 6 jam/minggu ≈ 5–7 soal baru + review.

| Tahap | Periode | Topik | ±Soal |
|---|---|---|---|
| **L1** | 1 Nov 2026 – 9 Jan 2027 | Arrays & Hashing, Two Pointers, Sliding Window, Stack, Binary Search, Linked List, Trees | 60 |
| **L2** | 10 Jan – pertengahan Apr 2027 | Heap / Priority Queue, Backtracking, Tries, Graphs (BFS/DFS), Intervals, Greedy, 1-D DP | 50 |
| **L3** | pertengahan Apr – 5 Jun 2027 | 2-D DP (pilihan), Advanced Graphs (Dijkstra, topological sort, Union-Find), 2–3 soal Bit Manipulation | 20 |
| **L4** | Jun – Agu 2027 | Soal per perusahaan (Google, TikTok), weekly contest, mock 45 menit | Ulang soal lemah |

**Dilewati / prioritas rendah** (jarang keluar untuk MLE new grad):
- Math & Geometry
- Bit Manipulation tingkat lanjut
- Segment Tree / Fenwick Tree
- DP bitmask / digit DP
- Mayoritas soal Hard (cukup ±10 Hard secara total)

### ML coding: ±30 soal di Deep-ML (mulai tahap L2)

Implementasikan dengan NumPy, tanpa library ML:
- Regresi linear & logistik (gradient descent)
- k-means, KNN, PCA
- Split decision tree (gini / entropy)
- Softmax + cross-entropy, backprop MLP, batch norm
- Conv2d, self-attention
- **IoU & Non-Maximum Suppression** (langsung relevan dengan P3)
- Metrik: precision, recall, F1, AUC

### Metode per soal

1. Batasi waktu 25–30 menit. Kalau buntu, baca hint lalu solusi.
2. Kerjakan ulang **tanpa melihat** setelah 1, 3, dan 7 hari.
3. Catat di spreadsheet: nama soal, pola, kesalahan, tanggal review berikutnya.
4. Selalu sebutkan kompleksitas waktu & ruang, sambil menjelaskan dengan suara (simulasi wawancara).

---

## 6. Strategi lamaran (Fase 5, mulai awal Juni 2027)

- **CV:** 1 halaman. Setiap poin memakai formula *"Mencapai X, diukur dengan Y, dengan melakukan Z"*, dan setiap project memuat angka (FPS, latensi, NDCG, perplexity).
- **Target utama:** posisi new grad / entry-level MLE atau SWE-ML di Google, TikTok/ByteDance, NVIDIA, AWS/Amazon, dan big tech lain yang membuka lowongan bergulir.
- **Cek syarat izin kerja** di setiap lowongan. Beberapa lowongan Singapura memprioritaskan pelamar yang sudah memiliki izin kerja.
- **Referral:** bangun jaringan LinkedIn sejak Fase 2. Bagikan write-up setiap project yang selesai, dan hubungi alumni atau engineer di perusahaan target.
- **Cadangan (opsional):** perusahaan tech besar Indonesia/Asia Tenggara (GoTo, Traveloka, Grab, Shopee/Sea, Agoda) memakai portofolio yang sama.

---

## 7. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| Kuota graduate "2027 Start" terisi sebelum Juni 2027 (konsekuensi pendekatan B) | Perbarui GitHub + LinkedIn setiap project selesai. Pantau lowongan bergulir; kalau ada lowongan sangat cocok sebelum Juni, pertimbangkan melamar lebih awal dengan portofolio yang sudah ada |
| Skripsi atau urusan lain menyita waktu di 2027 | Pakai jam cadangan dulu. Kalau masih kurang, P4 boleh dipangkas ke bagian benchmark saja. Flagship P3 tetap prioritas |
| Scope creep (project tak pernah selesai) | Patuhi *definition of done*. Fitur tambahan masuk daftar "nanti" |
| Burnout | Minimal 1 hari libur penuh per minggu |
| GPU 6GB tidak cukup | Pakai Kaggle/Colab untuk training besar; laptop untuk inference & benchmark |

---

## 8. Sumber riset

- [TikTok – Machine Learning Engineer Graduate (TikTok Search) – 2027 Start](https://www.themuse.com/jobs/tiktok/machine-learning-engineer-graduate-tiktok-search-2027-start)
- [TikTok – Machine Learning Engineer Graduate (TikTok Shop SEA) – 2027 Start](https://www.themuse.com/jobs/tiktok/machine-learning-engineer-graduate-tiktok-shop-sea-2027-start)
- [NVIDIA Spring 2027 Internships: Developer and Performance Technology](https://jobs.anitab.org/companies/nvidia/jobs/90492712-nvidia-spring-2027-internships-developer-and-performance-technology)
- [Google – Software Engineer, University Graduate](https://jobs.anitab.org/companies/google-24698/jobs/62025542-software-engineer-university-graduate)
- [Amazon – SDE, AWS Global Demand Organization (Singapura)](https://amazon.jobs/en/jobs/2767027/software-development-engineer-sde-aws-global-demand-organization)
- [AWS – Software Development Graduate 2027 (Sydney)](https://jobs.anitab.org/companies/amazon-3-60ad394d-c673-4474-9694-344b0cae748f/jobs/84555614-software-development-graduate-aws-2027-sydney)
- [GoTo – Software Engineer (AI/ML Engineering) Internship](https://jobs.lever.co/GoToGroup/e84b33d1-f97d-45dd-ae66-ec699ae9e84f)
- [Google ML Engineer interview experiences 2026 (Dataford)](https://dataford.io/interview-guides/google/machine-learning-engineer/experiences)
- [Google Machine Learning Engineer guide (Codemia)](https://codemia.io/guides/google-machine-learning-engineer)
- [The ML portfolio that actually gets you hired in 2026](https://letsdatascience.com/blog/the-ml-portfolio-that-actually-gets-you-hired-in-2026)
- [Machine Learning Engineer Portfolio guide (Interview Kickstart)](https://interviewkickstart.com/blogs/articles/machine-learning-engineer-portfolio)
