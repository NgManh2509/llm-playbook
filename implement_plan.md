# 🚀 Kế Hoạch Triển Khai: `llm-engineering-playbook`

> **Mục tiêu dự án:** Xây dựng một repository "sandbox" chuẩn mực (Modular Monorepo) kết hợp giữa lý thuyết cốt lõi, các bài thí nghiệm độc lập (Labs) và một hệ thống Backend API sẵn sàng cho môi trường thực tế (Production-ready).  
> **Kiến trúc chủ đạo:** Hybrid Cloud LLM — Tận dụng tốc độ phản hồi cực nhanh trên phần cứng LPU của **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) kết hợp với khả năng tạo Vector Embedding & xử lý ngữ cảnh dài của **Google Gemini** (`text-embedding-004`, `gemini-2.5-flash`).

---

## 🏗️ 1. Kiến trúc Tổng thể & Cấu trúc Thư mục

### 1.1. Sơ đồ Luồng xử lý (System Architecture)

```mermaid
flowchart TD
    User([User Request]) --> FastAPI[FastAPI Gateway / SSE Stream]
    FastAPI --> Cache{Semantic Cache\nRedis / pgvector}
    Cache -- Hit (>0.95) --> FastAPI
    Cache -- Miss --> Router[Intent Router\nGroq: llama-3.1-8b-instant]

    Router -- "Chitchat / Tác vụ đơn giản" --> FastLLM[Groq: llama-3.1-8b-instant]
    Router -- "Tra cứu tài liệu (RAG)" --> Retriever[Vector Search: pgvector\nEmbedding: Gemini text-embedding-004]
    Router -- "Suy luận / Gọi hàm" --> Agent[Agent / Tool Calling\nGroq: llama-3.3-70b-versatile]

    Retriever --> Rerank[Hybrid Filter & Context Builder]
    Rerank --> HeavyLLM[Grounded Generation\nGroq: llama-3.3-70b-versatile]

    FastLLM & HeavyLLM & Agent -. "Lỗi 429 Rate Limit / 503" .-> Fallback[Fallback Engine\nGemini 2.5 Flash]
    Fallback --> Response([Validated JSON / SSE Stream])
    FastLLM & HeavyLLM & Agent --> Response
```

### 1.2. Cấu trúc Thư mục Chuẩn

```text
llm-engineering-playbook/
├── docs/                           # Ghi chú lý thuyết & sơ đồ kiến trúc
│   ├── 01-parameters-and-tokens.md
│   ├── 02-rag-architecture.md
│   └── 03-agentic-patterns.md
├── experiments/                    # Các bài lab độc lập (chạy trực tiếp bằng Python)
│   ├── 01_basics/                  # Streaming, System Prompt, Benchmark tốc độ LPU vs GPU
│   ├── 02_structured_outputs/      # Ép kiểu JSON với Pydantic & Intent Classification
│   ├── 03_rag_pipeline/            # Chunking, Gemini Embedding, pgvector Retrieval
│   ├── 04_function_calling/        # Tool use & tích hợp API bên ngoài
│   └── 05_agent_workflows/         # Prompt Chaining, Routing, Evaluator-Optimizer
├── src/                            # Production API (FastAPI)
│   ├── core/                       # Config, Multi-Provider LLM Client, Retry/Fallback
│   ├── services/                   # Logic nghiệp vụ (RAG service, Agent service)
│   ├── routers/                    # REST & Streaming SSE Endpoints
│   └── main.py                     # Entrypoint của FastAPI
├── data/                           # Tài liệu mẫu (.pdf, .md) để test RAG
├── docker-compose.yml              # Khởi chạy FastAPI + PostgreSQL (pgvector) + Redis
├── .env.example                    # Template chứa GROQ_API_KEY, GEMINI_API_KEY, DB_URL
├── IMPLEMENTATION_PLAN.md          # File kế hoạch triển khai này
└── pyproject.toml                  # Quản lý thư viện với uv
```

---

## 📅 2. Lộ trình Triển khai 5 Giai đoạn (14 Ngày)

| Giai đoạn | Thời lượng | Trọng tâm | Công nghệ / Model | Sản phẩm đầu ra (Deliverables) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Ngày 1 - 2 | Nền tảng & Multi-Provider Client | `uv`, `groq`, `google-genai`, `tenacity` | Module gọi API dùng chung + Cơ chế tự động chuyển đổi (Fallback) khi chạm Rate Limit. |
| **Phase 2** | Ngày 3 - 4 | Kiểm soát Đầu ra & Định tuyến | `pydantic`, `llama-3.1-8b-instant` | Bộ bóc tách dữ liệu JSON chuẩn xác 100% & Bộ phân loại ý định (Intent Router) dưới 200ms. |
| **Phase 3** | Ngày 5 - 8 | Hệ thống Hybrid RAG | `Docker`, `pgvector`, `text-embedding-004`, `llama-3.3-70b` | Pipeline đọc tài liệu, băm nhỏ (Chunking), tìm kiếm ngữ nghĩa và trả lời kèm trích dẫn. |
| **Phase 4** | Ngày 9 - 11 | Tool Calling & Agentic Workflows | Native Tool Use, ReAct Pattern, Evaluator Loop | Agent tự động tra cứu Database/API bên ngoài và tự kiểm tra lỗi trước khi trả lời. |
| **Phase 5** | Ngày 12 - 14 | Đóng gói Production API | `FastAPI`, `SSE Streaming`, `Redis` | RESTful API hoàn chỉnh có Streaming từng chữ, bộ nhớ đệm (Caching) và tài liệu `README.md`. |

---

## 🛠️ 3. Checklist Công việc Chi tiết từng Giai đoạn

### Phase 1: Dựng Khung & Tích hợp Đa Mô hình (Ngày 1 - 2)

#### 📚 Lý thuyết cần nắm (`docs/01-parameters-and-tokens.md`)
- [ ] Cơ chế **Tokenization** (Byte-Pair Encoding) và lý do văn bản tiếng Việt tốn nhiều token hơn tiếng Anh.
- [ ] Sự khác biệt kiến trúc phần cứng giữa **LPU (Groq)** và **GPU truyền thống** đối với tốc độ sinh token (Time-to-First-Token & Tokens/sec).
- [ ] Phân phối xác suất trong quá trình Sampling: Khi nào chỉnh `temperature = 0.0` (Code, RAG, Trích xuất dữ liệu) và khi nào chỉnh `0.7 - 0.9` (Sáng tạo).
- [ ] Chiến lược xử lý lỗi `HTTP 429 (Rate Limit)` bằng thuật toán **Exponential Backoff**.

#### 💻 Thực hành & Code (`experiment/01_basics/` & `src/llm_playbook/core/`)
- [x] Khởi tạo dự án bằng `uv init` và cài đặt các gói: `groq`, `httpx`, `pydantic`, `tenacity`, `python-dotenv`, `chromadb`.
- [x] Viết script benchmark đo thời gian phản hồi thực tế giữa `openai/gpt-oss-20b` và `openai/gpt-oss-120b` (Groq).
- [x] Xây dựng class `UnifiedLLMClient` trong `src/llm_playbook/core/llm_client.py`:
  - Luồng chính gọi Groq API.
  - Bọc decorator `@retry` của `tenacity` và tự động chuyển hướng (Fallback) khi gặp lỗi.

---

### Phase 2: Ép kiểu JSON (Structured Outputs) & Bộ định tuyến (Ngày 3 - 4)

#### 📚 Lý thuyết cần nắm
- [x] Cơ chế **Constrained Decoding** (Giải mã có ràng buộc) giúp LLM tuân thủ 100% JSON Schema.
- [x] Kỹ thuật **Zero-shot** vs **Few-shot Prompting** khi định nghĩa quy tắc phân loại.
- [x] Các lỗ hổng **Prompt Injection** cơ bản và cách cô lập dữ liệu đầu vào của người dùng bằng thẻ phân cách (Delimiters).

#### 💻 Thực hành & Code (`experiment/02_structured_outputs/`)
- [x] **Bài Lab 2.1 (Data Extractor):** Xây dựng trình phân tích CV trả về đối tượng `Pydantic BaseModel` (sử dụng `response_format={"type": "json_object"}` trên Groq kết hợp `model_validate_json`).
- [x] **Bài Lab 2.2 (Semantic Intent Router):** Dùng `openai/gpt-oss-20b` phân loại câu hỏi người dùng vào 3 nhánh với độ trễ thấp:
  - `CHITCHAT`: Trả lời trực tiếp bằng model 20B.
  - `RAG_SEARCH`: Chuyển xuống pipeline truy xuất tài liệu nội bộ.
  - `COMPLEX_REASONING`: Chuyển cho model `openai/gpt-oss-120b`.

---

### Phase 3: Xây dựng Hybrid RAG Pipeline với `ChromaDB` & `Gemini Embedding` (Ngày 5 - 8)

#### 📚 Lý thuyết cần nắm (`docs/02-rag-architecture.md`)
- [x] Không gian Vector (Embedding Space) và công thức đo độ tương đồng **Cosine Similarity**:
  $$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- [x] Các chiến thuật chia nhỏ văn bản (**Chunking Strategies**): Fixed-size, Recursive Character, và Markdown Header Splitting; tầm quan trọng của `chunk_overlap` (10% - 15%).
- [x] Hiện tượng *"Lost in the Middle"* và lý do cần kết hợp tìm kiếm từ khóa (BM25) với tìm kiếm ngữ nghĩa (Vector Search).

#### 💻 Thực hành & Code (`experiment/03_rag_pipeline/`)
- [x] Cấu hình `ChromaDB PersistentClient` lưu trữ cục bộ tại `./data/chroma_db`.
- [x] **Script Ingestion (`01_chroma_basic.py`):**
  - Đọc tài liệu `.md` từ thư mục `data/company_policy.md`.
  - Chia đoạn theo tiêu đề Markdown (`## `).
  - Gọi **Gemini Embedding API** (`models/gemini-embedding-2`) tạo vector 3072 chiều và lưu vào ChromaDB.
- [x] **Script Retrieval & Grounded Generation (`02_query_rag.py`):**
  - Nhúng câu hỏi của người dùng thành vector 3072 chiều.
  - Truy vấn Top đoạn văn gần nhất trong `ChromaDB`.
  - Đưa ngữ cảnh vào System Prompt của **Groq** với quy tắc nghiêm ngặt: *"Chỉ trả lời dựa trên ngữ cảnh được cung cấp và bắt buộc trích dẫn nguồn theo định dạng [1], [2]"*.

---

### Phase 4: Function Calling & Agentic Patterns (Ngày 9 - 11)

#### 📚 Lý thuyết cần nắm (`docs/03-agentic-patterns.md`)
- [ ] Bản chất của **Tool Calling / Function Calling**: LLM không tự chạy hàm mà chỉ sinh ra cấu trúc JSON chứa tên hàm và tham số; Backend thực thi hàm và trả kết quả lại cho LLM.
- [ ] Vòng lặp suy luận **ReAct** (*Reason $\rightarrow$ Act $\rightarrow$ Observe*).
- [ ] Mẫu thiết kế **Evaluator-Optimizer** (Generator-Critic Loop) để tự động nâng cao chất lượng đầu ra.

#### 💻 Thực hành & Code (`experiments/04_function_calling/` & `05_agent_workflows/`)
- [ ] **Bài Lab 4 (Tool Use):** Định nghĩa 2 công cụ Python thực tế (`get_database_metrics` và `calculate_loan_schedule`), truyền JSON Schema vào tham số `tools` của Groq API và lập trình vòng lặp xử lý `tool_calls`.
- [ ] **Bài Lab 5 (Self-Reflection Loop):**
  - **Generator (`llama-3.3-70b-versatile`):** Viết một hàm Python giải quyết bài toán kèm Unit Test.
  - **Evaluator (`llama-3.1-8b-instant` hoặc `gemini-2.5-flash`):** Chấm điểm bảo mật và hiệu năng theo schema Pydantic. Nếu điểm $< 8/10$, gửi nhận xét ngược lại cho Generator viết lại (tối đa 3 vòng lặp).

---

### Phase 5: Đóng gói FastAPI, Streaming & Hoàn thiện Repo (Ngày 12 - 14)

#### 📚 Lý thuyết cần nắm
- [ ] Giao thức **Server-Sent Events (SSE)** giúp stream từng token về phía Client theo thời gian thực.
- [ ] Xử lý bất đồng bộ (Async I/O) trong Python với `AsyncGroq` để chịu tải cao.
- [ ] Cơ chế **Semantic Caching** giúp tiết kiệm chi phí token và giảm độ trễ xuống vài mili-giây.

#### 💻 Thực hành & Code (`src/`)
- [ ] Chuyển đổi các module từ `experiments/` thành các Service sạch sẽ trong `src/services/`.
- [ ] Xây dựng endpoint `POST /api/v1/chat/stream` sử dụng `StreamingResponse` của FastAPI với `media_type="text/event-stream"`.
- [ ] Tích hợp logging theo dõi số lượng `prompt_tokens`, `completion_tokens` và thời gian thực thi của mỗi request.
- [ ] Hoàn thiện file `README.md` ở thư mục gốc gồm: Giới thiệu kiến trúc, bảng kết quả Benchmark thực tế, và hướng dẫn khởi chạy nhanh bằng 1 lệnh `docker compose up --build`.

---

## 🌿 4. Lộ trình Git Commit Chuẩn (Conventional Commits)

Tuân thủ lộ trình commit dưới đây sẽ giúp lịch sử Git của bạn rõ ràng, chuyên nghiệp khi nhà tuyển dụng hoặc đồng nghiệp xem xét mã nguồn:

```bash
# Phase 1
git commit -m "chore: initialize project structure with uv and docker-compose for pgvector"
git commit -m "docs(theory): add notes on tokenization, sampling parameters, and LPU vs GPU"
git commit -m "feat(core): implement unified LLM client with Groq primary and Gemini fallback"

# Phase 2
git commit -m "feat(lab-02): add Pydantic structured output extractor with Groq JSON mode"
git commit -m "feat(router): build sub-200ms semantic intent router using Llama-3.1-8B"

# Phase 3
git commit -m "feat(rag): implement document chunking and Gemini embedding ingestion to pgvector"
git commit -m "feat(rag): add cosine similarity retrieval and grounded generation with citations"

# Phase 4
git commit -m "feat(tools): implement native function calling loop with Groq API"
git commit -m "feat(agents): add Evaluator-Optimizer reflection workflow experiment"

# Phase 5
git commit -m "feat(api): expose FastAPI endpoints with real-time SSE token streaming"
git commit -m "docs(readme): finalize system architecture diagrams and benchmark report"
```

---

## ⚙️ 5. Các Lệnh Khởi Tạo Nhanh (Quickstart Commands)

Sử dụng các lệnh sau trên Terminal để bắt đầu ngay **Phase 1**:

```bash
# 1. Tạo thư mục dự án và cấu trúc con
mkdir -p llm-engineering-playbook/{docs,data,src/{core,services,routers},experiments/{01_basics,02_structured_outputs,03_rag_pipeline,04_function_calling,05_agent_workflows}}
cd llm-engineering-playbook

# 2. Khởi tạo môi trường Python với uv
uv init
uv add groq google-genai openai pydantic fastapi uvicorn[standard] python-dotenv tenacity psycopg[binary] pgvector

# 3. Tạo file biến môi trường mẫu
cat <<EOT >> .env.example
GROQ_API_KEY=gsk_your_groq_api_key_here
GEMINI_API_KEY=AIzaSy_your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_sandbox
EOT
cp .env.example .env
```