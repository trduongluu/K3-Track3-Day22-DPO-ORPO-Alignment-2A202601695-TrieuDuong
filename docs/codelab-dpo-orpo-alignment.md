---
# Metadata của Lab. Điền giá trị bên phải dấu ":".
title: "Preference Alignment Lab: Dạy mô hình biết ưu tiên với DPO & ORPO" # Tên Codelab
description: "Hoàn thiện một repo skeleton kiểu production: validate dữ liệu preference, cài đặt hàm mất mát DPO/ORPO, đo pairwise accuracy và xuất báo cáo metrics.json." # Tóm tắt ngắn cho học viên
author: "VinUni Codelab" # Tác giả
duration: 240 # 4 tiếng (240 phút)
category: "AI Engineering" # Nhóm nội dung
updated: "2026-08-24" # Tự điền ngày hiện tại (YYYY-MM-DD)
day: "22" # Day của cohort
sequence: 1 # Thứ tự hiển thị trong cùng Day (số nhỏ hơn hiện trước)
keywords: ["DPO", "ORPO", "Preference Alignment", "LLM", "Python", "Pytest"]
level: "intermediate" # beginner hoặc intermediate
requiresSubmission: true # true nếu học viên cần nộp bài
workMode: "individual" # Bắt buộc: individual hoặc team
overview: # Tùy chọn; điền đủ các phần dưới để hiện Bản đồ Lab
  summary: "Bạn nhận một repo cố tình để trống các khối TODO(student). Nhiệm vụ là biến nó thành một pipeline alignment chạy được: dữ liệu sạch → hàm loss đúng toán → số đo tin cậy → báo cáo."
  knowledge:
    - "Định dạng dữ liệu preference (prompt / chosen / rejected) và vì sao cần validate theo dòng"
    - "Trực giác toán học của DPO: so sánh log-ratio của policy với reference"
    - "Trực giác toán học của ORPO: SFT loss cộng thêm phạt tỷ lệ odds"
    - "Chống rò rỉ dữ liệu khi chia train/val theo prompt"
    - "Thói quen production: typed code, config YAML, pytest, ruff, mypy, CI"
  conceptFlow:
    - "Preference pair là 'phiếu bầu' của con người, không phải nhãn tuyệt đối"
    - "Alignment = dạy mô hình nới rộng khoảng cách điểm giữa câu được chọn và câu bị loại"
    - "DPO cần mô hình tham chiếu để giữ policy không trôi quá xa; ORPO gộp cả SFT và preference vào một bước"
    - "Không có số đo thì không biết mình đã cải thiện hay chưa: pairwise accuracy là thước đo tối thiểu"
  phases:
    - time: "0-30 phút"
      owner: "Học viên"
      title: "Dựng môi trường và đọc bản đồ repo"
      description: "Tạo venv, cài package ở chế độ editable, chạy make test để nhìn thấy trạng thái xuất phát."
    - time: "30-75 phút"
      owner: "Học viên"
      title: "Task 1 - Data loader chống lỗi"
      description: "Sửa dòng JSON hỏng, thêm lỗi có số dòng, chặn prompt trùng, chia train/val theo prompt."
    - time: "75-105 phút"
      owner: "Học viên (tùy chọn)"
      title: "Task 1.5 - Sinh dữ liệu tổng hợp"
      description: "Dùng scripts/generate_data.py để mở rộng dataset bằng LLM, sau đó validate lại."
    - time: "105-165 phút"
      owner: "Học viên"
      title: "Task 2 - Cài đặt DPO hoặc ORPO"
      description: "Viết hàm loss ổn định số trong src/preference_lab/losses.py và tự viết test khẳng định giá trị."
    - time: "165-210 phút"
      owner: "Học viên"
      title: "Task 3 - Đánh giá và CLI"
      description: "Thay điểm giả bằng scorer tất định, xử lý hòa điểm, xuất outputs/metrics.json."
    - time: "210-240 phút"
      owner: "Học viên"
      title: "Task 4 - Báo cáo và demo 1 phút"
      description: "Điền REPORT_TEMPLATE.md, cập nhật data card, chạy smoke test và trình bày kết quả."
  outcomes:
    - "Repo có `make test`, `make lint`, `make typecheck` cùng xanh"
    - "File outputs/metrics.json chứa pairwise_accuracy có ý nghĩa (không phải 1.0 giả)"
    - "docs/REPORT_TEMPLATE.md được điền đầy đủ với ít nhất một failure mode quan sát được"
  reassurance: "Bạn không cần GPU và không cần huấn luyện mô hình thật. Toàn bộ Lab chạy trên CPU với NumPy; phần TRL/torch chỉ là tùy chọn mở rộng nếu bạn còn thời gian."

## 1. Thuật ngữ cần biết

| Thuật ngữ gốc | Bản chất khái niệm | Minh hoạ trực quan |
| --- | --- | --- |
| `Preference pair` | Bộ ba `prompt` + `chosen` + `rejected`. Nó không nói "câu này đúng", nó chỉ nói "câu này **tốt hơn** câu kia". Tín hiệu là *tương đối*, không tuyệt đối. | Trong `data/sample_preferences.jsonl`, cùng câu hỏi về self-attention: bản `chosen` giải thích đúng cơ chế trọng số, bản `rejected` nói sai rằng đó là "phiên bản đơn giản của RNN". |
| `Policy model` | Mô hình đang được huấn luyện — thứ bạn muốn thay đổi hành vi. | Trong Lab, policy được đại diện bằng mảng `policy_chosen_logps` / `policy_rejected_logps` truyền vào `dpo_loss`. |
| `Reference model` | Bản sao đông cứng của mô hình *trước khi* align, dùng làm mốc neo. Không có nó, policy có thể "chạy trốn" khỏi ngôn ngữ tự nhiên để tối đa hoá loss. | `ref_chosen_logps` / `ref_rejected_logps`. Hãy coi nó như bản gốc của một tài liệu để đối chiếu khi biên tập. |
| `DPO` (Direct Preference Optimization) | Học ưu tiên trực tiếp, **không cần** huấn luyện reward model riêng. Nó đẩy *log-ratio* của policy vượt lên trên *log-ratio* của reference. | Giống chấm bài theo đường cong: không chấm điểm tuyệt đối, chỉ yêu cầu bài A phải nhỉnh hơn bài B nhiều hơn mức chênh lệch ban đầu. |
| `ORPO` (Odds Ratio Preference Optimization) | Gộp SFT và preference vào **một** bước, dùng tỷ lệ odds thay vì cần reference model. Rẻ hơn về bộ nhớ vì chỉ giữ một mô hình. | Vừa dạy viết đúng văn phong (SFT), vừa phạt khi mô hình nghiêng về câu bị loại — trong cùng một buổi học. |
| `beta` | Hệ số nhiệt điều tiết mức độ tin vào tín hiệu ưu tiên. `beta` lớn → bám sát reference; `beta` nhỏ → cho phép trôi xa hơn. | Trong `configs/local.yaml`: `training.beta: 0.1`. |
| `log-sigmoid` | Cách tính `log(σ(x))` mà không bị tràn số. Viết thẳng `log(1/(1+exp(-x)))` sẽ ra `inf`/`nan` khi `x` quá âm. | Đây chính là "numerical stability" mà `TODO(student)` trong `losses.py` nhắc tới. |
| `Pairwise accuracy` | Tỷ lệ mẫu mà mô hình chấm `chosen` cao điểm hơn `rejected`. Thước đo tối thiểu để biết alignment có tác dụng. | `pairwise_accuracy()` trong `src/preference_lab/evaluate.py`. Nếu để nguyên điểm giả 1.0 vs 0.0 thì kết quả **luôn** là 100% — vô nghĩa. |
| `Data leakage` (rò rỉ dữ liệu) | Cùng một prompt xuất hiện ở cả train và validation, khiến điểm đánh giá bị thổi phồng. | `split_by_prompt()` phải nhóm theo prompt rồi mới cắt, chứ không cắt thẳng theo chỉ số dòng. |
| `TODO(student)` | Quy ước đánh dấu chỗ code bị **cố tình** để trống. Đây chính là phạm vi công việc của bạn. | Chạy `grep -rn "TODO(student)" src/` để lấy danh sách việc cần làm. |
| `Regression prompt` | Bộ prompt cố định chạy trước và sau khi thay đổi mô hình, để phát hiện việc "sửa chỗ này hỏng chỗ kia". | `docs/regression_prompts.md` gồm 4 tình huống: lời khuyên y tế rủi ro cao, tóm tắt giới hạn từ, thừa nhận không chắc chắn, xử lý sự cố thiếu ngữ cảnh. |

## 2. Mục tiêu & đầu ra

Bạn hoàn thành khi **repo chạy trọn chuỗi `validate → loss → evaluate → report` mà không còn `NotImplementedError`, và bạn giải thích được vì sao con số của mình đáng tin**.

Cụ thể, 5 bằng chứng đầu ra bắt buộc:

1. **Dữ liệu sạch** — `pref-lab validate data/sample_preferences.jsonl` in ra số example đã nạp, và loader báo lỗi **kèm số dòng** khi gặp JSON hỏng.
2. **Loss đúng toán** — `pytest tests/test_losses.py` xanh với các test *do bạn viết lại* để khẳng định **giá trị số**, thay vì chỉ khẳng định `NotImplementedError`.
3. **Metrics có ý nghĩa** — file `outputs/metrics.json` tồn tại và `pairwise_accuracy` **không** phải 1.0 do điểm giả cứng.
4. **Chất lượng code** — `make lint` và `make typecheck` cùng sạch (repo bật `mypy strict`).
5. **Tài liệu** — `docs/REPORT_TEMPLATE.md` và `docs/data_card_template.md` được điền, nêu **ít nhất một** failure mode bạn thực sự quan sát được.

> 💡 **Tiêu chí "đủ tốt"**: bạn chỉ cần chọn **một** trong hai hướng DPO **hoặc** ORPO để cài đặt trọn vẹn. Làm cả hai là điểm cộng, không bắt buộc.

---

## 3. Chuẩn bị

### 3.1. Công cụ tối thiểu

| Hạng mục | Yêu cầu | Cách kiểm tra |
| --- | --- | --- |
| Python | >= 3.10 (khai báo trong `pyproject.toml`) | `python3 --version` |
| Git | Bất kỳ bản nào | `git --version` |
| Terminal | bash hoặc zsh | `echo $SHELL` |
| Trình soạn thảo | VS Code hoặc tương đương | — |

> ⚠️ **Cảnh báo phiên bản**: Python 3.13+ có thể trục trặc khi build `pydantic`/`numpy` từ source. Nếu `pip install` báo lỗi biên dịch, hãy tạo venv bằng Python 3.11: `python3.11 -m venv .venv`.

### 3.2. Dữ liệu và điều kiện

- Dataset mẫu có sẵn: `data/sample_preferences.jsonl` — **24 dòng, trong đó dòng 1 cố tình hỏng cú pháp JSON**. Đây là bài tập, không phải lỗi của bạn.
- Config có sẵn: `configs/local.yaml`.
- **Không cần GPU.** Toàn bộ đường đi bắt buộc chỉ dùng NumPy.
- Nhóm phụ thuộc `train` (torch, transformers, trl, peft) là **tùy chọn**, chỉ cài nếu bạn muốn thử `PreferenceTrainer` thật.
- Task 1.5 cần `OPENAI_API_KEY`. Không có key thì **bỏ qua Task 1.5**, Lab vẫn đạt 100% yêu cầu.

### 3.3. Luật chơi của repo

Trích từ `README.md` — hãy tôn trọng nghiêm ngặt:

1. Không viết lại toàn bộ repository.
2. Chỉ cài đặt trong các khối `TODO(student)`, trừ khi có lý do rõ ràng và bạn ghi lại trong báo cáo.
3. Giữ test xanh sau mỗi milestone.
4. Không commit secret, model weight hay dataset riêng tư (`.gitignore` đã chặn `.env`, `*.safetensors`, `outputs/`).

### 3.4. Bản đồ repo

```text
src/preference_lab/     Package Python chính (nơi có mọi TODO)
  ├── schemas.py        PreferenceExample (pydantic) — validate 1 bản ghi
  ├── data.py           load_jsonl, split_by_prompt
  ├── losses.py         dpo_loss, orpo_loss  ← trái tim của Lab
  ├── evaluate.py       pairwise_accuracy, write_metrics
  ├── trainers.py       TrainingConfig, PreferenceTrainer (tùy chọn)
  ├── config.py         load_config đọc YAML
  └── cli.py            lệnh `pref-lab validate` và `pref-lab evaluate`
data/                   Dataset preference mẫu
configs/local.yaml      Siêu tham số cho thí nghiệm cục bộ
docs/                   Lab guide, report template, data card, regression prompts
scripts/                generate_data.py (sinh dữ liệu), smoke_test.sh
tests/                  Test cho phần bạn làm
```



## 4. Thực hành

### Bước 0 — Dựng môi trường (0-30 phút)

```bash
cd Phase2-Track3-Day7-DPO-ORPO-Alignment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

Kiểm tra CLI đã đăng ký:

```bash
pref-lab --help
```

**Kết quả mong đợi**: bảng trợ giúp liệt kê hai lệnh `validate` và `evaluate`.

Liệt kê toàn bộ việc phải làm:

```bash
grep -rn "TODO(student)" src/ scripts/
```

**Kết quả mong đợi**: 11 dòng khớp, trải trên 6 file — `losses.py` (4), `data.py` (2), `trainers.py` (2), `cli.py` (1), `evaluate.py` (1), `schemas.py` (1).

> Lưu ý: một số dòng là chuỗi thông báo bên trong `raise NotImplementedError(...)`, nên số dòng khớp nhiều hơn số **đầu việc** thực tế. Đầu việc thực sự gồm 7: 2 ở `data.py`, 2 ở `losses.py`, và mỗi file `schemas.py` / `evaluate.py` / `cli.py` một việc. Phần `trainers.py` là tùy chọn mở rộng.

Chạy test lần đầu để lấy mốc xuất phát:

```bash
make test
```

**Kết quả mong đợi**: `tests/test_data.py::test_load_sample_data` **THẤT BẠI**. Đây là điều đúng đắn — có hai nguyên nhân chồng lên nhau và bạn sẽ gỡ chúng ở Task 1:

1. Dòng 1 của dataset hỏng cú pháp JSON → `json.JSONDecodeError`.
2. Test đang kỳ vọng `len(examples) == 2`, trong khi file thực tế có 24 dòng.

> 🧭 **Định hướng**: Trong Lab này test **cũng là artefact bạn phải sửa**. Một test khẳng định sai sự thật thì tệ hơn là không có test.

---

### Task 1 — Data loader chống lỗi (30-75 phút)

#### 1a. Chẩn đoán dữ liệu trước khi sửa code

```bash
python3 -c "
import json
for i, line in enumerate(open('data/sample_preferences.jsonl', encoding='utf-8'), 1):
    if not line.strip(): continue
    try: json.loads(line)
    except json.JSONDecodeError as e: print(f'Dòng {i}: {e}')
"
```

**Kết quả mong đợi**:

```text
Dòng 1: Expecting ',' delimiter: line 1 column 36 (char 35)
```

Xem kỹ ký tự gây lỗi:

```bash
head -c 60 data/sample_preferences.jsonl
```

Bạn sẽ thấy `{"prompt":"Explain the concept of "self-attention" in Transf...` — cặp nháy kép quanh `self-attention` **chưa được escape**, khiến JSON parser tưởng chuỗi đã kết thúc ở đó.

#### 1b. Sửa dữ liệu

Escape hai dấu nháy kép bên trong giá trị `prompt` thành `\"`:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("data/sample_preferences.jsonl")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
lines[0] = lines[0].replace('"self-attention"', '\\"self-attention\\"')
p.write_text("".join(lines), encoding="utf-8")
print("Đã sửa dòng 1")
PY
```

Xác nhận không còn dòng hỏng bằng cách chạy lại lệnh chẩn đoán ở bước 1a — output phải **rỗng**.

> 📝 Ghi lại thao tác này vào mục *Cleaning steps taken* của `docs/REPORT_TEMPLATE.md`.

#### 1c. Viết lại `load_jsonl`

Mở `src/preference_lab/data.py`. Yêu cầu của `TODO(student)`:

- **Lỗi có số dòng**: khi JSON hỏng hoặc schema sai, thông báo phải chỉ rõ dòng nào.
- **Chặn prompt trùng**: hai bản ghi cùng `prompt` là dấu hiệu dữ liệu bẩn.
- **Guardrail PII** (tùy chọn): cảnh báo nếu phát hiện email/số điện thoại.

Gợi ý khung xử lý — bạn tự viết phần thân:

```python
for line_no, line in enumerate(f, start=1):
    if not line.strip():
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: JSON không hợp lệ - {exc}") from exc
    try:
        example = PreferenceExample.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path}:{line_no}: schema không hợp lệ - {exc}") from exc
    # ... kiểm tra prompt trùng tại đây ...
```

> ⚠️ Repo bật `mypy strict`. Mọi hàm bạn thêm **phải** có type annotation đầy đủ, kể cả kiểu trả về `-> None`.

#### 1d. Làm chặt validation trong `schemas.py`

`TODO(student)` trong `chosen_and_rejected_must_differ` hiện chỉ so sánh `chosen == rejected` — quá lỏng. Hãy làm nó **bền với khác biệt khoảng trắng và hoa/thường**. Gợi ý: chuẩn hoá cả hai chuỗi (lower + gộp khoảng trắng) trước khi so sánh.

#### 1e. Chia train/val theo prompt

`split_by_prompt()` hiện cắt thẳng theo chỉ số → nếu một prompt có nhiều bản ghi, nó sẽ nằm ở cả hai phía. Yêu cầu:

1. **Nhóm** các example theo `prompt`.
2. **Xáo trộn tất định** danh sách prompt bằng `random.Random(seed)` (seed lấy từ `configs/local.yaml`, mặc định 42).
3. Cắt theo **nhóm**, đảm bảo mọi bản ghi của một prompt chỉ về một phía.
4. Giữ bất biến: `len(train) + len(val) == len(examples)`.

#### 1f. Cập nhật test cho khớp sự thật

Sửa `tests/test_data.py` để phản ánh dữ liệu thật (24 dòng sau khi sửa) và bổ sung test cho hành vi mới:

```python
def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected

def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="2"):
        load_jsonl(bad)

def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)
    assert not ({e.prompt for e in train} & {e.prompt for e in val})
```

**Chốt Task 1**:

```bash
pytest tests/test_data.py -q
pref-lab validate data/sample_preferences.jsonl
```

**Kết quả mong đợi**: test xanh, và CLI in `Loaded 24 preference examples`.
