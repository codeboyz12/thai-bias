# thai-bias

Thai gender bias detection in multilingual sentence embeddings using SEAT.

## Install

```bash
pip install git+https://github.com/<YOUR_USERNAME>/thai-bias.git
```

To get the latest changes:

```bash
pip install --force-reinstall git+https://github.com/<YOUR_USERNAME>/thai-bias.git
```

## Quick Start (Colab)

```python
import thai_bias as tb
import matplotlib.pyplot as plt

# 1. Thai font
tb.setup_matplotlib_thai()
plt.rcParams.update(tb.config.DEFAULT_PLOT_STYLE)

# 2. Word sets (user-defined)
X = ["พ่อ","ลูกชาย","พี่ชาย","น้องชาย","ปู่","ตา","ลุง","หลานชาย"]
Y = ["แม่","ลูกสาว","พี่สาว","น้องสาว","ย่า","ยาย","ป้า","หลานสาว"]
A = ["งาน","อาชีพ","เงินเดือน","ตำแหน่ง","บริษัท","การงาน","สำนักงาน","ผู้บริหาร"]
B = ["บ้าน","ครอบครัว","ลูก","แม่บ้าน","ดูแล","แต่งงาน","พ่อแม่","เลี้ยงดู"]
OCCUPATIONS = ["วิศวกร","โปรแกรมเมอร์","ทหาร","พยาบาล","ครู","แพทย์","นักวิจัย"]

WORD_SETS = {"X": X, "Y": Y, "A": A, "B": B}

# 3. Run one model
result = tb.run_model(
    "BAAI/bge-m3",
    word_sets   = WORD_SETS,
    occupations = OCCUPATIONS,
)
tb.plot_model(result, WORD_SETS, OCCUPATIONS)

# 4. Run all models
ALL_RESULTS = {}
for name in tb.EmbedderFactory.list_models():
    try:
        ALL_RESULTS[name] = tb.run_model(name, WORD_SETS, OCCUPATIONS)
    except Exception as e:
        print(f"Skip {name}: {e}")

results = list(ALL_RESULTS.values())

# 5. Compare
tb.plot_within_group(results, "A", OCCUPATIONS)
tb.plot_within_group(results, "B", OCCUPATIONS)
tb.plot_within_group(results, "C", OCCUPATIONS)
tb.plot_cross_group(results, OCCUPATIONS)

# 6. Ranking
df = tb.final_ranking(results, OCCUPATIONS)
```

## Package Structure

```
thai_bias/
├── config.py      device detection, Thai font, constants
├── embedders.py   EmbedderBase, 3 subclasses, Registry, Factory
├── metrics.py     vectorized SEAT, permutation test, direct bias
├── visualize.py   all plot functions (no metric logic)
├── pipeline.py    ModelResult, run_model, plot_model
└── analysis.py    within/cross-group comparison, final ranking
```

## Supported Models

| Group | Type | Example |
|---|---|---|
| A | Instruction-tuned | `intfloat/multilingual-e5-large-instruct` |
| B | Sentence Transformer | `BAAI/bge-m3` |
| C | Base Contextual BERT | `airesearch/wangchanberta-base-att-spm-uncased` |

## Adding a New Model

เปิด `thai_bias/embedders.py` แล้วเพิ่ม entry ใน `MODEL_REGISTRY`:

```python
"your-org/your-model": {
    "cls": SentenceEmbedder,          # หรือ InstructionEmbedder / BERTEmbedder
    "short": "your-model-short-name",
    "cfg": EmbedConfig(group="B"),
},
```
