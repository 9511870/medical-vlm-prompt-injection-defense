# Medical VLM prompt-injection defence

Judge scoring and statistics for "[paper title]". The study tests a
structured-reasoning defence with a self-check against textual, visual and
delayed visual prompt injection in five vision-language models, on 58 medical
images.

## What this repository contains
- `src/`: the code for judge scoring and the statistical tests.
- `results/`: model responses, judge scores and statistics (text and tables only).
- `data/image_ids.csv`: identifiers of the 58 images used, with their source.
  **The images themselves are not included** (see "Data" below).

Not included: code for generating the injected images and for collecting model
responses. The attack configurations and prompt templates are described in the
Supplementary Information of the paper.

## Repository structure
```
data/image_ids.csv                       image identifiers and sources
data/README.md                           how to obtain the images
results/                                 responses, judge scores, statistics
src/judge_independent.py                 judge scoring functions
src/judge_two_files.py                   scoring pipeline (responses + ground truth)
src/judge_ablation_gt_standalone_v7.py   scoring for the ablation conditions
src/wilcoxon.py                          paired Wilcoxon signed-rank tests
```

## Model labels
| Label in files | Model in the paper |
|---|---|
| gpt | GPT-5.5 |
| gemini | Gemini 3 Flash |
| claude | Claude Sonnet 4.6 |
| qwen | Qwen 3.6 Plus |
| reka | Reka Edge |

## Column and label glossary
| In the files | In the paper |
|---|---|
| Condition `normal`, Injection Type `none` | No attack |
| Condition `attack` | Attacked |
| Condition `defense` | Defended |
| Injection Type `textual` / `visual` / `delayed` | Textual / Visual / Delayed visual injection |
| `REFRENCE RESPONSE` | No-attack response (spelling as in the files) |
| `WITHOUT DEFENSE` | Attacked response |
| `WITH DEFENSE` | Defended response |
| `SELF CHECK` = Fail | The self-check flagged an inconsistency |
| `RE-ANALYSIS` = Fail | Final defence failure |

## Judge scoring
An independent medical language model (Llama3-Med42-8B, Q4_K_M, run locally
with Ollama at temperature 0, output limited to 400 tokens, JSON output) scores
how closely each response agrees with the ground-truth description, from 0 to 1.
The judge sees only the ground truth and the response, not the image.
Restoration gain is the defended score minus the attacked score.

```
python src/judge_two_files.py --responses <responses.xlsx> --gt_source <combined.xlsx> \
  --backend ollama --judge_model hf.co/QuantFactory/Llama3-Med42-8B-GGUF:Q4_K_M \
  --out_dir results/
```

## Statistics
`src/wilcoxon.py` runs two-tailed paired Wilcoxon signed-rank tests on
image-level mean scores (attacked vs defended), pooled across the three
pathways for each model. See the script header for its arguments.


