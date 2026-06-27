# medical-vlm-prompt-injection-defense# Medical Vision-Language Model Prompt Injection Defense

This repository contains the implementation, datasets, experimental results, and supplementary material for the paper:

Defending Medical Vision-Language Models Against Prompt Injection Through Structured Reasoning and Self-Check Verification


# Dataset Organization

The repository contains both original medical images and adversarially modified images used for prompt injection evaluation.

* **data/abnormal/** contains the original medical images.
* **data/injected/** contains the corresponding adversarial images with embedded prompt injections.


# Image Naming Convention

All experimental results are indexed using the **original image identifier**.

For adversarial experiments, injected images may include one of the following prefixes:

| Prefix   | Description                                            |
|----------|--------------------------------------------------------|
| `high_`  | Prompt injection with a high-visibility text overlay.  |
| `low_`   | Prompt injection with a low-visibility text overlay.   |
| `small_` | Prompt injection using a small-font text overlay.      |

For example,

```
SOB_M_DC-14-12312-100-008.png
high_SOB_M_DC-14-12312-100-008.png
low_SOB_M_DC-14-12312-100-008.png
small_SOB_M_DC-14-12312-100-008.png
```

all correspond to the **same underlying medical image** with different adversarial variants.

Accordingly, all response files and semantic similarity analyses use the original image identifier (e.g., `SOB_M_DC-14-12312-100-008`) to maintain consistent indexing across clean, attacked, and defended evaluations.

---

# Experimental Results

## Repeated Generation Responses

```
results/repeated_generation_responses/
```

This directory contains the complete model responses used for semantic consistency evaluation.

Each Excel workbook corresponds to one evaluated vision-language model and contains repeated generations under three prompting conditions:

* Clean
* Adversarial
* Defended

Three independent generations were produced for each prompting condition.

---

## Semantic Similarity Results

```
results/semantic_similarity/
```

This directory contains the semantic similarity scores computed between:

* Clean vs. Attacked generations
* Clean vs. Defended generations

These scores were used to compute restoration gain and evaluate semantic consistency recovery.

---


