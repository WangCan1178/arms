> **Correction (anonymous repository)**  
> Due to an error on our side, the anonymous review code repository for *Towards Robust Tool Use in LLM Agents via Experience-Driven Adaptive Guidance* is actually at:  
> **<https://anonymous.4open.science/r/ExpG-7715>**

---

# Arms: Efficient Routing of Vision-Language Models as API Services

> Anonymous submission repository.

Arms is a router for vision-language model (VLM) selection.  
Given an image-text query and a candidate VLM, Arms predicts whether the VLM can answer correctly, then supports confidence-aware routing in both in-distribution (ID) and out-of-distribution (OOD) settings.

## Overview

This repository contains:

- Code for training and evaluating Arms.
- Routing examples for single-router and cascaded strategies.
- Data interface used for supervised routing labels.

## Key Ideas

- **Multimodal routing formulation**: route VLMs with image-text queries instead of text-only prompts.
- **Capability-aware signals**: include VLM profile text as part of router input.
- **MoE-based router**: use a mixture-of-experts architecture to model interactions between query difficulty and VLM capability.
- **Scalable extension**: support unseen VLMs with incremental and independent training strategies.

The overall architecture is shown below (replace with your final PNG):

<p align="center">
  <img src="pic/architecture.jpg" alt="arms architecture" width="85%">
</p>

## Data Format

For training, prepare a pickle file containing records with the following schema:

```python
class JudgedRecord(TypedDict):
    image: Image.Image
    question: str
    lang: str
    answer: str | list[str]
    model_name: str
    model_answer: str
    source_dataset: str
    source_id: str
    record_id: str
    correct: bool
```

`image` is an opened PIL image object. You can modify `pkl_data_files` to use your own dataset.

## Inference and Routing

- For inference, provide `acc_judge_mapper` to `routing.py`.
- The mapper format is `{key: boolean}`, where key is `dataset-name_model-name_record-id`.
- For cascaded routing, see `route_example.py`.

