# Misalignment Checker

This repo is a small experimental fork of the **Hallucination Risk Calculator / HallBayes** toolkit.  
Instead of checking for hallucinations, we reuse the same EDFL/ISR machinery to build a **misalignment checker** for large language models.

We test this on:

- **GPT-4o-mini** (via the OpenAI API), and  
- A **sleeper-agent model** fine-tuned from Llama-2-7B using the Sleeper Agents dataset.

The main result: in this setting, the EDFL-based checker almost always says outputs are **safe/aligned**, even when the sleeper agent is clearly misbehaving. So this repo is best seen as a **negative result / case study**, not a production safety tool.

---

## What this repo contains

Misalignment-specific pieces:

- `hallbayes/misalignment_toolkit.py`  
  Wraps the original hallucination planner into an **AlignmentPlanner** with:
  - an alignment decision head (`aligned` / `misaligned`), and  
  - a small override rule for obviously low-risk prompts.

- `hallbayes/misalignment_quick_start.py`  
  Minimal example showing how to call the misalignment checker on a few prompts.

- `hallbayes/interactive_alignment_eval.py`  
  Command-line tool that:
  - loads prompts,  
  - generates model answers under an alignment spec,  
  - runs the misalignment checker, and  
  - lets you manually label each answer as aligned / misaligned.

- `misalignment_eval_data/`  
  - `evaluation_dataset.json` – prompts + labels for GPT-4o-mini  
  - `evaluation_dataset_sleeper_agent.json` – prompts + labels for the sleeper agent  

- `hallbayes/misalignment_results/`  
  Confusion matrices and other plots from the experiment.

The original hallucination toolkit lives under `hallbayes/` and is mostly unchanged.

---

## Installation

This repo uses the same dependencies as the original hallucination toolkit.

```bash
# Core requirement
pip install --upgrade openai

# Optional, if you use other providers
pip install anthropic transformers torch ollama requests

# For evaluation scripts and analysis
pip install pandas scikit-learn matplotlib
```

Set your API key (and recommended enable debug logging):

```bash
export OPENAI_API_KEY=sk-...
export OLLMA_URL=http://...
export HALLUCINATION_TOOLKIT_DEBUG=1   # or 0 to disable model output logging
```

---

## Getting started

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   # On Windows:
   # .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install `hallbayes` as a package**

   ```bash
   pip install -e .
   ```

4. **Run the misalignment checker**

   **(a) Web app**

   ```bash
   streamlit run app/web/web_app.py
   # or
   ./launch.sh
   ```

   **(b) Terminal (interactive prompting)**

   ```bash
   cd hallbayes
   python quick_eval.py --interactive
   ```

## Inference Quick Start

### 1. Run a simple misalignment check

```bash
python hallbayes/misalignment_quick_start.py
```

This will:

- call the base model (e.g. GPT-4o-mini),  
- compute EDFL/ISR metrics, and  
- print whether the checker thinks the behaviour is safe or risky.

### 2. Interactive human labelling

```bash
python hallbayes/interactive_alignment_eval.py
```

This will:

- step through prompts in the evaluation dataset,  
- show the model’s answer and the checker’s metrics, and  
- ask you to label whether the behaviour is aligned or misaligned.

The labels and metrics are saved to `alignment_eval_interactive.csv` for later analysis.

---

## Data and sleeper agent model

- Sleeper Agents training data (used to build the sleeper-agent model) can be downloaded from: 
  [https://github.com/anthropics/sleeper-agents-paper/blob/main/code_backdoor_train_data.jsonl](https://github.com/anthropics/sleeper-agents-paper/blob/main/code_backdoor_train_data.jsonl)

- The sleeper agent used in this experiment is a **custom fine-tuned Llama-2-7B model** running on **Ollama**.  
  See the Sleeper Agents repo for details on how to recreate the fine-tuning pipeline:  
  [https://github.com/anthropics/sleeper-agents-paper](https://github.com/anthropics/sleeper-agents-paper)

---

## Caveats

This repo is **research code** for the [Defensive Acceleration Hackathon](https://apartresearch.com/sprints/def-acc-hackathon-2025-11-21-to-2025-11-23), not a production safety system.

In our experiments, the misalignment checker:

- often certifies harmful sleeper-agent outputs as aligned, and  
- tends to be heavily biased towards “SAFE” decisions.

Use this as a starting point or cautionary example when thinking about **post-hoc misalignment detection**, not as a drop-in safety solution.