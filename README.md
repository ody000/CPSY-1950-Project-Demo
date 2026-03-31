# CPSY 1950 — Computational Models of Human Behavior: Demo Notebook
This repository contains a demo notebook for CPSY 1950 at Brown University, illustrating how large language models can be evaluated as models of human decision making using the **two-step task**: a classic paradigm from computational psychiatry.
## Overview
The notebook covers two experiments:
1. **Model Alignment with Human Decisions**: Given a real human participant's trial by trial transcript from the two step task, we query a frontier LLM at each decision point and measure how surprised the model is by the human's actual choice using negative log-likelihood (NLL).
2. **Model Behavior in an Open Environment**: We let the model play the two-step task autonomously against a simulated environment with drifting reward probabilities, and analyze its behavior using the classic stay/switch analysis to probe for model based vs model free reasoning.
Both experiments use the **magic carpet cover story** version of the task, in which the abstract decision structure is embedded in a fantasy narrative to test generalization.
## Setup
1. Clone the repository:
```bash
git clone https://github.com/TekinGunasar/CPSY-1950-Project-Demo
cd CPSY-1950-Project-Demo
```

2. Create and activate a conda environment:
```bash
conda create -n cpsy1950 python=3.11
conda activate cpsy1950
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. After installing, you usually also want to register the environment as a Jupyter kernel:
````
python -m ipykernel install --user --name cpsy1950
````

6. Open the notebook and run all cells.
## References
- Binz et al. (2025). *Centaur: a foundation model of human cognition*
- Feher da Silva et al. (2023). *Rethinking model-based and model-free influences on mental effort and striatal prediction errors.* Nature Human Behaviour, 7(6), 956–969.
