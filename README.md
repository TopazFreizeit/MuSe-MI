# MuSe-MI: A Framework for Assessing Multi-Session Motivational Interviewing with AI-Generated Patients

Official repository and evaluation benchmark for **MuSe-MI** (*Multiple Sessions of Motivational Interviewing*), accepted to the **Findings of AACL-IJCNLP 2026**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Venue: AACL-IJCNLP Findings 2026](https://img.shields.io/badge/Venue-AACL--IJCNLP%202026%20Findings-purple.svg)](https://aacl2025.org/)

---

## 📌 Overview

**MuSe-MI** introduces a clinically grounded multi-session patient simulation framework that:
1. **Tracks 4 Latent Psychological States** ($V \in [1, 100]^4$): *Anger & Defensiveness*, *Self-Efficacy*, *Problem Recognition*, and *Motivational Readiness*.
2. **Appraises Therapist Actions Turn-by-Turn** via a *Cognitive Interpreter* grounded in Cognitive Appraisal Theory.
3. **Maintains Longitudinal Continuity** through an inter-session *Life Event Simulator* and *Rolling Memory*.
4. **Benchmarks Therapist Adherence**: Differentiates counselor competence across MI-Consistent ($0$ MIIN), MI-Inconsistent ($3$ MIIN), and Highly Confrontational ($6$ MIIN) interventions.

---

## 📂 Repository Structure

```tree
.
├── src/                               # Core MuSe-MI Simulation
│   ├── patient/                       # Patient Agent Generative Policy & Prompts
│   ├── therapist/                     # MET Therapist & MINA Inconsistent Injection Engine
│   ├── patient_state_manager/         # Cognitive Interpreter & Deterministic State Tracker & Patient Memory
│   ├── between_session_events/        # Life Event Simulator (and Macro up)
│   ├── graphs/                        # LangGraph Multi-Session State Machine & Workflow
│   ├── vanilla_baseline/              # Vanilla Prompt-only Simulated Patient Baseline
│   ├── consistent_client_baseline/    # Consistent Client Multi-Session Baseline
│   ├── patient_psi_baseline/          # Patient-Ψ Multi-Session Baseline
│   ├── simpatient_baseline/           # SimPatient Multi-Session Baseline
│   └── utils/                         # Tracing, Cost Tracking & LLM Wrappers
├── evaluation/                        # Automated Evaluation
│   ├── src_automisc/                  # Utterance-level Speech Coder (Sustain, DARN, CAT)
│   ├── src_wai/                       # Working Alliance Inventory (WAI-SR) Judge
│   ├── src_ambivalence/               # Client Ambivalence Scale Evaluation
│   └── misc_coder_models/             # MIV6.3A Benchmark Annotations (Gemma, Llama, Qwen)
├── data/                              # Patient Personas & Clinical Grounding
│   ├── patient_profiles/              # 30 AnnoMI-grounded Patient Personas for MuSeMI
│   ├── patient_psi_profiles/          # CBT-adapted Profiles for Patient-Ψ
|── run_therapy.py                     # Single-cohort Multi-session Simulation Entrypoint
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository & Setup Environment
```bash
git clone https://github.com/your-username/MuSe-MI.git
cd MuSe-MI

conda create -n musemi python=3.10 -y
conda activate musemi
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Populate your API keys (we use OpenRouter for uniform access to Llama 3.3, Gemma 4, and Qwen):
```env
OPENROUTER_API_KEY="your-openrouter-key"
PATIENT_MODEL="meta-llama/llama-3.3-70b-instruct"
JUDGE_MODEL="google/gemma-4-31b-it"
```

---

## 🚀 Running Simulations

### Run a 4-Session Simulation for a Single Patient
```bash
python run_therapy.py --patient_idx 1 --miin_level 0 --num_sessions 4
```
Parameters:
* `--patient_idx`: Target patient profile index ($1 \dots 30$).
* `--miin_level`: Therapist adherence tier ($0$: MI-Consistent, $3$: MI-Inconsistent, $6$: Highly Inconsistent).
* `--num_sessions`: Number of therapy sessions (Default: $4$).

---

## 🤖 Automated Speech Coding (AutoMISC)

Utterance-level speech acts are classified into **Sustain Talk**, **Preparatory Change Talk (DARN)**, and **Mobilizing Change Talk (CAT)** using our validated `gemma-4-31b-it` coding model:
```bash
python evaluation/src_automisc/run_batch_coder.py --input_transcripts ./results/runs --output_csv ./results/session_data.csv
```

---

## 📑 Citation

If you use MuSe-MI or AutoMISC in your research, please cite our paper:

```bibtex
@inproceedings{freizeit2026musemi,
  title     = {MuSe-MI: A Framework for Assessing Multi-Session Motivational Interviewing with AI-Generated Patients},
  author    = {Freizeit, Topaz and Zisquit, Moreah and Hashiloni, Kai Golan and Tavens, Chaitze and Friedman, Doron and Bar, Kfir},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```

---

## 📜 Attribution & Third-Party Artifacts

This research repository incorporates and adapts components from open-source academic resources.
For complete details on licensing and attribution, please refer to [ATTRIBUTION.md](ATTRIBUTION.md).

---

## ⚖️ Ethical Statement & License

* **Code License:** The code in this repository is licensed under the [MIT License](LICENSE).
* **Ethical Considerations:** All patient profiles in MuSe-MI are entirely synthetic and grounded in publicly available demonstration transcripts from AnnoMI. No Protected Health Information (PHI) or real patient records were used. MuSe-MI is intended strictly as a research benchmark and educational testbed, not as a replacement for human psychotherapy. 
