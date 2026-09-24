# Third-Party Code, Models, and Datasets Attribution

This project incorporates, adapts, or builds upon artifacts from the following open-source research projects and resources for academic research and evaluation purposes:

## 1. Datasets & Benchmarks
- **AnnoMI Dataset**: by Wu et al.
  - *Paper*: "AnnoMI: A Dataset of Mentoring and Motivational Interviewing Conversations".
  - *Source*: https://github.com/uccollab/AnnoMI
  - *Usage*: Base client profiles and clinical persona grounding.
- **MIV6.3A Dataset**: Annotated dataset of motivational interviewing utterances by Ali et al.
  - *Source*: https://github.com/cimhasgithub/AutoMISC
  - *Usage*: Ground-truth validation benchmark for automated speech coding annotators.

## 2. Baseline Architectures & Evaluation Tools
- **ConsistentMIClientSimulator**: by Yang et al.
  - *Paper*: "Consistent Client Simulation for Motivational Interviewing".
  - *Source*: https://github.com/IzzetYoung/ConsistentMIClientSimulator
  - *Usage*: Persona 5-tuple schema and multi-session baseline comparison.
- **AutoMISC**: by Ali et al.
  - *Paper*: "Automated Coding of Counsellor and Client Behaviours in Motivational Interviewing Transcripts: Validation and Application"
  - *Source*: https://github.com/cimhasgithub/AutoMISC
  - *Usage*: Adapted for utterance-level MISC speech act coding and therapist adherence metrics.
- **Patient-Ψ**: by Wang et al.
  - *Paper*: "Patient-Ψ: Using Large Language Models to Simulate Patients for Mental Health Training".
  - *Source*: https://github.com/ruiyiw/patient-psi
  - *Usage*: Multi-session baseline evaluation using CBT cognitive state representations.
- **SimPatient**: by Steenstra et al.
  - *Paper*: "Scaffolding Empathy: Training Counselors with Simulated Patients and Utterance-level Performance Visualizations"
  - *Source*: https://github.com/IanSteenstra/SimPatient
  - *Usage*: Inter-session state transition baseline comparison.

## 3. Foundation Models
- **Meta Llama 3.3 (meta-llama/llama-3.3-70b-instruct)**: Governed by the Llama 3.3 Community License Agreement.
- **Google Gemma 4 (google/gemma-4-31b-it)**: Governed by the Apache 2.0 License / Gemma Terms of Use.
- **Qwen 2.5 (qwen/qwen-2.5-72b-instruct)**: Governed by the Qwen Research License.

## 4. Software Libraries & Frameworks
- **LangGraph**: MIT License (LangChain, Inc.).
- **OpenInference / Arize Phoenix**: Apache 2.0 License.
- **SciPy / Statsmodels / Pingouin / Pandas / NumPy**: BSD / MIT / GPL Open Source Licenses.

All third-party materials are used strictly in compliance with their respective academic licenses, terms of use, and the doctrine of academic fair use for scientific research and reproducible benchmarking.