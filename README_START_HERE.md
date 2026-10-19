# GPU Memory-Aware Scheduling for Large-Batch LLM Inference — Project Bundle

Open in this order:
1. `PROJECT_OVERVIEW.md` – what the project is, what was built, key results, limitations
2. `paper/paper.pdf` – the IEEE conference-style paper (5 pages)
3. `code_repo/report/REPORT.md` – the full report (problem, design, evaluation, results, references)
4. `code_repo/README.md` – repo layout, all result tables with their source files, how to run
5. `papers/PAPERS.md` – base paper + 5 journal papers (verified) + background papers
6. `demo/scheduler_demo.ipynb` – self-contained notebook for showing the professor (any Python, no GPU)
7. `docs_overview/GPU_Memory_Aware_LLM_Scheduling_Overview.pdf` – the earlier planning document (see PROJECT_OVERVIEW §8)

Quick run (no GPU needed):
```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn matplotlib fastapi uvicorn httpx
cd demo/scheduler_scripts_standalone && ../../.venv/bin/python run_demo.py
```

Real-engine experiment (Mac with Apple Silicon): `code_repo/docs/setup.md`.
