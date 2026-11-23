# AI Assistance Declaration

This project used GitHub Copilot Agent (via the Copilot SDK in VS Code) to assist with repository structure, Python implementation, notebook generation, documentation, debugging, tests, and Streamlit dashboard code.

The underlying Copilot model name was not exposed by the development interface and is not claimed here.

| File or section | AI-generated contribution | Human verification/modification |
|---|---|---|
| `src/data_processing.py` | Cleaning, audit, duplicate, and complete-panel implementation | Human required to review duplicate resolution and panel assumption |
| `src/features.py` | Lag, rolling, cyclical, and country-tier feature code | Human must verify split and leak-free feature logic |
| `src/train.py` | Baseline and ensemble training pipeline | Human must verify real results and parameter choices |
| `src/evaluate_test.py` | Test metrics and forecast artifact pipeline | Human must verify test is used once and interpret results |
| `app.py` | Six-page Streamlit dashboard structure | Human must test navigation and visual output |
| EDA and model notebooks | Notebook structure and chart/model cells | Human must inspect outputs and add academic interpretation |
| `README.md` | Initial project and submission documentation | Human must insert repository/GitHub details and verify claims |
