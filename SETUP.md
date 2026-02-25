# Local Setup Guide

This repository excludes large/local runtime artifacts from Git. After cloning, you need to create or copy them locally.

## 1) Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## 2) Run the app

```bash
streamlit run home.py
```

If your entry file differs in your branch, run the correct Streamlit entrypoint for your version.

## 3) Local files not tracked in Git

These are ignored and must exist locally if your workflow depends on them:

- `attendance.db`
- `*.joblib`
- `*.pkl`
- `ml_models/`
- `encodings/`
- `employee_photos/`
- `data/captured_faces/`
- `data/employee_photos/`

## 4) Team workflow recommendation

- Keep models/data in shared storage (Drive/S3/private artifact store), not in Git.
- Add a small script later (for example `scripts/bootstrap_local_assets.sh`) to download required local assets.
