# Sports Analytics Academy

A personal learning platform for statistics, machine learning and advanced modeling, taught by applying each method to football analytics research questions. Python + Streamlit. Fully separate from PSFAL: no PSFAL code, files or models are read or modified.

## Run it on your computer (the chosen setup)

The academy runs locally. Your progress is saved on your own disk and never gets wiped, and PSFAL keeps your one private Streamlit Cloud slot.

**One-time setup**

1. Install Python 3.11 or newer from python.org. On Windows, tick **"Add Python to PATH"** during install.
2. Get the code: `git clone https://github.com/ben-levin2/sports-analytics-academy.git` (or download it from GitHub with **Code → Download**).
3. Open a terminal (Mac: Terminal app; Windows: PowerShell) and go into the folder:
   `cd sports-analytics-academy`
4. Create a private Python environment for the academy:
   `python3 -m venv .venv`   (Windows: `python -m venv .venv`)
5. Turn it on:
   `source .venv/bin/activate`   (Windows: `.venv\Scripts\activate`). You'll see `(.venv)` at the start of the line.
6. Install the libraries: `pip install -r requirements.txt` (takes a minute or two).

**Every time you study**

1. Open a terminal and run `cd sports-analytics-academy`.
2. Turn the environment on: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. Start the app: `streamlit run app.py`. Your browser opens at http://localhost:8501.
4. To stop: press **Ctrl+C** in the terminal.

**After pulling new code:** run `git pull`, then `pip install -r requirements.txt` again in case libraries changed.

Optional checks: `pip install pytest && pytest -q` runs the test suite.

**Optional: PyTorch (real-LSTM labs, Level 4) and PyMC (the PyMC version of the Level 5 model).** Not needed for anything else; the app detects them and shows an install note when they're missing.
`pip install -r requirements-optional.txt` (or CPU-only and smaller: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Curriculum

Six levels, **28 tracked concepts**. The four predictive-modeling methods sit inside the levels where you first need them:

| Level | Concepts | Projects |
|---|---|---|
| 1 Bayesian Statistics & Shrinkage | 1–6 | P1 Bayesian Player Evaluation (**fully built**) |
| 2 Player Archetypes & Latent Models | 7–10 | P2 Archetype Discovery |
| 3 Player Development & Spatial Modeling | 11–17, **25 Calibration & Brier**, **26 Time-based CV** | P3A Time Until Contribution, P3B Geographic Recruiting |
| 4 Sequential Machine Learning | 18–20, **28 Conformal prediction** | P4 Game-by-Game Performance |
| 5 Advanced Statistical Modeling | 21–23, **27 Mixed-effects models** | P5 Hierarchical Player Development |
| 6 Optimization & Research | 24 | P6A Constrained Lineup Optimization, P6B Integrated Research |

Built now: all 40 concepts have lessons (44 lessons across six levels), including the small-data set (concepts 29–33: regularization, Bayesian logistic with informative priors, monotonic GAMs, transfer learning, discrete-time survival) and honest evaluation (concepts 34–36: write the bar before you look, weak baselines make fake wins, frozen forward tests), plus three classic methods (concepts 37–40: Lasso, Bayesian linear regression, gradient boosting, where gradient boosting is a fourth P3C challenger, and tuning with nested cross-validation). All 9 projects (P1–P6B, including P3C Small-Data Draft Model) have interactive Learning Lab pages and automated Production Candidate checks, the Integrated Research Project runs a scoreboard across every level, and there are 12 Python Lab exercises. Only NumPy, SciPy, scikit-learn, statsmodels (plus Streamlit, pandas, Matplotlib) are required; PyTorch and PyMC are optional.

## Progress rules

- **Lesson completion** uses your *best* quiz score: completed at 70%, mastered at 90%.
- **Concept mastery** is tracked separately. It uses your *latest* answers on the questions tagged with that concept, so a later poor attempt can drop a concept back down. Mastered needs 90% and every lesson teaching the concept attempted.
- Opening a lesson never counts. Short answers are graded by keyword matching, so compare with the model answer.

## Learning Lab vs Production Candidate

Every project has both tracks. A Production Candidate verdict requires all gates: real data you have rights to use, a defined baseline, a predefined metric, time-respecting out-of-sample validation, beating the baseline, a leakage check, and calibration where relevant. Simulated data can never pass. The best possible verdict is "Meets Production Candidate criteria (review before any use)". Nothing is labelled production-ready.

P1's gates are automated (`academy/shrinkage.py`). The other projects list their gates and stay Learning Lab until their automated checks are built.

## Using your own data

Data comes from, in order: an uploaded CSV (**Data Sources** page, with column mapping) → a path in `datasets.toml` (copy `datasets.example.toml`, local runs only) → simulated data. Required columns per dataset are listed on the Data Sources page and in `academy/datasets.py`. Export what you need from your other projects as CSV; the academy never reads them directly.

## Persistence

Progress is saved to `data/progress.json` in the project folder after every quiz, lab check and project step. Locally it is permanent until you delete it. `data/` is git-ignored, so your progress is never pushed to GitHub. Use **Progress & Backup → Download** to keep a copy or move it to another computer.

If you ever deploy to Streamlit Community Cloud instead: its disk is wiped on every restart, uploads are wiped too, and a public app shares one progress file with every visitor. That is why local is the chosen setup.

## Security

Lesson, lab and project code runs with `exec()`, which is not a sandbox. Editing code is on when you run locally. It would be off by default on Community Cloud.

## Adding content (no rebuild)

- **Lesson:** copy any `academy/lessons/l1_*.py`, set `id`, `level`, `order` and `concept_ids` (ids from `academy/curriculum.py`), and write the 10 steps. It's discovered automatically, and the concept stops showing as "planned".
- **Lab exercise:** append an `Exercise` to `academy/labs.py`.
- **Project:** edit or append in `academy/projects_data.py`. For an interactive page and automated gates, add a view (like `academy/views/p1_bayes.py`) and register it in `INTERACTIVE` in `academy/views/projects.py`.
- **Dataset:** add a `Spec` in `academy/datasets.py` with required columns and a simulator in `academy/data.py`.
- **PyMC for Level 5:** add `pymc` to `requirements.txt` when the hierarchical lessons are built. It is heavy, so it is not installed yet.

Run `pytest -q` after any change.

## Structure

```
app.py                     entry point, sidebar navigation
academy/config.py          thresholds, paths, timezone
academy/curriculum.py      6 levels, 40 concepts, concept mastery, recommendations
academy/progress.py        persistence and lesson status (no Streamlit, testable)
academy/datasets.py        dataset registry: upload > configured path > simulated
academy/data.py            simulated football data generators
academy/shrinkage.py       Level 1 engine: empirical Bayes, rolling-origin validation, gates
academy/quiz.py            grading
academy/runner.py          runs code, captures output and charts
academy/lessons/           one file per lesson (auto-discovered)
academy/labs.py            Python Lab exercises
academy/projects_data.py   8 projects with Learning Lab and Production Candidate tracks
academy/views/             one module per page (+ p1_bayes.py, curriculum_map.py)
tests/                     pytest suite
```
