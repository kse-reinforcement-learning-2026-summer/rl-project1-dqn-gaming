# Project 1: DQN — Playing Flappy Bird from Pixels

**Course:** Reinforcement Learning, Summer 2026 — Kyiv School of Economics  
**Instructor:** Yurii Hannich  
**Points:** 15 (all-or-nothing)

---

## Overview

In this project you will reproduce the core ideas from the foundational DQN paper:

> Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning.* [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

You will train a Deep Q-Network to play **Flappy Bird from raw pixels** — implementing the preprocessing pipeline, network architecture, and training algorithm described in the paper, then adapting it to the [FlappyBird-v0](https://github.com/markub3327/flappy-bird-gymnasium) environment.

In the final section, you will also implement the key improvement from:

> Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* [Nature 518, 529–533](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

**Your primary source of information is the paper itself.** Read it carefully — all implementation details (preprocessing, architecture, hyperparameters, algorithm) come from there.

---

## Repository Structure

```
├── README.md                ← You are here
├── project.ipynb            ← YOUR WORKING FILE
├── model.pt                 ← Your trained model (produced by notebook)
├── requirements.txt         ← Python dependencies
├── tests/                   ← Automated tests (run in CI)
├── .github/                 ← CI workflow configuration
├── .gitignore
└── LICENSE
```

### What you modify

| File | Description |
|------|-------------|
| `project.ipynb` | Your template notebook — follow instructions inside to complete. Do not add, delete or rearrange cells. Final version that you will commit must have same structure, so parser could correcty process your code for autotests |
| `model.pt` | Your trained model weights (~2.7 MB) — produced by the notebook |

### What you do NOT modify

Everything else. `tests/`, `requirements.txt`, `.github/`, `.gitignore`, `LICENSE` — all read-only.

> **Warning:** Any modifications to files outside `project.ipynb` and `model.pt` are logged by GitHub and **immediately visible to the instructor**. Tampering with tests or CI configuration is treated as an academic integrity violation.

---

## Workflow

### 1. Accept the GitHub Classroom invite

You are here.

### 2. Clone your repository

```bash
git clone https://github.com/kse-reinforcement-learning-2026-summer/rl-project1-dqn-gaming-<your-username>.git
cd rl-project1-dqn-gaming-<your-username>
```

> If you haven't authenticated Git yet, install [GitHub CLI](https://cli.github.com/) and run `gh auth login`.

### 3. Upload to Google Colab (Optional)

- Go to [colab.research.google.com](https://colab.research.google.com/)
- **File → Upload notebook** → select `project.ipynb`
- **Runtime → Change runtime type → T4 GPU**

### 4. Work through the notebook

Fill in `...` placeholders and write your code between `# --- Fill in below ---` and `# --- Fill in above ---` markers. The notebook guides you through each section.

Final training takes ~30 min on a free Colab T4 GPU.

### 5. Save your model

The notebook has a cell that saves `model.pt`. Make sure it runs successfully.

**Tip:** Also save to Google Drive as backup — Colab sessions can disconnect.

### 6. Download files back (Optional)

From Colab:
- **File → Download → Download .ipynb** → save as `project.ipynb` (overwrite the original)
- Download `model.pt` from the Colab files panel (or from Google Drive)

Place both files in your repository root folder.

### 7. Test locally (recommended)

Create a virtual environment matching CI (Python 3.11), install dependencies, and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test.py -v
```

This runs the exact same tests as GitHub Actions. Fix any failures before pushing.

### 8. Commit and push

```bash
git add project.ipynb model.pt
git commit -m "Complete DQN project"
git push origin main
```

### 9. Check the Actions tab

Go to your repository on GitHub → click the **Actions** tab:
- **Green ✓** = all tests pass → you're done
- **Red ✗** = something failed → click the run to see error details, fix, and push again

---

## Evaluation

### Automated checks (CI)

Every push to `main` triggers automated tests that verify:
- Your preprocessing pipeline, network architecture, epsilon schedule, and hyperparameters are correct
- Your `model.pt` achieves **median reward ≥ 30.0** over 30 evaluation episodes (deterministic policy, different random seeds)

The reward threshold is conservative — a correctly trained model typically achieves 40–50.

### After automated checks pass

1. **AI code review** — verifies your training code is consistent with the submitted model (you actually trained it)
2. **Plagiarism check** — code similarity analysis between all student submissions

All checks must pass to receive the 15 points.

### Push limit

You have **10 pushes** that trigger CI. Use them wisely — **always test locally before pushing.**

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| Model doesn't train (reward stays near 0) | Double-check your training loop against algorithm in the paper. Are you sampling from replay buffer? Computing TD targets correctly? |
| Training is unstable (reward oscillates wildly) | Expected, wait some time to see if it start converging |
| Hyperparameter tests fail | Re-read the paper's experimental setup. Some constants are intentionally downscaled — the notebook tells you which ones and by how much. |
| Performance test fails but training looked good | Make sure you saved the model from the **target network training** (last section), not the vanilla DQN. Also verify `model.pt` is committed: `git status`. |
| Colab disconnects mid-training | Save checkpoints to Google Drive periodically. |
| Tests pass locally but fail in CI | Verify both `project.ipynb` and `model.pt` are pushed (`git log --oneline`, check file list on GitHub). |

---

## AI Tools

All AI tools are **fully allowed** — chat assistants, coding agents, deep research, anything.

If you haven't used AI coding tools before, here are free options to try:
- [**Cursor**](https://cursor.sh/) — VS Code fork with built-in AI agent
- [**Windsurf**](https://windsurf.com/) — AI coding agent, free tier available
- [**GitHub Copilot**](https://github.com/features/copilot) — free for students via GitHub Education

You may push any AI-generated documentation files (`.md`, notes, etc.) alongside your code — this won't affect grading.

---

## Quick Reference

```bash
# Clone
git clone https://github.com/kse-reinforcement-learning-2026-summer/rl-project1-dqn-gaming-<username>.git

# Test locally
pytest tests/test.py -v

# Submit
git add project.ipynb model.pt
git commit -m "your message"
git push origin main
```

**Good luck!**
