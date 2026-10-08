# Publication-readiness audit

The original scripts in this repository are useful prototypes, but they should not be treated as evidence for a paper without further validation. The canonical corrected implementation for the next stage is `publication_sarsa.py`.

## Scope of the corrected implementation

- Repeated two-player Prisoner's Dilemma with `R=3`, `T=5`, `S=0`, `P=1`.
- Memory-one state representation: `start`, `C`, or `D`, representing the opponent's previous action.
- Independent epsilon-greedy learners in self-play.
- Two-step SARSA with explicit finite-episode truncation.
- Seeded runs and joint-outcome counts (`CC`, `CD`, `DC`, `DD`).

This is a multi-agent learning experiment. Since both policies change during training, the environment seen by each learner is non-stationary. The code therefore makes no general convergence claim.

## Run the corrected tests

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

If pytest is not installed, the tests can still be inspected and run after installing it:

```bash
python -m pip install pytest
python -m pytest -q test_publication_sarsa.py
```

## Recommended experimental design

Before publication, run multiple independent seeds and report means with uncertainty intervals. Separate training from evaluation by setting exploration to zero during evaluation. Compare two-step SARSA with one-step SARSA and fixed-opponent baselines such as Always Cooperate, Always Defect, Tit-for-Tat, and Pavlov.

Do not conclude that cooperation has "emerged" from a single run. Report cooperation rate, average reward, and the frequency of `CC`, `CD`, `DC`, and `DD` outcomes over a declared evaluation window, together with the number of seeds and all hyperparameters.

## Known limitations

- The state contains only the opponent's last action; it cannot represent longer-memory strategies.
- Independent self-play does not provide a stationary single-agent environment.
- The current repository also contains historical exploratory scripts under dated filenames. They should be isolated from the final analysis and audited separately before use in a paper.
