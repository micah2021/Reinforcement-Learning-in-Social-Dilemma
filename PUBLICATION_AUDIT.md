# Publication-readiness audit

The merged two-step SARSA implementation is now accompanied by controlled
fixed-opponent evaluation in `publication_experiments.py`.

## Corrected scope

- Repeated two-player Prisoner's Dilemma with `R=3`, `T=5`, `S=0`, `P=1`.
- Memory-one state representation: `start`, `C`, or `D`, representing the opponent's previous action.
- Independent epsilon-greedy learner in self-play or against a fixed policy.
- Two-step SARSA with explicit finite-episode return truncation.
- Seeded runs and joint-outcome counts (`CC`, `CD`, `DC`, `DD`).
- Evaluation explicitly sets `epsilon=0` and performs no learning updates.

## Run validation

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

## Recommended experimental design

Run multiple independent seeds and report means with uncertainty intervals. Compare
two-step SARSA with one-step SARSA and fixed-opponent baselines: Always Cooperate,
Always Defect, Tit-for-Tat, and Pavlov. Declare the training length, evaluation
length, random seeds, payoff matrix, state representation, and all hyperparameters.

Report cooperation rate, average reward, and the frequency of `CC`, `CD`, `DC`, and
`DD` outcomes over the evaluation window. Do not conclude that cooperation has
"emerged" from a single run. Independent self-play is non-stationary, so empirical
stabilization must not be described as a general convergence theorem.

## Known limitations

- The state contains only the opponent's last action; it cannot represent longer-memory strategies.
- Independent self-play does not provide a stationary single-agent environment.
- Pavlov requires the previous joint outcome; the included baseline is an explicit compact heuristic and must be described as such.
- Historical exploratory scripts under dated filenames should be isolated from the final analysis and audited separately.
