# Working instructions

## Privacy and communication

- Always use professional, neutral, precise language in responses, documentation,
  comments, commit messages, and other output.
- Do not repeat, quote, paraphrase into documentation, or imitate the user's
  profanity, insults, accusations, or emotional language. Record the underlying
  requirement in neutral terms.
- Do not introduce suggestions of user fault, responsibility, or intent that
  the user has not expressed. Address corrections directly and take responsibility
  for assistant errors without speculating about the user's emotions or motives.
- The source research workspace is private. Do not include its repository
  address, internal paths, commit history, unrelated source, or internal notes
  in public-facing material. Publication requires explicit owner authorization.

## Authorship and purpose

Sparse Parallel Q-Learning is Jaeyoung Oh's research project. The original
formulation, frequency-based WTA method, code, and initial empirical work were
developed independently through three months of full-time work before the
later experiment-setup and documentation phase. Preserve that history and
credit. Assistance with experiments or writing does not confer research
authorship. Do not add AI bylines or acknowledgments without an explicit request.

The project compares selection and averaging in parallel tabular Q-learning
on LunarLander-v3 with truly sparse terminal-only rewards. Emphasize what the
comparison reveals and how the findings might inform deep RL. Do not market
the absence of neural networks, claim universal superiority, or assert a
world-first result without evidence.

Jaeyoung independently developed the reverse replay step. Prior reverse replay
is acknowledged in Lin (1992), Section 2.3, p. 299; related planning ideas are
explained in Sutton and Barto (2018), Chapter 8. See docs/related_work.md.
Use "reverse-order experience replay" in technical prose, while recognizing
"backward induction" as the historical code label.

## Maintenance

- Place the author name once at the top of the README. Do not repeat the
  author name or narrate personal development history in method, results, or
  literature sections. Keep those sections focused on the work. Retain accurate
  structured author information in citation and package metadata.
- Follow research writing conventions: a plain author line, impersonal
  technical prose, and methods, results, interpretation, and limitations.
  Keep development history in a brief provenance note. Do not explain trivial
  code snippets or implementation plumbing in research exposition; state the
  experimental protocol and substantive algorithmic choices directly.
- Never use em dashes in responses, titles, or documentation.
- Use descriptive function names without a leading underscore. Standard
  dunder methods such as `__init__` are fine.
- Update documentation with changes to behavior, settings, commands, or findings.
- Keep public runtime code clean and usable: remove experimental scratch
  comments, commented-out alternatives, debug leftovers, and unused interface
  parameters. Put supported experimental settings in explicit configurations.
  Preserve the learning procedure and verify behavioral equivalence when
  cleaning executable code.
- Keep the public package focused on reproducible experiments. Retain durable
  metrics, explicit run configurations, frozen checkpoints, and evaluation
  records. Do not reintroduce personal convenience servers, implicit checkpoint
  loading, per-step debug displays, or ad hoc testing modes.
- This is a demonstration and research repository. Keep the runtime direct
  and minimal. Do not add test suites, smoke-check tools, CI, development
  reports, generic orchestration frameworks, or unused compatibility paths.
  Necessary local verification belongs outside the repository.
- Keep facts, owner-reported observations, working hypotheses, and formal
  claims distinct. Visit frequency is not proof that a Q-value is correct.
- Preserve the original learning procedure. Do not replace WTA with averaging
  or silently alter counts, tie-breaking, reward gating, or replay order.
- Do not launch a full study to validate packaging or documentation. Use only
  focused checks appropriate to the change, without adding verification tooling.
- A question or observation about training is not authorization to restart it.
- Do not commit or push without the owner's explicit authorization for that step.
- Dependency attribution is separate from authorship of the research.

## Algorithm contract

The `make_sparse` conditional is a human-readable implementation convenience,
not a research contribution. Public writing should state the reward definition
directly without discussing the flag or its conditional syntax.

The learner uses `make_sparse = True`: zero nonterminal Bellman reward and
native terminal `+100/-100`, in both online updates and replay. Dense raw
returns are reporting only. Do not change this explicit gate as a cleanup.

The reference has sixteen MPI workers, fresh empty Q-tables, a 64,000-step local
threshold, alpha 0.05, gamma 0.99, and epsilon 0.03 decreasing linearly to zero
at epoch index 100. Each worker finishes native episodes until its local count
exceeds the threshold. Gymnasium truncation is recorded but intentionally
ignored for training termination. There is no artificial idle penalty.

The last 100,000 transitions are replayed in reverse order, then cleared,
before Q-table synchronization. Counts start at one, increment only during
online action selection, and reset after synchronization. WTA selects each
entry from the largest-count rank; equal counts choose the lowest rank.
Missing Q-values default to zero, counts to one. Random selection uses a
separate root-only RNG and sorted keys/candidates. See docs/method.md.

The completed comparison holds the budget at 264 completed epochs for all four
methods and paired seeds 101, 102, 103. It has nine monitoring checkpoints with
100 common evaluation episodes each, plus 200 separate final episodes per run.
The evaluation-only 100,000-step guard censors unresolved episodes. Censored
episodes are neither crashes nor omitted from denominators.

## Repository map and verification

- `sparse_parallel_q/train.py`: tabular training loop and direct MPI entry point.
- `core.py`: four aggregation rules and shared MPI synchronization.
- `comparison_support.py`: seeds, per-epoch research records, frozen checkpoints,
  and the epoch budget.
- `compare_aggregations.py`: frozen study, optional condition/seed subset,
  source records and sequential execution of fresh runs.
- `comparison_evaluate.py`: frozen greedy evaluation with explicit seed banks.
- `analyze.py`: analysis of portable records and newly completed studies.
- `results/2026-11-12/`: original study records and historical component fingerprints.

The existing study was completed before the standalone extraction. Do not
label its results as a new run of this package. The included historical bundle
contains outcomes and source fingerprints, not the historical source files.
Exported configuration retains scientific settings and omits display-only
settings; the export metadata records that scope.
Preserve these records and hashes. New standalone studies archive their own
source snapshots. Generated data and checkpoints belong under ignored `data/`
directories.
Historical names do not establish checkpoint provenance. The study launcher
always starts fresh. Run an interrupted experiment again in a new output
directory; partial results remain available for inspection.

The trainer requires a study controller. The controller owns seeded resets,
aggregation selection, durable metrics, frozen checkpoints, and the epoch
budget. Training always initializes empty Q-tables and count tables.

Use `spq-compare --dry-run` to inspect the scientific configuration and
`spq-analyze` to reproduce the recorded results. Follow README installation
steps for Box2D and Open MPI.
