# Related work and attribution

## Reverse-order experience replay

Long-Ji Lin's 1992 paper explicitly discusses replaying experiences backward in
time in Section 2.3, p. 299, and uses backward replay with lambda zero. This is
a close precedent for the order of replay updates. Lin's experiments use
connectionist AHC and Q-learning systems with their own replay-selection rules.
[Original paper](https://link.springer.com/content/pdf/10.1007/BF00992699.pdf#page=7)

Sutton and Barto's *Reinforcement Learning: An Introduction*, second edition
(2018), Chapter 8, explains planning with tabular methods. Section 8.4 describes
backward focusing and prioritized sweeping. These methods use models and
priorities to propagate changes through predecessor states. The textbook's
historical remarks credit prioritized sweeping to Moore and Atkeson (1993)
and Peng and Williams (1993).
[Textbook](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

Here, each worker applies one-step Q-learning updates to recorded transitions
in reverse chronological order at an epoch boundary. It has no predecessor
model or priority queue. Reverse replay is shared by all four experimental
conditions, so the study evaluates aggregation within that local-learning
procedure. It does not claim that reverse replay itself is new or measure its
effect separately.

## Sparse-reward LunarLander

The source review conducted on November 12, 2026 distinguished true terminal
outcomes from dense rewards accumulated and paid at termination. It also
checked demonstrations, pretraining, episode limits, and success predicates.
The reviewed external implementations were not rerun.

**PPO with demonstrations.** Libardi, De Fabritiis, and Dittert, *Guided
Exploration with Proximal Policy Optimization using a Single Demonstration*
(ICML 2021), report successful learning on a terminal-only LunarLander variant
using ten demonstrations for that task. The authors report sporadic success
for ordinary PPO in one of five training seeds. No separate numerical final
success rate for that seed was located. The released custom environment uses
terminal `0/1` rewards and a 300-step registration limit, so it differs from
the native `-100/+100` protocol here.
[Paper](https://proceedings.mlr.press/v139/libardi21a/libardi21a.pdf),
[supplement](https://proceedings.mlr.press/v139/libardi21a/libardi21a-supp.pdf),
[reward implementation](https://github.com/compsciencelab/ppo_D/blob/1870c908f498ceb29295e5625ff5598bed82cbb3/bullet/sparse_lunar_lander.py#L321-L363),
[environment registration](https://github.com/compsciencelab/ppo_D/blob/1870c908f498ceb29295e5625ff5598bed82cbb3/main/train_ppo_bullet.py#L40-L44).

**Deep Q-learning with intermediate-reward assistance.** Shirali, Schubert,
and Alaa's *Pruning the Way to Reliable Policies: A Multi-Objective Deep
Q-Learning Approach to Critical Care* reports a LunarLander variant with a
terminal objective. Intermediate-reward critics are trained first and used
to restrict actions during sparse-objective learning. Its changed observation
and action spaces and reward-assisted training distinguish it from learning
solely from terminal feedback from an empty initialization.
[Paper](https://arxiv.org/html/2306.08044v3#S5),
[training commands](https://github.com/alishiraliGit/multi-criteria-deep-q-learning/blob/e88fcb9d463dade0edbe5fb3e2aed6fd4cd13e73/LL_DQN_Training.py#L151-L177).

**Evolutionary architecture search.** *Exploring Sparse Reward Environments
with Weight Agnostic Neural Networks* uses terminal-only LunarLander fitness,
with a time-discounted positive outcome and zero failure reward. Its reported
positive fitness supports successful landings, but the reported quantity
1135.37 is an inverse transformation of mean fitness, not a landing percentage.
Its objective, rollout horizon, and evolutionary procedure differ from this
study.
[Project report](https://github.com/Tobi-Tob/Sparse-RL-Wann/blob/a03c6cfdce74f4b9e24cd5c37b8b503e919ca252/Sparse-RL-WANN.pdf),
[reward implementation](https://github.com/Tobi-Tob/Sparse-RL-Wann/blob/a03c6cfdce74f4b9e24cd5c37b8b503e919ca252/domain/lunar_lander.py#L587-L622).

These sources establish relevant work on sparse feedback. They do not form a
matched benchmark against this repository's experiment, and this review does
not establish a claim of being the first solution. The motivation here is to
understand differences between ways of sharing local estimates and investigate
whether those insights can inform deep RL in similarly sparse environments.

## Bibliographic references

1. Lin, Long-Ji. 1992. *Self-improving reactive agents based on reinforcement
   learning, planning and teaching*. Machine Learning 8: 293-321.
   [DOI](https://doi.org/10.1007/BF00992699).
2. Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning: An
   Introduction*. Second edition. MIT Press. Chapter 8.
3. Libardi, Gabriele, Gianni De Fabritiis, and Sebastian Dittert. 2021. *Guided
   Exploration with Proximal Policy Optimization using a Single Demonstration*.
   Proceedings of Machine Learning Research 139: 6611-6620.
   [Proceedings](https://proceedings.mlr.press/v139/libardi21a.html).
