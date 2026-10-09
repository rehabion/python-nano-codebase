# Roadmap: from a validated baseline to novel architectures

This repository is deliberately a **rigorous supervised baseline**. Its value
for a publication is twofold: (1) it is reproducible and honestly evaluated,
and (2) it is the control any novel architecture must beat. The advanced ideas
below are staged so each is justified by a measurable gap this baseline
exposes.

### Stage 0 — Baseline (this repo)
Nested-CV benchmark of 8 algorithms, engineered nano-QSAR descriptors,
leave-one-family-out generalisation, permutation importance. **Done.**

### Stage 1 — Richer representation
- Graph representation of particle core + surface ligands (atoms/groups as
  nodes) feeding a GNN encoder; compare against the tabular baseline on the
  same splits and metrics defined here.
- Self-supervised pre-training on unlabelled material descriptors to improve
  the cross-family LOFO numbers this repo reports.

### Stage 2 — Dynamic protein corona
- Replace static descriptors with a time-indexed corona representation; a
  cross-attention layer conditions particle features on a medium/plasma
  context vector. Evaluate whether corona-aware inputs reduce the LOFO gap.

### Stage 3 — Biokinetics
- Neural-ODE module over a compartmental (PBPK-style) state to predict
  continuous accumulation trajectories rather than a single binary label.
  Requires time-resolved uptake data.

### Stage 4 — Causal / regulatory layer
- Move beyond permutation importance (association) to a structural-causal
  model with counterfactual checks, so claims like
  "surface charge → oxidative stress → apoptosis" are testable, not merely
  correlational.

**Guiding principle:** every stage reuses this repo's evaluation harness
(nested CV, hold-out, LOFO, the same metric panel) so improvements are
measured, not asserted. Keep real experimental data flowing in via the
loader before drawing any biological conclusion.
