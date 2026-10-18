# Sim to Table with a multi-level digital twin

This project builds a multi-level digital twin for sim-to-real pick-and-place on a SO-101 arm:
simulation in Isaac Sim gives a coarse, cheap-to-scale overview of the task; real teleoperated
episodes on the physical station provide the fine-grained ground truth; Cosmos Transfer bridges
the two by restyling simulated video toward the real station's visual domain; Cosmos Reason acts
as an automatic auditor, scoring and curating episodes before they reach training; and a GR00T
N1.7 policy is post-trained on the resulting mixed dataset, all orchestrated on Nebius AI Cloud.

Status: work in progress, hackathon day.

## Layout

- `so101/` — vendored simulation + training environment (Isaac Lab / Isaac Sim / GR00T), see its README for details.
- `scripts/` — project-level scripts (Cosmos Reason scoring, Cosmos Transfer batch job, dataset manifest).
- `patches/` — patches applied on top of vendored/upstream code (e.g. Isaac Lab API compatibility).
- `results/` — manifests, curation notes, and other run outputs.
