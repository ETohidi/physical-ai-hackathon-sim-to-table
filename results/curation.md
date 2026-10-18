# Curation decision (Cosmos Reason2)

Rule: every episode (sim, Cosmos-transferred, real) is scored by nvidia/cosmos-reason2-8b with the same rubric
(grasp_success, placed_on_pad, motion_quality 1-5, realism 1-5). Episodes with grasp_success=false or
motion_quality<=2 are removed before training. The realism score per source is reported as the "fidelity gap".

Fill in after running scripts/cosmos_reason_score.py:
- sim:    n=…  grasp_success=…  motion_quality=…  realism=…
- cosmos: n=…  grasp_success=…  motion_quality=…  realism=…
- real:   n=…  grasp_success=…  motion_quality=…  realism=…
- dropped: … episodes (list ids)
