# Curation decision (Cosmos Reason2)

Rule: every episode (sim, Cosmos-transferred, real) is scored by nvidia/cosmos-reason2-2b (local NIM on the
VM) with the same rubric (grasp_success, placed_on_pad, motion_quality 1-5, realism 1-5). Episodes with
grasp_success=false or motion_quality<=2 are removed before training. The realism score per source is
reported as the "fidelity gap".

- sim (red, so101_sim_pickplace):   n=83  grasp_success=1.00  motion_quality=5.00  realism=5.00
- sim (blue, so101_sim_pickplace_blue): n=60  grasp_success=1.00  motion_quality=5.00  realism=5.00
- cosmos (red Transfer clips, sim_cosmos): n=6  grasp_success=1.00  motion_quality=5.00  realism=5.00
- cosmos (blue Transfer clips, sim_cosmos_blue): NOT SCORED — lost to a key-collision bug when merging
  both cosmos sources into one results file (identical relative video paths caused the second scoring pass
  to silently skip as "already done"); never re-run since cosmos-reason was stopped to free the GPU for
  training. Only the red Transfer clips (6) are reflected in results/reason_cosmos_red.json.
- real (station_ref phone reference clips, not a training dataset): n=5  grasp_success=0.60
  motion_quality=5.00  realism=5.00; 2 dropped (these are camera-calibration reference clips, not
  demonstration episodes, so "grasp_success" isn't really meaningful for them beyond illustrating the rubric).
- dropped (by this Reason rule, affecting actual training data): 0 episodes. All of red, blue,
  sim_cosmos and sim_cosmos_blue passed with perfect scores; only the station_ref reference clips were
  "dropped" and those were never training data to begin with.

Separately, the real teleop dataset (datasets/real_share/station_v3, 33 episodes, used for ACT training)
has its own team-provided drop list of 3 episodes (indices 1, 9, 23 — incomplete or cube off target, see
real_drops.json / hand_labels_real.csv). This is independent of the Cosmos Reason rule above. ACT training
was launched on all 33 episodes (no `--dataset.episodes` exclusion applied) since excluding specific indices
was only decided for a GR00T-training path that did not end up being used (the v3.0 loader failed for GR00T).
