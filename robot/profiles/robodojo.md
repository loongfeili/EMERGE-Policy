# RoboDojo dual ARX-X5 embodiment

Driver: `robodojo`. Two six-joint arms, each with a gripper. Read the task
instruction and live proprioception from `ROBOT_STATE.md`. Stop when
`robots.robodojo.done` becomes true. Only RoboDojo determines the final score.

Use camera observations for task verification and object localization. The
head and two wrist cameras are calibrated in `robodojo_env`, the same frame
used by motion commands. Distances are metres and orientations are XYZW
quaternions. Do not infer object locations from benchmark code or assets.

Supported actions:

- `vla_execute`: `instruction` and positive `step`; optional `replan_steps`.
  Pi0.5 controls both arms using the RoboDojo checkpoint. Give it the full task
  instruction or a visually grounded subgoal. Each returned chunk is bounded
  by the remaining requested steps and the official episode budget.
- `move_to_pose`: `arm` (`left` or `right`), `position_m`, and
  `orientation_quat` (XYZW). Use localized geometry and leave collision clearance.
- `move_linear`: same target fields; move through Cartesian waypoints.
- `set_gripper`: `arm` and `opening` between 0 (closed) and 1 (open).
- `follow_arc`: `arm`, `center`, `axis`, `radius_m`, and `angle_deg` for a measured arc.
- `recover`: use the generic recovery action when a bounded motion fails.

Re-read `ROBOT_STATE.md` after each action. Update the plan from new images;
sub-agent judgements are advisory, not benchmark success labels. Do not reset
the environment, alter the task, or extend its step limit.
