# RoboDojo evaluation with remote inference

This adapter runs EMERGE against RoboDojo's official EvalEnv, native task layouts,
step limits, and partial-credit scoring. It uses ARX-X5 14-dimensional joint actions
and the RoboDojo Pi0.5 checkpoint; the LIBERO 7-dimensional policy is incompatible.

Install the `robodojo` optional dependencies for EMERGE; Open3D runs point-cloud
post-processing on the agent node even when VGGT/SAM3 inference is remote.

Use separate Python environments for Isaac Sim, EMERGE, and OpenPI. Keep the
simulator and controller on the same node, with one workspace per episode.
Run VGGT, SAM3, and Pi0.5 on remote inference GPUs. Camera images and calibration
travel over WebSocket; no shared image directory is needed. Saved perception
images are resized to 518x392 with intrinsics adjusted to match; Pi0.5 receives
native simulator images.

An eight-GPU simulator node can use two policy replicas by passing comma-separated
URLs to `scripts/eval_robodojo_agent.py --policy-server-url`; even and odd GPU IDs
use alternating replicas. Start with one worker per GPU. Set the perception URLs
in `subagents.objectLocation.vggtUrl` and `sam3Url` in the private agent config.

A `custom` provider whose `apiBase` ends in `/responses` uses the streaming
Responses API (text, images, tool calls, tool results). Other custom endpoints
continue to use Chat Completions. The Responses provider uses
`EMERGE_RESPONSES_PROXY`, then the existing HTTPS proxy environment variable.
Keep API keys outside this repository in a mode-0600 config file.

For standard seed-0 evaluation use the default task list and `--layouts native`:
54 task configurations represent 42 benchmark tasks and 2100 episodes. Missing
layouts fail validation instead of silently reducing the sample. Results retain
both official partial-credit score and binary success. Infrastructure errors stay
separate and are eligible for retry on `--resume`; completed failures are not
selectively rerun. `scripts/summarize_robodojo_eval.py RUN --expected-episodes 2100`
reports coverage before a run can be considered complete.

Before scaling, verify real calibrated images, both perception services, a native
Pi0.5 action, the VLM's image/tool roundtrip, and a complete scored episode. A
successful launcher exit alone does not prove success; inspect structured episode
status/results, because Isaac shutdown can mask earlier errors.
