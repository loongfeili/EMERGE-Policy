"""Small protocol smoke test; real model invocation is supplied by an adapter."""
from .protocol import ActionCandidate, RolloutRequest, RolloutResult
from .selector import AcWmSelector


def main() -> None:
    candidates = tuple(ActionCandidate(f"candidate-{i}", "smoke", ((float(i), 0.0),)) for i in range(2))
    req = RolloutRequest("smoke task", "/tmp/observation.jpg", candidates, "bridge_orig_lerobot")
    selector = AcWmSelector(
        lambda _r, c: RolloutResult(c.candidate_id, "success", f"/tmp/{c.candidate_id}.mp4"),
        lambda _t, c, _r: (0.2 + 0.5 * int(c.candidate_id[-1]), "deterministic smoke score"),
    )
    print(selector.select(req))


if __name__ == "__main__":
    main()
