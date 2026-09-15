"""One authorized public canary, resumed from durable state without relaunching it.

This is host orchestration. Workloads remain unaware of the supervisor. Launchd
runs this process under caffeinate; the TReqs target also shuts down when idle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TERMINAL = {"COMPLETED", "FAILED", "STOPPED", "CANCELLED", "CANCELED", "EXPIRED"}
INSTANCE_TERMINAL = {"stopped", "failed", "terminated"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def clean_environment():
    env = dict(os.environ)
    # A stale inherited API token must not override the saved interactive login.
    env.pop("TREQS_API_TOKEN", None)
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return env


class Actions:
    def __init__(self, plan, root):
        self.plan, self.root = plan, root
        self.env = clean_environment()

    def command(self, argv, *, cwd=None, timeout=60):
        result = subprocess.run(argv, cwd=cwd or self.plan["controlWorkspace"],
                                env=self.env, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            # Never persist credential-bearing subprocess output in the event stream.
            raise RuntimeError(f"{Path(argv[0]).name} {argv[1:3]} exited {result.returncode}")
        return result.stdout

    def cli(self, *args):
        return json.loads(self.command([self.plan["treqs"], "--json", *args]))

    def instances(self):
        value = self.cli("compute", "targets", "instances", self.plan["targetId"],
                         "--owner", "reproducible-ai")
        return value if isinstance(value, list) else value["instances"]

    def find(self, kind, **criteria):
        found = []
        for offset in range(0, 10000, 100):
            args = [kind, "list", "--limit", "100"]
            if kind == "tr":
                args += ["--offset", str(offset)]
            rows = self.cli(*args)
            found.extend(row for row in rows if all(row.get(k) == v for k, v in criteria.items()))
            if kind == "jobs":
                if len(rows) == 100 and not found:
                    raise RuntimeError("Job inventory truncated; refusing another launch")
                break
            if len(rows) < 100:
                break
        else:
            raise RuntimeError("Could not reconcile complete request/job inventory")
        if len(found) > 1:
            raise RuntimeError("Ambiguous request/job identity; refusing another launch")
        return found[0] if found else None

    def prepare(self):
        from public_canary_verify import prepare_repository
        target = next(x for x in self.cli("compute", "targets", "list", "--owner", "reproducible-ai")
                      if x["id"] == self.plan["targetId"])
        if not target.get("autoShutdownEnabled") or target.get("idleTimeoutMinutes", 999) > 15:
            raise RuntimeError("Target must have automatic idle shutdown within 15 minutes")
        resources = target["resources"]
        if resources["instanceType"] != "g7e.2xlarge" or resources["amiId"] != self.plan["amiId"]:
            raise RuntimeError("Target differs from the authorized recipe")
        for directory, expected in ((self.plan["sourceWorkspace"], self.plan["sourceCommit"]),
                                    (self.plan["harnessWorkspace"], self.plan["harnessCommit"])):
            if self.command(["git", "rev-parse", "HEAD"], cwd=directory).strip() != expected:
                raise RuntimeError("Source revision changed")
            if self.command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory).strip():
                raise RuntimeError("Source has uncommitted changes")
        remote = self.command(["git", "ls-remote", "origin", f"refs/heads/{self.plan['sourceBranch']}"],
                              cwd=self.plan["sourceWorkspace"])
        if remote.split()[0] != self.plan["sourceCommit"]:
            raise RuntimeError("Remote source branch moved")
        prepare_repository(self.plan, self.root)

    def create(self):
        return self.cli("tr", "create", "--title", self.plan["title"],
                        "--status", "open", "--workflow-path", ".treqs/workflows/droid-canary.yaml",
                        "--compute-target", self.plan["targetId"],
                        "--source-branch", self.plan["sourceBranch"],
                        "--source-commit", self.plan["sourceCommit"],
                        "--lineage-mode", "public", "--yes")

    def queue(self, request_id):
        return self.cli("tr", "queue", request_id)

    def job(self, job_id):
        return self.cli("jobs", "show", job_id)

    def stop(self, job_id):
        return self.cli("jobs", "stop", job_id)

    def logs(self, job_id):
        path = self.root / "workload.jsonl"
        cursor = 0
        if path.exists():
            lines = path.read_text().splitlines()
            if lines:
                cursor = json.loads(lines[-1])["result"]["nextSequence"]
        text = self.command([self.plan["treqs"], "jobs", "logs", job_id, "--jsonl",
                             "--from-sequence", str(cursor), "--poll-timeout-ms", "1000"])
        row = json.loads(text)
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return row["result"]

    def verify(self, state):
        from public_canary_verify import verify_release
        job = self.job(state["jobId"])
        tasks = self.cli("jobs", "tasks", state["jobId"])
        if isinstance(tasks, dict):
            tasks = tasks["tasks"]
        if len(tasks) != 7 or any(t["status"].upper() != "COMPLETED" for t in tasks):
            raise RuntimeError("All seven tasks must be completed")
        if job.get("lineagePublicationMode") != "public":
            raise RuntimeError("Job lineage is not public")
        if job.get("lineagePublicationStatus") != "published":
            if job.get("lineagePublicationStatus") == "failed" and state.get("republishAttempts", 0) < 3:
                state["republishAttempts"] = state.get("republishAttempts", 0) + 1
                atomic_json(self.root / "state.json", state)
                self.cli("jobs", "republish-lineage", state["jobId"])
            raise RuntimeError("Waiting for public lineage publication")
        result = self.logs(state["jobId"])
        if result["hasMore"]:
            raise RuntimeError("Draining complete workload log before verification")
        return verify_release(self.plan, self.root, job)

    def publish(self, state):
        from public_canary_verify import publish_notes
        return publish_notes(self.plan, self.root, state, self)


class Supervisor:
    def __init__(self, plan, root, actions, clock=time.time):
        self.plan, self.root, self.actions, self.clock = plan, Path(root), actions, clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "phase": "prepare", "planSha256": digest, "createdAt": clock(), "instances": {}}
        if self.state["planSha256"] != digest:
            raise RuntimeError("Saved plan differs; refusing to launch or resume")

    def save(self, event):
        self.state["updatedAt"] = self.clock()
        atomic_json(self.path, self.state)
        row = {"at": datetime.now(timezone.utc).isoformat(), "event": event,
               "phase": self.state["phase"], "jobId": self.state.get("jobId")}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    def transition(self, phase):
        self.state["phase"] = phase
        self.state["phaseAt"] = self.clock()
        self.state.pop("lastError", None)
        self.save(phase)

    def observe_instances(self):
        for instance in self.actions.instances():
            if instance["id"] not in self.state["baselineInstances"]:
                self.state["instances"][instance["id"]] = instance
        cost = 0.0
        active = False
        for instance in self.state["instances"].values():
            stopped = instance["status"].lower() in INSTANCE_TERMINAL
            active |= not stopped
            reported = (instance.get("totalCostCents") or 0) / 100
            elapsed = 0.0
            if instance.get("launchedAt") and (not stopped or instance.get("terminatedAt")):
                end = timestamp(instance["terminatedAt"]) if instance.get("terminatedAt") else self.clock()
                elapsed = max(0, end - timestamp(instance["launchedAt"]))
            estimate = elapsed / 3600 * self.plan["conservativeHourlyUsd"]
            cost += reported if stopped and instance.get("totalCostCents") is not None else max(reported, estimate)
        self.state["observedCostUsd"] = round(cost, 4)
        self.state["allocationActive"] = active
        self.save("cost.observed")
        return cost, active

    def tick(self):
        s, a = self.state, self.actions
        phase = s["phase"]
        if phase in {"complete", "failed"}:
            return False
        if phase == "prepare":
            baseline = a.instances()
            if any(i["status"].lower() not in INSTANCE_TERMINAL for i in baseline):
                raise RuntimeError("Target is busy; waiting before taking baseline")
            s["baselineInstances"] = [i["id"] for i in baseline]
            a.prepare()
            self.transition("create")
        elif phase == "create":
            request = a.find("tr", title=self.plan["title"])
            if request is None:
                self.transition("creating")  # intent is durable before the external effect
                request = a.create()
            s["requestId"] = request["id"]
            self.transition("queue")
        elif phase == "creating":
            request = a.find("tr", title=self.plan["title"])
            if request:
                s["requestId"] = request["id"]
                self.transition("queue")
            elif self.clock() - s["phaseAt"] > 300:
                s["failure"] = "Ambiguous request creation; no paid launch was repeated"
                self.transition("failed")
        elif phase == "queue":
            job = a.find("jobs", trainingRequestId=s["requestId"])
            if job is None:
                self.transition("queueing")
                job_id = a.queue(s["requestId"])["jobId"]
            else:
                job_id = job["id"]
            s["jobId"] = job_id
            s["queuedAt"] = self.clock()
            self.transition("monitor")
        elif phase == "queueing":
            job = a.find("jobs", trainingRequestId=s["requestId"])
            if job:
                s["jobId"] = job["id"]
                s["queuedAt"] = s["phaseAt"]
                self.transition("monitor")
            elif self.clock() - s["phaseAt"] > 300:
                # Keep reconciling: an ambiguous queue may have provisioned compute.
                s["stopReason"] = "Ambiguous queue acknowledgement"
                self.observe_instances()
        elif phase == "monitor":
            # Job failure must not prevent allocation cleanup monitoring.
            job = a.job(s["jobId"])
            s["job"] = {key: job.get(key) for key in (
                "id", "status", "startedAt", "completedAt", "onDemandInstanceId",
                "lineagePublicationMode", "lineagePublicationStatus", "lineagePublishedSessionHash",
                "lineagePublishedUrl", "lineagePublicationCounts")}
            cost, active = self.observe_instances()
            status = job["status"].upper()
            if status not in TERMINAL:
                elapsed = self.clock() - s["queuedAt"]
                if cost >= self.plan["stopAtUsd"] or elapsed > self.plan["maxJobSeconds"]:
                    s["stopReason"] = "Budget or elapsed-time stop threshold reached"
                if s.get("stopReason"):
                    self.save("stop.requested")
                    a.stop(s["jobId"])
                self.save("job.observed")
                return True
            if active:
                self.save("waiting.for.idle.shutdown")
                return True
            if status != "COMPLETED" or s.get("stopReason") or cost > self.plan["budgetUsd"]:
                s["failure"] = s.get("stopReason") or f"Job ended {status}; cost ${cost:.2f}"
                self.transition("failed")
            elif not s["instances"] or job.get("onDemandInstanceId") not in s["instances"]:
                raise RuntimeError("Completed job lacks a matching settled allocation")
            elif any(i.get("launchedAt") and i.get("totalCostCents") is None
                     for i in s["instances"].values()):
                raise RuntimeError("Waiting for final allocation cost accounting")
            else:
                self.transition("verify")
        elif phase == "verify":
            s["verification"] = a.verify(s)
            atomic_json(self.root / "verification.json", s["verification"])
            self.transition("publish")
        elif phase == "publish":
            s["publication"] = a.publish(s)
            self.transition("complete")
        else:
            raise RuntimeError(f"Unknown phase: {phase}")
        return True

    def error(self, exc):
        self.state["lastError"] = f"{type(exc).__name__}: {exc}"
        self.save("operation.retry")
        # API outages must not turn off the budget guard. Retry stop independently.
        if self.state.get("jobId") and self.state["phase"] == "monitor":
            if self.clock() - self.state["queuedAt"] > self.plan["maxJobSeconds"]:
                self.state["stopReason"] = "Elapsed-time guard during monitoring failure"
                self.save("stop.requested")
                try:
                    self.actions.stop(self.state["jobId"])
                except Exception:
                    self.save("stop.retry")
        elif self.state["phase"] in {"prepare", "verify", "publish"}:
            if self.clock() - self.state.get("phaseAt", self.state["createdAt"]) > 10800:
                self.state["failure"] = self.state["lastError"]
                self.transition("failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(plan, root, Actions(plan, root))
        while True:
            try:
                if not supervisor.tick():
                    break
            except Exception as exc:
                supervisor.error(exc)
            if args.once:
                break
            time.sleep(20)


if __name__ == "__main__":
    main()
