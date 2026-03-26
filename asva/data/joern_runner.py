from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_JAVA_HOME = Path("/usr/lib/jvm/java-11-openjdk-amd64")


class JoernRunner:
    def __init__(
        self,
        joern_cli_dir,
        script_path,
        timeout_sec=180,
        verbose=0,
    ):
        self.joern_cli_dir = Path(joern_cli_dir)
        self.script_path = Path(script_path)
        self.timeout_sec = timeout_sec
        self.verbose = verbose
        self.joern_bin = self.joern_cli_dir / "joern"

    def materialize_source(self, sample, graph_dir):
        graph_dir = Path(graph_dir)
        sample_id = str(sample.get("id", "") or "").strip() or "unknown"
        source_path = graph_dir / f"{sample_id}.c"
        code = str(sample.get("func_before", "") or "")
        if (not source_path.exists()) or source_path.read_text(encoding="utf-8", errors="ignore") != code:
            source_path.write_text(code, encoding="utf-8")
        return source_path

    def generate(self, source_path):
        source_path = Path(source_path)
        node_path = Path(f"{source_path}.nodes.json")
        edge_path = Path(f"{source_path}.edges.json")
        if node_path.exists() and edge_path.exists():
            return True
        if not self.joern_bin.exists():
            LOGGER.warning("Joern binary missing at %s", self.joern_bin)
            return False
        if not self.script_path.exists():
            LOGGER.warning("Joern script missing at %s", self.script_path)
            return False
        env = os.environ.copy()
        if "JAVA_HOME" not in env and DEFAULT_JAVA_HOME.exists():
            env["JAVA_HOME"] = str(DEFAULT_JAVA_HOME)
            env["PATH"] = f"{DEFAULT_JAVA_HOME / 'bin'}:{env.get('PATH', '')}"
        command = [
            str(self.joern_bin),
            "--script",
            str(self.script_path),
            f"--params=filename={source_path}",
        ]
        LOGGER.info("Running Joern for missing graph: %s", source_path.name)
        try:
            result = subprocess.run(
                command,
                cwd=self.joern_cli_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            LOGGER.warning("Joern timed out for %s after %ss", source_path, self.timeout_sec)
            return False
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            LOGGER.warning(
                "Joern failed for %s with code %s%s%s",
                source_path,
                result.returncode,
                f"; stderr={stderr[:400]}" if stderr else "",
                f"; stdout={stdout[:400]}" if stdout else "",
            )
            return False
        if not (node_path.exists() and edge_path.exists()):
            LOGGER.warning("Joern finished but graph JSON is still missing for %s", source_path)
            return False
        return True


def _generate_single_sample(sample, graph_dir, joern_runner):
    sample_id = str(sample.get("id", "") or "").strip()
    source_path = joern_runner.materialize_source(sample, graph_dir)
    generated = joern_runner.generate(source_path)
    return sample_id, generated, str(source_path)


def precompute_missing_joern_graphs(
    samples,
    repository,
    joern_runner,
    workers=8,
):
    queued_samples = []
    seen_ids = set()
    existing = 0
    for sample in samples:
        sample_id = str(sample.get("id", "") or "").strip()
        if not sample_id or sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        match = repository.match(sample)
        if match.node_path and match.edge_path:
            existing += 1
            continue
        queued_samples.append(sample)
    generated_ids = []
    failed_ids = []
    if not queued_samples:
        return {
            "existing": existing,
            "queued": 0,
            "generated": 0,
            "failed": 0,
            "generated_ids": generated_ids,
            "failed_ids": failed_ids,
        }
    if workers <= 1:
        for sample in queued_samples:
            sample_id, generated, source_path = _generate_single_sample(sample, repository.graph_dir, joern_runner)
            if generated:
                generated_ids.append(sample_id)
                repository._register_key(Path(source_path).name)
            else:
                failed_ids.append(sample_id)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sample = {
                executor.submit(_generate_single_sample, sample, repository.graph_dir, joern_runner): sample
                for sample in queued_samples
            }
            for future in as_completed(future_to_sample):
                sample_id, generated, source_path = future.result()
                if generated:
                    generated_ids.append(sample_id)
                    repository._register_key(Path(source_path).name)
                else:
                    failed_ids.append(sample_id)
    return {
        "existing": existing,
        "queued": len(queued_samples),
        "generated": len(generated_ids),
        "failed": len(failed_ids),
        "generated_ids": generated_ids,
        "failed_ids": failed_ids,
    }
