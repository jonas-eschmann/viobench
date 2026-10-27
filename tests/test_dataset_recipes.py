"""Recipe preflight protects existing datasets before invoking either generator."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RECIPES = Path(__file__).resolve().parents[1] / "relpose/tools/datasets/generate/recipes"


@pytest.fixture
def recipe_runner(tmp_path):
    binary = tmp_path / "bin" / "relpose"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable}\n" + """
import json
import os
import sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['RECIPE_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
output = Path(args[args.index('-o') + 1])
output.mkdir(parents=True, exist_ok=True)
(output / 'episode.json').write_text('{}')
""")
    binary.chmod(0o755)
    output = tmp_path / "dataset with spaces"
    calls = tmp_path / "calls.jsonl"
    environment = {**os.environ, "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
                   "RELPOSE_DATASET_OUTPUT": str(output), "RECIPE_CALLS": str(calls)}

    def run(recipe):
        return subprocess.run(["bash", str(RECIPES / recipe)], env=environment, text=True, capture_output=True)

    return run, output, calls


@pytest.mark.parametrize("recipe", ["generate_public.sh", "generate_hidden.sh"])
@pytest.mark.parametrize("existing", ["real/old/episode.json", ".old-dataset", "file"])
def test_recipe_rejects_existing_output_before_any_generation(recipe_runner, recipe, existing):
    run, output, calls = recipe_runner
    marker = output if existing == "file" else output / existing
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("old data")

    result = run(recipe)

    assert result.returncode != 0
    assert "output must be absent or empty" in result.stderr
    assert "RELPOSE_DATASET_OUTPUT" in result.stderr
    assert not calls.exists(), "preflight must run before the first generation command"
    assert marker.read_text() == "old data"


@pytest.mark.parametrize("recipe,expected_calls", [("generate_public.sh", 5), ("generate_hidden.sh", 9)])
@pytest.mark.parametrize("empty_directory", [False, True])
def test_recipe_accepts_fresh_output_and_rejects_rerun(recipe_runner, recipe, expected_calls, empty_directory):
    run, output, calls = recipe_runner
    if empty_directory:
        output.mkdir()

    result = run(recipe)

    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == expected_calls
    assert all(Path(args[args.index("-o") + 1]).parent == output for args in commands)
    original_calls = calls.read_text()
    assert run(recipe).returncode != 0
    assert calls.read_text() == original_calls
