"""Bundled Fig. 4 data and the rerun workflow."""
import json

import numpy as np
import pytest

from scripts.tune_fig4 import stage
from src.evaluation.fig4 import DATA, FIELDS, load_groups, manifest, selected_policies
from src.rl_env.noise_covariance import MODEL_VERSION


def test_bundled_records_cover_every_controller():
    expected = {"mlp": 5, "mlp_large": 5, "affine": 5, "differentiator": 1, "kalman": 1, "particle": 5}
    for scenario, families in load_groups().items():
        counts = {family: len(records) for family, records in families.items()}
        assert counts == (expected | {"none": 1} if scenario == "hot_delta12p5" else expected)


def test_new_tuning_is_used_without_replacing_paper_inputs(tmp_path):
    tuning = json.loads((DATA / "tuning.json").read_text())
    tuning["selected"]["nominal"]["particle"]["parameters"]["depth"] = 0.456
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps(tuning))
    originals = manifest()["policies"]
    selected = selected_policies(path)
    for old, new in zip(originals, selected):
        if old["family"] == "particle":
            assert old["parameters"]["filter_seed"] == new["parameters"]["filter_seed"]
            if old["scenario"] == "nominal":
                assert new["parameters"]["depth"] == 0.456
                assert old["parameters"]["depth"] != 0.456
    assert manifest()["policies"] == originals
    # A subset evaluation needs only its own tuning selections.
    path.write_text(json.dumps({"selected": {"nominal": {"particle": tuning["selected"]["nominal"]["particle"]}}}))
    subset = selected_policies(path, "nominal", ["particle"])
    assert len(subset) == 5
    assert all(p["parameters"]["depth"] == 0.456 for p in subset)


def test_tuning_does_not_reuse_old_physics_cache(tmp_path):
    units = [np.full(4, .5)]
    path = tmp_path / 'stage.jsonl'
    path.write_text(json.dumps({'unit': units[0].tolist(), 'reward': 999.}) + '\n')

    class Pool:
        calls = 0
        def map(self, function, tasks):
            self.calls += 1
            return [[2.] for _ in tasks]

    pool = Pool()
    first = stage(pool, 'differentiator', 'nominal', units, 7, 1, path)
    assert pool.calls == 1 and first[0]['reward'] == 2.
    assert first[0]['noise_model'] == MODEL_VERSION
    assert stage(pool, 'differentiator', 'nominal', units, 7, 1, path) == first
    assert pool.calls == 1


def test_rerun_loader_rejects_unversioned_and_mixed_records(tmp_path):
    names = [p['id'] for p in manifest()['policies'][:2]]
    with np.load(DATA / 'evaluations.npz') as data:
        rows = [{key: data[f'{name}/{key}'] for key in FIELDS} for name in names]
    np.savez_compressed(tmp_path / (names[0]+'.npz'), **rows[0])
    with pytest.raises(ValueError, match='Unversioned'):
        load_groups(tmp_path)
    for name, row, version in zip(names, rows, ['old', MODEL_VERSION]):
        np.savez_compressed(tmp_path / (name+'.npz'), noise_model=np.array(version), **row)
    with pytest.raises(ValueError, match='different noise models'):
        load_groups(tmp_path)


def test_tuning_cache_requires_the_same_episodes(tmp_path):
    units = [np.full(4, .5)]
    path = tmp_path / 'stage.jsonl'

    class Pool:
        calls = 0
        def map(self, function, tasks):
            self.calls += 1
            return [[1.]*len(task[3]) for task in tasks]

    pool = Pool()
    stage(pool, 'differentiator', 'nominal', units, 11, 1, path)
    records = stage(pool, 'differentiator', 'nominal', units, 99, 8, path)
    assert pool.calls == 2
    assert records[0]['seed_start'] == 99 and len(records[0]['rewards']) == 8


def test_export_accepts_only_checkpoints_of_a_completed_run(tmp_path):
    from scripts.export_actors import completed
    from src.evaluation.fig4 import sha256
    checkpoint = tmp_path / 'best_model.zip'
    checkpoint.write_bytes(b'previous run')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'noise_model': MODEL_VERSION}))         # new run interrupted
    with pytest.raises(ValueError, match='did not complete'):
        completed(checkpoint)
    finished = {'noise_model': MODEL_VERSION, 'total_timesteps': 10, 'checkpoints_sha256': {'best_model.zip': sha256(checkpoint)}}
    config.write_text(json.dumps(finished))
    assert completed(checkpoint) == checkpoint
    checkpoint.write_bytes(b'another run')
    with pytest.raises(ValueError, match='does not belong'):
        completed(checkpoint)
    config.write_text(json.dumps({**finished, 'noise_model': 'old', 'checkpoints_sha256': {'best_model.zip': sha256(checkpoint)}}))
    with pytest.raises(ValueError, match='noise model'):
        completed(checkpoint)


def test_normalized_kalman_command_is_bounded_and_leaves_the_estimate_unchanged():
    from src.evaluation.benchmark_controllers import KalmanQuadratureController
    parameters = {"frequency_hz": 10500., "process_noise": .3, "phase": 1., "gain": .5, "depth": .8}
    plain = KalmanQuadratureController(**parameters)
    normalized = KalmanQuadratureController(**parameters, amplitude_floor=1e-4)
    rng = np.random.default_rng(0)
    for k in range(60):
        observation = np.zeros(28)
        observation[-3] = .3*np.cos(2*np.pi*10500*27.3e-6*k) + .05*rng.normal()
        plain.predict(observation)
        action, _ = normalized.predict(observation)
        depth = ((float(action[0]) + 1)/2)**2
        assert abs(depth - .8) <= .5 + 1e-6
    assert np.allclose(plain.x, normalized.x) and np.allclose(plain.P, normalized.P)
