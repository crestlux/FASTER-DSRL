"""Checkpoint, logging, data-batch and BC helpers for FASTER-DSRL.

Checkpoint files are trusted local pickle bundles. Frozen decoder source hashes,
optimizer states and data RNGs retain the original experiment contract.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['WANDB_MODE'] = 'online'

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from faster.agents.cached_dsrl_learner import CachedDSRLConfig, CachedDSRLLearner
from faster.data.action_chunk import ActionChunkSpec
from faster.data.dsrl_buffers import SOURCE_NAMES

QA_KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'masks', 'discounts', 'valids')
MAP_KEYS = ('observations', 'noise', 'actions')
ROOT = Path(__file__).resolve().parent.parent

@dataclass
class RunConfig:
    dataset: str = 'datasets/robomimic/can/mh/image_v141.hdf5'
    env_horizon: int = 500
    discount: float = .99
    seed: int = 42
    batch_size: int = 256
    bc_steps: int = 50000
    bc_eval_interval: int = 25000
    base_steps: int = 4000
    qa_calibration_steps: int = 2000
    qz_calibration_steps: int = 1000
    total_steps: int = 25000
    eval_interval: int = 5000
    eval_episodes: int = 10
    final_eval_episodes: int = 100
    checkpoint_interval: int = 5000
    qa_updates: int = 4
    qz_updates: int = 4
    actor_updates: int = 1
    training_mode: str = 'standard'
    offline_fraction: float = .5
    replay_capacity: int = 100000
    cache_capacity: int = 100000
    wandb_project: str = 'faster-cached-dsrl'
    learner: dict = field(default_factory=lambda: {'observation_dim': 23})

    def validate(self):
        if self.training_mode not in ('standard', 'hold_temperature', 'freeze_actor'):
            raise ValueError('Unknown explicit training experiment')
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError('batch_size must be positive and divisible by4 for source quotas')
        if not 0 <= self.offline_fraction < 1 or not 0 <= self.discount <= 1:
            raise ValueError('Invalid offline fraction or discount')
        if min(self.qa_updates, self.qz_updates, self.actor_updates) < 1:
            raise ValueError('Each decision must update QA/Qz/actor at least once')
        if min(self.bc_steps, self.base_steps, self.qa_calibration_steps, self.qz_calibration_steps,
               self.env_horizon, self.eval_episodes, self.eval_interval, self.checkpoint_interval,
               self.bc_eval_interval, self.replay_capacity) < 1:
            raise ValueError('Stage budgets/intervals/capacities must be positive')
        if self.cache_capacity < 4 or self.cache_capacity % 4:
            raise ValueError('cache_capacity must be divisible by4')
        return self


def learner_config(values):
    values = dict(values)
    for key in ('actor_dims', 'qa_dims', 'qz_dims', 'bc_cond_dims'):
        if key in values:
            values[key] = tuple(values[key])
    return CachedDSRLConfig(**values)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def load_bundle(path):
    # Pickle is intentionally restricted to our own local experiment artifacts.
    path = Path(path)
    blob = path.read_bytes()
    sidecar = path.with_suffix('.json')
    if sidecar.exists() and json.loads(sidecar.read_text())['sha256'] != hashlib.sha256(blob).hexdigest():
        raise ValueError('Checkpoint checksum mismatch')
    bundle = pickle.loads(blob)
    if bundle['version'] != 1:
        raise ValueError('Unsupported checkpoint version')
    cfg = learner_config(bundle['learner_config'])
    agent = CachedDSRLLearner.create(bundle['seed'], cfg).replace(decoder_frozen=bundle['decoder_frozen'])
    restored = serialization.from_bytes(agent, bundle['learner'])
    template_leaves = jax.tree_util.tree_leaves(agent)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if len(template_leaves) != len(restored_leaves) or any(
        np.shape(a) != np.shape(b) for a, b in zip(template_leaves, restored_leaves)):
        raise ValueError('Checkpoint tensor shapes disagree with its declared configuration')
    agent = restored
    if agent.decoder_frozen and agent.decoder_fingerprint() != bundle['decoder_fingerprint']:
        raise ValueError('Decoder fingerprint mismatch')
    bundle['_loaded_sha256'] = hashlib.sha256(blob).hexdigest()
    return agent, bundle


def save_bundle(path, agent, cfg, phase, counters, dataset, replay=None, cache=None, rollout_rng=None, extra=None):
    bundle = {'version': 1, 'phase': phase, 'seed': cfg.seed, 'run_config': asdict(cfg),
              'learner_config': asdict(agent.config), 'decoder_frozen': agent.decoder_frozen,
              'decoder_fingerprint': agent.decoder_fingerprint() if agent.decoder_frozen else None,
              'learner': serialization.to_bytes(agent), 'counters': dict(counters),
              'dataset_rng': dataset._rng.bit_generator.state,
              'replay': replay.state_dict() if replay is not None else None,
              'cache': cache.state_dict() if cache is not None else None,
              'rollout_rng': np.asarray(rollout_rng) if rollout_rng is not None else None,
              'simulator_resume': 'episode_restart_required' if phase == 'train' else 'new_episode',
              'extra': extra or {}}
    path = Path(path)
    temporary = path.with_suffix('.pkl.tmp')
    blob = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    with temporary.open('wb') as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    atomic_json(path.with_suffix('.json'), {'sha256': hashlib.sha256(blob).hexdigest(),
        'phase': phase, 'counters': dict(counters), 'decoder_fingerprint': bundle['decoder_fingerprint'],
        'simulator_resume': bundle['simulator_resume']})


def source_metadata(root=ROOT):
    """Hash public source/configuration bytes independently of checkout location."""
    root = Path(root)
    files = set(root.glob('*.py'))
    for directory in ('faster', 'scripts'):
        files.update((root / directory).rglob('*.py'))
    files.update((root / 'configs').rglob('*.json'))
    files.update(root / name for name in ('pyproject.toml', 'uv.lock'))
    paths = sorted(path.relative_to(root).as_posix() for path in files)
    digest = hashlib.sha256(b'faster-dsrl-public-source-v1\0')
    for relative in paths:
        name = relative.encode()
        content = (root / relative).read_bytes()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return {'source_sha256': digest.hexdigest(), 'source_files': paths,
            'source_sha256_scope': 'repository_relative_paths_and_bytes_v1'}


class Logger:
    def __init__(self, output, cfg, command, lineage=None):
        import wandb
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.tick = 0
        versions = {}
        from importlib.metadata import version
        for name in ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'robomimic', 'robosuite', 'wandb'):
            versions[name] = version(name)
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        git = git or os.environ.get('DSRL_SOURCE_GIT_HEAD', '')
        lineage_snapshot = lineage if isinstance(lineage, dict) else None
        lineage_path = lineage_snapshot['path'] if lineage_snapshot else lineage
        manifest = {'config': asdict(cfg), 'command': command, 'lineage': lineage_path,
                    'lineage_snapshot': lineage_snapshot, 'git_head': git,
                    **source_metadata(), 'versions': versions,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'jax_devices': [str(d) for d in jax.devices()], 'wandb_mode': 'online',
                    'run_group': os.environ.get('DSRL_RUN_GROUP'), 'job_name': os.environ.get('DSRL_RUN_NAME'),
                    'numeric_policy': 'fail on nonfinite including optimizer; no repair or clipping'}
        atomic_json(self.output / 'manifest.json', manifest)
        self.run = None
        try:
            self.run = wandb.init(project=cfg.wandb_project, name=os.environ.get('DSRL_RUN_NAME', self.output.name),
                                  group=os.environ.get('DSRL_RUN_GROUP'), dir=str(self.output),
                                  config=manifest, mode='online', save_code=False,
                                  settings=wandb.Settings(disable_git=True, disable_code=True, console='off'))
            if self.run is None or self.run.settings.mode != 'online':
                raise RuntimeError('W&B must be online')
            self.status('running')
        except BaseException:
            if self.run is not None:
                self.run.finish(exit_code=1)
            raise

    def status(self, state, **kwargs):
        atomic_json(self.output / 'status.json', {'state': state, 'elapsed_seconds': time.monotonic() - self.started, **kwargs})

    def log(self, metrics, **counters):
        values = {k: float(np.asarray(v)) for k, v in metrics.items()}
        values.update(counters)
        values['elapsed_seconds'] = time.monotonic() - self.started
        if not all(np.isfinite(v) for v in values.values()) or values.get('finite', 1) != 1:
            self.status('failed_nonfinite', counters=counters)
            raise FloatingPointError('Nonfinite loss/gradient/parameter/optimizer detected; no numeric repair applied')
        self.tick += 1
        with (self.output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(values) + '\n')
        self.run.log(values, step=self.tick)

    def close(self, state='complete', **kwargs):
        exit_code = 1
        try:
            self.status(state, **kwargs)
            exit_code = 0 if state == 'complete' else 1
        finally:
            self.run.finish(exit_code=exit_code)


def device_batch(batch, keys):
    return {key: jnp.asarray(batch[key], dtype=jnp.float32) for key in keys}


def mixed_batch(dataset, replay, cfg):
    off = int(cfg.batch_size * cfg.offline_fraction)
    online = replay.sample(cfg.batch_size - off)
    if off == 0:
        return device_batch(online, QA_KEYS)
    offline = dataset.sample(off)
    return {k: jnp.asarray(np.concatenate((offline[k], online[k])), dtype=jnp.float32) for k in QA_KEYS}


def add_mapping(cache, mapping, subtype='backup'):
    host = jax.device_get(mapping)
    if 'source' in host:
        for i, source in enumerate(SOURCE_NAMES[:2]):
            index = host['source'] == i
            cache.add_batch(*(host[k][index] for k in MAP_KEYS), source=source)
    else:
        cache.add_batch(*(host[k] for k in MAP_KEYS), source='selected', selected_subtype=subtype)


def make_env(cfg, seed):
    from faster.data.dsrl_runtime import RawRobomimicEnv
    return RawRobomimicEnv(cfg.dataset, horizon=cfg.env_horizon, seed=seed)


def run_bc(cfg, output):
    """Fresh BC phase with the original initialization, sampling and save order.

    The reported task policies share seed-42 BC weights after 50,000 updates.
    BC evaluation is separate and does not consume the training RNG streams.
    """
    from faster.data.dsrl_runtime import make_raw_dataset

    cfg.validate()
    desired = learner_config(cfg.learner)
    agent = CachedDSRLLearner.create(cfg.seed, desired)
    cfg.learner = asdict(agent.config)
    spec = ActionChunkSpec(desired.horizon, desired.execution_horizon, desired.action_dim)
    dataset, metadata = make_raw_dataset(cfg.dataset, spec, cfg.discount, seed=cfg.seed)
    log = Logger(output, cfg, 'bc', None)
    counters = {'primitive_steps': 0, 'decisions': 0, 'decoder_rows': 0, 'eval_steps': 0}
    rollout_rng = jax.random.PRNGKey(cfg.seed + 1)

    def save(name):
        save_bundle(log.output / name, agent, cfg, 'bc', counters, dataset,
                    rollout_rng=rollout_rng, extra={'dataset': metadata})

    try:
        atomic_json(log.output / 'dataset.json', metadata)
        while int(agent.bc.step) < cfg.bc_steps:
            batch = dataset.sample(cfg.batch_size)
            agent, metrics = agent.update_bc(device_batch(batch, ('observations', 'actions', 'action_valids')))
            log.log(metrics, **counters, bc_updates=int(agent.bc.step), qa_updates=int(agent.qa.step),
                    qz_updates=int(agent.qz.step), actor_updates=int(agent.actor.step))
            step = int(agent.bc.step)
            if step % cfg.bc_eval_interval == 0 or step == cfg.bc_steps:
                save(f'bc_{step}.pkl')
        save('latest.pkl')
    except BaseException:
        try:
            (log.output / 'error.txt').write_text(traceback.format_exc())
        finally:
            log.close('failed')
        raise
    else:
        log.close()
