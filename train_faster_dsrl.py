#!/usr/bin/env python3
"""Train and evaluate the published FASTER-DSRL Action Predictor recipe.

Stages use trusted local checkpoints. Training starts from a calibrated shared
bundle; exact mid-episode simulator resumption is intentionally unsupported.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import traceback

from faster import training as tr
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from faster.agents.action_predictor import initialise_model, load_model, save_model, update_model
from faster.agents.cached_dsrl_learner import CachedDSRLLearner
from faster.data.action_chunk import ActionChunkSpec
from faster.data.dsrl_buffers import DecisionReplayBuffer, MappingCache
from faster.data.dsrl_runtime import execute_chunk, make_raw_dataset, read_env_metadata
from faster.faster_dsrl_policy import collect_set, update_qa_surrogate
from faster.data.evaluation_fixture import HASH_VERSION, portable_digest as digest
from faster.inference_policy import FrozenInferencePolicy

ROOT = Path(__file__).resolve().parent
STUDENT = dict(widths=[1024, 1024, 1024], learning_rate=3e-4,
               calibration_updates=50000, online_updates_per_decision=4)
CHUNK = ActionChunkSpec(4, 4, 7)
PREPARE_RNG_OFFSET = {'can': 1, 'square': 4}
MILESTONES = (50000, 100000, 250000, 500000)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def run_config(values):
    values = dict(values)
    # Historical Can bundles recorded this selector flag in the runner config.
    # Final surrogate selection is explicit in collect_set/update_qa_surrogate.
    if values.pop('selection_rule', 'early_qz') != 'early_qz':
        raise ValueError('Checkpoint is not from the canonical early-selection runner')
    known = {field.name for field in fields(tr.RunConfig)}
    if set(values) - known:
        raise ValueError(f'Unknown run configuration fields: {sorted(set(values) - known)}')
    return tr.RunConfig(**values).validate()


def check_data(expected, actual):
    # Relocating/recompressing an HDF5 file does not change its content digest.
    location_fields = {'source_path', 'source_size_bytes', 'source_mtime_ns'}
    def content(metadata):
        return {key: value for key, value in metadata.items() if key not in location_fields}
    if content(expected) != content(actual):
        raise ValueError('Dataset contents or preprocessing differ from the checkpoint')


def check_task_dataset(dataset, task):
    expected = {'can': 'PickPlaceCan', 'square': 'NutAssemblySquare'}[task]
    actual = read_env_metadata(dataset)['env_name']
    if actual != expected:
        raise ValueError(f'--task {task} requires HDF5 env_name={expected}, found {actual}')


def check_recipe(cfg, task):
    expected = dict(env_horizon=500 if task == 'can' else 400, discount=.99,
                    batch_size=256, bc_steps=50000, base_steps=4000,
                    qa_calibration_steps=2000, qz_calibration_steps=1000,
                    qa_updates=4, qz_updates=4, actor_updates=1,
                    offline_fraction=.5, replay_capacity=100000, cache_capacity=100000,
                    training_mode='standard')
    for name, value in expected.items():
        if getattr(cfg, name) != value:
            raise ValueError(f'Canonical {task} recipe requires {name}={value}')
    reference = json.loads((ROOT / 'configs' / f'faster_dsrl_{task}.json').read_text())
    if tr.learner_config(cfg.learner) != tr.learner_config(reference['learner']):
        raise ValueError('Learner architecture or proposal recipe differs from the published configuration')


def config_for(args, bundle=None):
    cfg = run_config(bundle['run_config'] if bundle else json.loads(
        (ROOT / 'configs' / f'faster_dsrl_{args.task}.json').read_text()))
    changes = {}
    for name in ('dataset', 'seed', 'total_steps'):
        value = getattr(args, name, None)
        if value is not None:
            changes[name] = value
    cfg = replace(cfg, **changes)
    cfg.learner = {**cfg.learner, 'learned': 8, 'gaussian': 0}
    check_recipe(cfg, args.task)
    return cfg


def finite(metrics):
    if not bool(np.asarray(metrics['finite'])):
        raise FloatingPointError('Nonfinite training update; no numerical repair applied')


def calibrate_predictor(agent, cache, cfg, log=None):
    student = initialise_model(cfg.seed + 6100500, 23, 28, tuple(STUDENT['widths']),
                               noise_dim=28, learning_rate=STUDENT['learning_rate'])
    cache_rng = deepcopy(cache.rng.bit_generator.state)
    try:
        for index in range(STUDENT['calibration_updates']):
            batch = tr.device_batch(cache.sample(cfg.batch_size), tr.MAP_KEYS)
            student, metrics = update_model(student, batch['observations'], batch['noise'], batch['actions'])
            finite(metrics)
            if log is not None and index % 100 == 0:
                log.log({'student/calibration_step': index + 1, 'student/mse': metrics['mse']})
    finally:
        cache.rng.bit_generator.state = cache_rng
    return student


def update_decision(agent, student, dataset, replay, cache, cfg, counters, before_steps):
    """Exact non-return surrogate branch, including data/cache RNG call order."""
    latest = {}
    for _ in range(4):
        batch = tr.mixed_batch(dataset, replay, cfg)
        agent, metrics, backup = update_qa_surrogate(agent, student, batch)
        finite(metrics)
        tr.add_mapping(cache, backup)
        counters['decoder_rows'] += cfg.batch_size
        latest.update(metrics)
    agent, fresh = agent.fresh_mapping(tr.mixed_batch(dataset, replay, cfg)['observations'])
    tr.add_mapping(cache, fresh)
    counters['decoder_rows'] += cfg.batch_size
    for index in range(4):
        batch = ({key: fresh[key] for key in tr.MAP_KEYS} if index == 0 else
                 tr.device_batch(cache.sample(cfg.batch_size), tr.MAP_KEYS))
        agent, metrics = agent.update_qz(batch)
        finite(metrics)
        latest.update(metrics)
        student, metrics = update_model(student, batch['observations'], batch['noise'], batch['actions'])
        finite(metrics)
        latest.update({f'student/{key}': value for key, value in metrics.items()})
    temperature = agent.temperature
    actor_observations = tr.mixed_batch(dataset, replay, cfg)['observations']
    agent, metrics = agent.update_actor(actor_observations)
    finite(metrics)
    if before_steps >= 50000:
        agent = agent.replace(temperature=temperature)
    latest.update(metrics)
    return agent, student, latest


def prepare(args, log, cfg, bc, bundle):
    if bundle['phase'] != 'bc' or bc.decoder_frozen or int(bc.bc.step) != 50000 or bundle['seed'] != 42:
        raise ValueError('Prepare requires the shared seed-42 BC checkpoint at update 50000')
    agent = CachedDSRLLearner.create(cfg.seed, tr.learner_config(cfg.learner))
    agent = agent.replace(bc=bc.bc, bc_ema=bc.bc_ema).freeze_decoder('ema')
    dataset, metadata = make_raw_dataset(cfg.dataset, CHUNK, .99, seed=cfg.seed)
    check_data(bundle['extra']['dataset'], metadata)
    replay = DecisionReplayBuffer(cfg.replay_capacity, seed=cfg.seed + 2)
    n = cfg.cache_capacity // 4
    cache = MappingCache(dict(zip(tr.SOURCE_NAMES, (n, n, 2*n))),
                         decoder_fingerprint=agent.decoder_fingerprint(), seed=cfg.seed + 3)
    rng = jax.random.PRNGKey(cfg.seed + PREPARE_RNG_OFFSET[args.task])
    counters = dict(primitive_steps=0, decisions=0, decoder_rows=0, eval_steps=0)
    env = tr.make_env(cfg, cfg.seed)
    try:
        obs = env.reset(seed=cfg.seed)
        while counters['primitive_steps'] + 4 <= cfg.base_steps:
            rng, key = jax.random.split(rng)
            mapping = agent.select(jnp.asarray(obs[None]), key, learned=0, gaussian=1)
            transition, obs, _ = execute_chunk(env, obs, np.asarray(mapping['noise'][0]),
                                                np.asarray(mapping['actions'][0]), CHUNK, .99)
            replay.insert(transition)
            tr.add_mapping(cache, mapping, 'base')
            counters['primitive_steps'] += int(transition['valid_steps'])
            counters['decisions'] += 1
            counters['decoder_rows'] += 1
            if transition['dones']:
                obs = env.reset()
    finally:
        env.close()
    for index in range(cfg.qa_calibration_steps):
        agent, metrics, mapping = agent.update_qa(tr.mixed_batch(dataset, replay, cfg), base_policy=True)
        finite(metrics)
        tr.add_mapping(cache, mapping)
        counters['decoder_rows'] += cfg.batch_size
        if index % 100 == 0:
            log.log({'prepare/qa_updates': index + 1, 'prepare/qa_loss': metrics['qa/loss']})
    for index in range(cfg.qz_calibration_steps):
        agent, fresh = agent.fresh_mapping(tr.mixed_batch(dataset, replay, cfg)['observations'])
        tr.add_mapping(cache, fresh)
        counters['decoder_rows'] += cfg.batch_size
        agent, metrics = agent.update_qz({key: fresh[key] for key in tr.MAP_KEYS})
        finite(metrics)
        if index % 100 == 0:
            log.log({'prepare/qz_updates': index + 1, 'prepare/qz_loss': metrics['qz/loss']})
    agent = agent.finish_calibration()
    tr.save_bundle(log.output / 'shared.pkl', agent, cfg, 'shared', counters, dataset, replay, cache, rng,
                   extra=dict(dataset=metadata, bc_sha256=bundle['_loaded_sha256'], fresh_RL=True,
                              task=args.task, prepare_rollout_rng_offset=PREPARE_RNG_OFFSET[args.task]))


def train(args, log, cfg, agent, bundle):
    if bundle['phase'] != 'shared' or not agent.decoder_frozen or int(agent.actor.step) != 0:
        raise ValueError('Train requires a fresh calibrated shared checkpoint; simulator resume is unsupported')
    if cfg.seed != bundle['seed']:
        raise ValueError('Training must preserve the prepared RL seed')
    if agent.config != tr.learner_config(cfg.learner):
        raise ValueError('Shared checkpoint must use the canonical 8-learned-candidate configuration')
    agent = agent.replace(actor=agent.actor.replace(tx=optax.adam(3e-4)))
    dataset, metadata = make_raw_dataset(cfg.dataset, CHUNK, .99, seed=cfg.seed)
    check_data(bundle['extra']['dataset'], metadata)
    dataset._rng.bit_generator.state = bundle['dataset_rng']
    decoder = agent.decoder_fingerprint()
    replay = DecisionReplayBuffer.from_state_dict(bundle['replay'])
    cache = MappingCache.from_state_dict(bundle['cache'], decoder_fingerprint=decoder)
    counters = dict(bundle['counters'])
    initial = dict(counters)
    optimizer_start = {key: int(getattr(agent, key).step) for key in ('actor', 'qa', 'qz', 'temperature')}
    rollout_rng = jnp.asarray(bundle['rollout_rng'])
    student = calibrate_predictor(agent, cache, cfg, log)
    actor_calls = alpha_calls = 0
    targets = iter(target for target in MILESTONES if counters['primitive_steps'] < target < cfg.total_steps)
    next_target = next(targets, None)

    def save(target, final=False):
        if agent.decoder_fingerprint() != decoder:
            raise RuntimeError('Frozen decoder changed')
        delta = counters['decisions'] - initial['decisions']
        expected = dict(qa=4*delta, qz=4*delta, actor=actor_calls, temperature=alpha_calls)
        for key, count in expected.items():
            if int(getattr(agent, key).step) - optimizer_start[key] != count:
                raise RuntimeError(f'Optimizer update-count mismatch: {key}')
        checkpoint = log.output / ('latest.pkl' if final else f'step_{target}.pkl')
        predictor = log.output / f'student_{target}.pkl'
        tr.save_bundle(checkpoint, agent, cfg, 'train', counters, dataset, replay, cache, rollout_rng,
                       extra=dict(dataset=metadata, task=args.task, nominal_step=target,
                                  student_contract=STUDENT, actor_value_path='qz',
                                  actor_optimizer=dict(learning_rate=3e-4, entropy_scale=1.),
                                  resume_policy='No exact simulator continuation'))
        save_model(predictor, student)
        tr.atomic_json(log.output / f'pair_{target}.json', dict(
            task=args.task, checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint),
            student_checkpoint=predictor.name, student_checkpoint_sha256=sha(predictor),
            decoder_fingerprint=decoder, actual_steps=counters['primitive_steps'], nominal_step=target))

    env = tr.make_env(cfg, cfg.seed + 500)
    latest = {}
    try:
        obs = env.reset(seed=cfg.seed + 500 + counters['primitive_steps'])
        while counters['primitive_steps'] + 4 <= cfg.total_steps:
            before_steps = counters['primitive_steps']
            rollout_rng, key = jax.random.split(rollout_rng)
            mapping, _ = collect_set(agent, student, jnp.asarray(obs[None]), key, True)
            transition, obs, info = execute_chunk(env, obs, np.asarray(mapping['noise'][0]),
                                                   np.asarray(mapping['actions'][0]), CHUNK, .99)
            replay.insert(transition)
            tr.add_mapping(cache, mapping, 'rollout')
            counters['primitive_steps'] += int(transition['valid_steps'])
            counters['decisions'] += 1
            counters['decoder_rows'] += 1
            agent, student, latest = update_decision(agent, student, dataset, replay, cache, cfg, counters, before_steps)
            actor_calls += 1
            alpha_calls += int(before_steps < 50000)
            if transition['dones']:
                obs = env.reset()
            if counters['decisions'] % 25 == 0 or transition['dones']:
                log.log({**latest, 'rollout/success': float(info['success'])}, **counters)
            while next_target is not None and counters['primitive_steps'] >= next_target:
                save(next_target)
                next_target = next(targets, None)
    finally:
        env.close()
    save(cfg.total_steps, final=True)
    log.log({**latest, 'finite': 1}, **counters)


def load_pair(path, task):
    path = Path(path)
    pair = json.loads(path.read_text())
    if pair['task'] != task:
        raise ValueError('Checkpoint task does not match --task')
    paths = [path.parent / pair[key] for key in ('checkpoint', 'student_checkpoint')]
    for file, key in zip(paths, ('checkpoint_sha256', 'student_checkpoint_sha256')):
        if sha(file) != pair[key]:
            raise ValueError(f'Paired checkpoint checksum mismatch: {file.name}')
    agent, bundle = tr.load_bundle(paths[0])
    student, _ = load_model(paths[1])
    expected = dict(observation_dim=23, action_dim=28, noise_dim=28,
                    widths=tuple(STUDENT['widths']), learning_rate=STUDENT['learning_rate'], bounded_output=True)
    if asdict(student.config) != expected:
        raise ValueError('Paired Action Predictor differs from the canonical model configuration')
    if agent.decoder_fingerprint() != pair['decoder_fingerprint']:
        raise ValueError('Paired decoder fingerprint mismatch')
    return agent, student, bundle, pair


def evaluate(args, log, cfg, agent, student, pair):
    fixture = json.loads((ROOT / 'configs' / f'evaluation_{args.task}.json').read_text())
    if fixture.get('hash_version') != HASH_VERSION:
        raise ValueError(f'Evaluation fixture requires hash_version={HASH_VERSION}')
    exact_panel = args.seed_start is None
    start = fixture['start'] if exact_panel else args.seed_start
    if args.episodes < 1 or (exact_panel and args.episodes > len(fixture['hashes'])):
        raise ValueError('Invalid episode count for the selected evaluation panel')
    original = (serialization.to_bytes(agent), serialization.to_bytes(student))
    infer = (FrozenInferencePolicy(agent, student, candidates=8, mode='surrogate',
                                   variant='time_cache').compiled(1) if args.task == 'can' else None)
    action_error = 0.
    parity_checks = 0
    env = tr.make_env(cfg, start)
    rows = []
    try:
        for index in range(args.episodes):
            seed = start + index
            obs = env.reset(seed=seed)
            reset_hash = digest(env.get_state())
            if exact_panel and reset_hash != fixture['hashes'][index]:
                raise ValueError(f'Historical simulator reset fixture mismatch at seed {seed}')
            rng = jax.random.PRNGKey(seed)
            rewards = []
            success = False
            while True:
                observations = jnp.asarray(obs[None])
                if infer is None:
                    rng, key = jax.random.split(rng)
                    mapping, _ = collect_set(agent, student, observations, key, True)
                    actions = mapping['actions']
                else:
                    previous_rng = rng
                    rng, actions = infer(observations, rng)
                    if index == 0:
                        expected_rng, key = jax.random.split(previous_rng)
                        mapping, _ = collect_set(agent, student, observations, key, True)
                        action_error = max(action_error, float(np.max(np.abs(
                            np.asarray(actions) - np.asarray(mapping['actions'])))))
                        parity_checks += 1
                        if not np.array_equal(rng, expected_rng) or action_error > 1e-5:
                            raise RuntimeError('Time-cached Can inference failed original-policy parity')
                for action in np.asarray(actions[0]).reshape(4, 7):
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    success |= bool(info['success'])
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            rows.append(dict(seed=seed, success=success, primitive_steps=len(rewards),
                             discounted_return=sum(.99**i * r for i, r in enumerate(rewards)),
                             initial_snapshot_sha256=reset_hash))
            tr.atomic_json(log.output / 'progress.json', dict(completed_episodes=len(rows), total_episodes=args.episodes))
            if len(rows) % 5 == 0:
                log.log({'eval/completed': len(rows), 'eval/success_count': sum(row['success'] for row in rows)})
    finally:
        env.close()
    if original != (serialization.to_bytes(agent), serialization.to_bytes(student)):
        raise RuntimeError('Evaluation changed model state')
    tr.atomic_json(log.output / 'evaluation.json', dict(
        episodes=rows, success_rate=sum(row['success'] for row in rows) / len(rows),
        success_count=sum(row['success'] for row in rows), checkpoint=pair,
        historical_reset_hashes_verified=exact_panel, model_unchanged=True,
        decoder_calls_per_decision=1, seed_start=start,
        inference_variant='time_cache' if infer is not None else 'original',
        original_action_parity=dict(checked_decisions=parity_checks, max_action_error=action_error)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bc', 'prepare', 'train', 'evaluate'))
    parser.add_argument('--task', choices=('can', 'square'), required=True)
    parser.add_argument('--dataset', help='HDF5 containing the recorded low-dimensional observations')
    parser.add_argument('--checkpoint', help='BC/shared pickle, or pair_N.json for evaluate')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, help='Prepare only; BC also accepts its fixed shared seed 42')
    parser.add_argument('--total-steps', type=int, default=None, help='Train only: total primitive budget including 4000 base steps')
    parser.add_argument('--episodes', type=int, help='Evaluate only: episode count (default: 200)')
    parser.add_argument('--seed-start', type=int, help='Custom evaluation panel; disables historical reset matching')
    args = parser.parse_args(argv)
    allowed = {'seed': ('bc', 'prepare'), 'total_steps': ('train',),
               'episodes': ('evaluate',), 'seed_start': ('evaluate',)}
    for name, commands in allowed.items():
        if getattr(args, name) is not None and args.command not in commands:
            parser.error(f'--{name.replace("_", "-")} is only supported for {", ".join(commands)}')
    if args.episodes is None:
        args.episodes = 200
    if args.command == 'evaluate':
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        if args.seed_start is not None and not 0 <= args.seed_start <= 2**32 - args.episodes:
            parser.error('The evaluation seed range must fit unsigned 32-bit simulator seeds')
    # BC and the preserved Can preparation used JAX's environment/default
    # precision. Square preparation and final RL/evaluation explicitly used highest.
    if args.command in ('train', 'evaluate') or (args.command == 'prepare' and args.task == 'square'):
        jax.config.update('jax_default_matmul_precision', 'highest')
    if args.command == 'bc':
        if args.seed not in (None, 42):
            parser.error('The published experiments share one seed-42 BC policy per task')
        cfg = config_for(args)
        check_task_dataset(cfg.dataset, args.task)
        cfg.learner = {**cfg.learner, 'learned': 4, 'gaussian': 4}
        tr.run_bc(cfg, args.output)
        return
    if not args.checkpoint:
        parser.error('--checkpoint is required for this phase')
    if args.command == 'evaluate':
        agent, student, bundle, pair = load_pair(args.checkpoint, args.task)
    else:
        agent, bundle = tr.load_bundle(args.checkpoint)
    cfg = config_for(args, bundle)
    if args.command == 'prepare':
        if bundle['phase'] != 'bc' or agent.decoder_frozen or int(agent.bc.step) != 50000 or bundle['seed'] != 42:
            parser.error('prepare requires the shared seed-42 BC checkpoint at update 50000')
        cfg = replace(cfg, seed=args.seed if args.seed is not None else 42, total_steps=500000)
    elif args.command == 'train':
        if bundle['phase'] != 'shared' or not agent.decoder_frozen or int(agent.actor.step) != 0:
            parser.error('train requires a fresh calibrated shared checkpoint; exact simulator resume is unsupported')
        if cfg.seed != bundle['seed'] or agent.config != tr.learner_config(cfg.learner):
            parser.error('train must preserve the prepared RL seed and the canonical learner configuration')
        cfg = replace(cfg, total_steps=args.total_steps if args.total_steps is not None else 500000)
        if cfg.total_steps <= bundle['counters']['primitive_steps']:
            parser.error('--total-steps must exceed the prepared primitive count')
    elif agent.config != tr.learner_config(cfg.learner):
        parser.error('evaluate requires the canonical learner configuration')
    check_task_dataset(cfg.dataset, args.task)
    log = tr.Logger(args.output, cfg, f'faster_dsrl_{args.command}', str(Path(args.checkpoint).resolve()))
    try:
        recipe = dict(task=args.task, student=STUDENT, actor_value_path='qz', alpha_hold_after=50000,
            prepare_rng_offset=PREPARE_RNG_OFFSET[args.task], runner_sha256=sha(__file__),
            jax_default_matmul_precision=jax.config.jax_default_matmul_precision)
        if args.command == 'evaluate':
            fixture_path = ROOT / 'configs' / f'evaluation_{args.task}.json'
            fixture = json.loads(fixture_path.read_text())
            if fixture.get('hash_version') != HASH_VERSION:
                raise ValueError(f'Evaluation fixture requires hash_version={HASH_VERSION}')
            recipe['evaluation'] = dict(panel='historical' if args.seed_start is None else 'custom',
                seed_start=fixture['start'] if args.seed_start is None else args.seed_start,
                requested_episodes=args.episodes, horizon=cfg.env_horizon,
                historical_reset_matching=args.seed_start is None,
                fixture=fixture_path.relative_to(ROOT).as_posix(), fixture_sha256=sha(fixture_path),
                hash_version=fixture['hash_version'])
        tr.atomic_json(log.output / 'recipe.json', recipe)
        if args.command == 'prepare':
            prepare(args, log, cfg, agent, bundle)
        elif args.command == 'train':
            train(args, log, cfg, agent, bundle)
        else:
            evaluate(args, log, cfg, agent, student, pair)
    except BaseException:
        try:
            (log.output / 'error.txt').write_text(traceback.format_exc())
        finally:
            log.close('failed')
        raise
    else:
        log.close()


if __name__ == '__main__':
    main()
