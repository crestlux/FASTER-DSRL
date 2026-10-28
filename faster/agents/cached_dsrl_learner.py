"""Independent cached DSRL-NA + FASTER early selection, with hard QA backup.

Entropy regularizes the proposal, not the selected mixture. No residual, alpha
projection, gradient clipping, or silent nonfinite repair. Reference differences
and solver semantics are recorded in docs/dsrl_reference_audit.md.
"""
from dataclasses import dataclass
from functools import partial
import hashlib
import inspect
import json
import math
from numbers import Integral
from pathlib import Path

import flax.linen as nn
from flax import serialization, struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from faster.networks.diffusion import DDPM, DiffusionMLP, DiffusionMLPResNet, FourierFeatures, vp_beta_schedule


@dataclass(frozen=True)
class CachedDSRLConfig:
    observation_dim: int
    action_dim: int = 7
    horizon: int = 4
    execution_horizon: int = 4
    denoise_steps: int = 10
    actor_dims: tuple = (256, 256, 256)
    qa_dims: tuple = (1024, 1024, 1024)
    qz_dims: tuple = (512, 512, 512)
    bc_width: int = 256
    bc_blocks: int = 3
    bc_cond_dims: tuple = (256, 256)
    bc_lr: float = 3e-4
    rl_lr: float = 3e-4
    bc_ema_beta: float = .999
    qa_target_beta: float = .995
    qz_ema_beta: float = .995
    actor_ema_beta: float = .98
    noise_scale: float = 1.5
    initial_alpha: float = 1.0
    target_entropy: float = 0.0
    learned: int = 4
    gaussian: int = 4
    decoder_solver: str = 'faster_ddim'

    def __post_init__(self):
        for name in ('observation_dim', 'action_dim', 'horizon', 'execution_horizon',
                     'denoise_steps', 'bc_width', 'bc_blocks'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('actor_dims', 'qa_dims', 'qz_dims', 'bc_cond_dims'):
            widths = getattr(self, name)
            if not widths or any(isinstance(width, bool) or not isinstance(width, Integral) or width < 1
                                 for width in widths):
                raise ValueError(f'{name} must contain positive integer widths')
        if not 1 <= self.execution_horizon <= self.horizon:
            raise ValueError('Require 1 <= C <= H')
        for name in ('learned', 'gaussian'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.learned < 0 or self.gaussian < 0 or self.learned + self.gaussian < 1:
            raise ValueError('Require a nonempty proposal bank')
        for name in ('noise_scale', 'initial_alpha', 'bc_lr', 'rl_lr'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.target_entropy):
            raise ValueError('target_entropy must be finite')
        if any(not 0 <= x < 1 for x in (self.bc_ema_beta, self.qa_target_beta, self.qz_ema_beta, self.actor_ema_beta)):
            raise ValueError('EMA beta must be in [0, 1)')
        if self.decoder_solver != 'faster_ddim':
            raise ValueError('Only explicit FASTER xt-clipped deterministic DDIM is supported')

    @property
    def latent_dim(self):
        return self.horizon * self.action_dim

    @property
    def prefix_dim(self):
        return self.execution_horizon * self.action_dim


class HiddenMLP(nn.Module):
    dims: tuple

    @nn.compact
    def __call__(self, x):
        # Official SB3-NA topology; widths are a separate engineering choice.
        for width in self.dims:
            x = nn.tanh(nn.LayerNorm()(nn.Dense(width)(x)))
        return x


class NoiseActor(nn.Module):
    dims: tuple
    action_dim: int

    @nn.compact
    def __call__(self, observations):
        x = HiddenMLP(self.dims)(observations)
        return nn.Dense(self.action_dim, name='mean')(x), jnp.clip(
            nn.Dense(self.action_dim, name='log_std')(x), -20., 2.)


class TwinQ(nn.Module):
    dims: tuple

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate((observations, actions), -1)
        return jnp.stack([nn.Dense(1, name=f'out_{i}')(
            HiddenMLP(self.dims, name=f'q_{i}')(x))[..., 0] for i in range(2)])


def sample_tanh_normal(mean, log_std, epsilon):
    """Paired sample/log-density in normalized u; never invert a rounded tanh."""
    pre = mean + jnp.exp(log_std) * epsilon
    normal = -.5 * jnp.sum(epsilon ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    jacobian = jnp.sum(2 * (jnp.log(2.) - pre - jax.nn.softplus(-2 * pre)), axis=-1)
    return jnp.tanh(pre), normal - jacobian


def ema(old, new, beta):
    return jax.tree_util.tree_map(lambda a, b: beta * a + (1 - beta) * b, old, new)


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))


@struct.dataclass
class CachedDSRLLearner:
    config: CachedDSRLConfig = struct.field(pytree_node=False)
    bc: TrainState
    bc_ema: object
    decoder: object
    actor: TrainState
    actor_ema: object
    qa: TrainState
    qa_target: object
    qz: TrainState
    qz_ema: object
    temperature: TrainState
    rng: object
    decoder_frozen: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(cls, seed, config):
        keys = jax.random.split(jax.random.PRNGKey(seed), 6)
        obs = jnp.zeros((1, config.observation_dim))
        actions = jnp.zeros((1, config.latent_dim))
        model = DDPM(
            cond_encoder_cls=partial(DiffusionMLP, hidden_dims=config.bc_cond_dims, activations=nn.swish),
            reverse_encoder_cls=partial(DiffusionMLPResNet, num_blocks=config.bc_blocks,
                out_dim=config.latent_dim, hidden_dim=config.bc_width, use_layer_norm=True, activations=nn.swish),
            time_preprocess_cls=partial(FourierFeatures, output_size=128))
        bc = TrainState.create(apply_fn=model.apply,
            params=model.init(keys[0], obs, actions, jnp.zeros((1, 1)))['params'],
            tx=optax.adamw(optax.cosine_decay_schedule(config.bc_lr, 3_000_000)))
        actor_model = NoiseActor(config.actor_dims, config.latent_dim)
        actor = TrainState.create(apply_fn=actor_model.apply,
            params=actor_model.init(keys[1], obs)['params'], tx=optax.adam(config.rl_lr))
        def critic(dims, key, size):
            net = TwinQ(dims)
            return TrainState.create(apply_fn=net.apply,
                params=net.init(key, obs, jnp.zeros((1, size)))['params'], tx=optax.adam(config.rl_lr))
        qa = critic(config.qa_dims, keys[2], config.prefix_dim)
        qz = critic(config.qz_dims, keys[3], config.latent_dim)
        temperature = TrainState.create(apply_fn=jnp.exp,
            params={'log_alpha': jnp.log(config.initial_alpha)}, tx=optax.adam(config.rl_lr))
        return cls(config, bc, bc.params, bc.params, actor, actor.params, qa, qa.params,
                   qz, qz.params, temperature, keys[5])

    @jax.jit
    def update_bc(self, batch):
        if self.decoder_frozen:
            raise ValueError('Cannot update the frozen decoder')
        rng, t_key, e_key = jax.random.split(self.rng, 3)
        actions = batch['actions']
        ts = jax.random.randint(t_key, (len(actions),), 0, self.config.denoise_steps)
        epsilon = jax.random.normal(e_key, actions.shape)
        hats = jnp.cumprod(1 - vp_beta_schedule(self.config.denoise_steps))[ts, None]
        noisy = jnp.sqrt(hats) * actions + jnp.sqrt(1 - hats) * epsilon
        mask = batch.get('action_valids', batch.get('action_masks', jnp.ones_like(actions)))
        if mask.shape[-1] == self.config.horizon:
            mask = jnp.repeat(mask, self.config.action_dim, axis=-1)
        if mask.shape != actions.shape:
            raise ValueError('BC action_valids must have shape [B,H] or [B,H*d]')
        def loss(params):
            predicted = self.bc.apply_fn({'params': params}, batch['observations'], noisy, ts[:, None])
            return jnp.sum((predicted - epsilon) ** 2 * mask) / jnp.maximum(mask.sum(), 1)
        value, grads = jax.value_and_grad(loss)(self.bc.params)
        has_valid = mask.sum() > 0
        bc = jax.lax.cond(has_valid, lambda _: self.bc.apply_gradients(grads=grads), lambda _: self.bc, None)
        bc_ema = jax.lax.cond(has_valid,
            lambda _: ema(self.bc_ema, bc.params, self.config.bc_ema_beta), lambda _: self.bc_ema, None)
        return self.replace(bc=bc, bc_ema=bc_ema, rng=rng), {
            'bc/loss': value, 'bc/grad_norm': optax.global_norm(grads),
            'bc/updated': has_valid,
            'finite': all_finite((value, grads, bc.params, bc.opt_state))}

    def freeze_decoder(self, source='ema'):
        if source not in ('ema', 'raw'):
            raise ValueError('Decoder source must be ema or raw')
        if self.decoder_frozen:
            raise ValueError('Decoder already frozen; restore without refreezing')
        return self.replace(decoder=self.bc_ema if source == 'ema' else self.bc.params, decoder_frozen=True)

    def decoder_fingerprint(self):
        contract = {k: getattr(self.config, k) for k in (
            'observation_dim', 'action_dim', 'horizon', 'execution_horizon', 'denoise_steps', 'decoder_solver')}
        contract['mapping_contract_version'] = 1
        contract['preprocessing'] = 'raw_eef_pos_eef_quat_gripper_object_v1_identity_actions'
        # Whole source modules conservatively cover decoder construction,
        # schedule, network blocks, and sampler implementation. Even unrelated
        # edits may invalidate a cache; changed mappings must never reuse it.
        contract['implementation_sha256'] = {
            name: hashlib.sha256(Path(inspect.getsourcefile(owner)).read_bytes()).hexdigest()
            for name, owner in (('cached_dsrl_learner', type(self)), ('diffusion', DDPM))
        }
        # JIT tree reconstruction and checkpoint restoration may change dict
        # insertion order without changing the decoder. Hash canonical state so
        # an exact restored mapping remains eligible for cache reuse.
        def canonical(value):
            if isinstance(value, dict):
                return {key: canonical(value[key]) for key in sorted(value)}
            return value
        state = canonical(serialization.to_state_dict(self.decoder))
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode() +
                              serialization.msgpack_serialize(state)).hexdigest()

    @jax.jit
    def decode(self, observations, noise):
        """Preserve FASTER clipping of every x_next, including the final action."""
        hats = jnp.cumprod(1 - vp_beta_schedule(self.config.denoise_steps))
        def step(x, t):
            eps = self.bc.apply_fn({'params': self.decoder}, observations, x, jnp.full((len(x), 1), t))
            x0 = (x - jnp.sqrt(1 - hats[t]) * eps) / jnp.sqrt(hats[t])
            previous = jnp.where(t > 0, hats[jnp.maximum(t - 1, 0)], 1.)
            return jnp.clip(jnp.sqrt(previous) * x0 + jnp.sqrt(1 - previous) * eps, -1., 1.), None
        return jax.lax.scan(step, noise, jnp.arange(self.config.denoise_steps - 1, -1, -1))[0]

    @partial(jax.jit, static_argnames=('learned', 'gaussian', 'actor_shadow'))
    def candidates(self, observations, key, learned=8, gaussian=8, actor_shadow=False):
        mean, std = self.actor.apply_fn({'params': self.actor_ema if actor_shadow else self.actor.params}, observations)
        def bank(source, count):
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.fold_in(key, source), i))(jnp.arange(count))
            return jax.vmap(lambda k: jax.random.normal(k, mean.shape))(keys).swapaxes(0, 1)
        u, _ = sample_tanh_normal(mean[:, None], std[:, None], bank(0, learned))
        return jnp.concatenate((self.config.noise_scale * u, bank(1, gaussian)), 1)

    @partial(jax.jit, static_argnames=('learned', 'gaussian', 'actor_shadow', 'qz_shadow', 'late'))
    def select(self, observations, key, learned=None, gaussian=None, actor_shadow=False, qz_shadow=False, late=False):
        learned = self.config.learned if learned is None else learned
        gaussian = self.config.gaussian if gaussian is None else gaussian
        if min(learned, gaussian) < 0 or learned + gaussian < 1:
            raise ValueError('Require nonempty nonnegative candidate counts')
        bank = self.candidates(observations, key, learned, gaussian, actor_shadow)
        count = learned + gaussian
        repeated = jnp.repeat(observations, count, axis=0)
        noises = bank.reshape(-1, self.config.latent_dim)
        if late:
            decoded = self.decode(repeated, noises)
            scores = self.qa.apply_fn({'params': self.qa.params}, repeated, decoded[:, :self.config.prefix_dim]).min(0)
        else:
            scores = self.qz.apply_fn({'params': self.qz_ema if qz_shadow else self.qz.params}, repeated, noises).min(0)
        winner = scores.reshape(len(observations), count).argmax(-1)
        selected = bank[jnp.arange(len(observations)), winner]
        actions = decoded.reshape(len(observations), count, -1)[jnp.arange(len(observations)), winner] if late else self.decode(observations, selected)
        return {'observations': observations, 'noise': selected, 'actions': actions,
                'selected_learned': winner < learned, 'selected_index': winner,
                'selected_q': scores.reshape(len(observations), count).max(-1)}

    @partial(jax.jit, static_argnames=('base_policy',))
    def update_qa(self, batch, base_policy=False):
        if not self.decoder_frozen:
            raise ValueError('Freeze BC before RL')
        rng, key = jax.random.split(self.rng)
        selected = self.select(batch['next_observations'], key,
            learned=0 if base_policy else self.config.learned, gaussian=1 if base_policy else self.config.gaussian)
        target = batch['rewards'] + batch['discounts'] * batch['masks'] * self.qa.apply_fn(
            {'params': self.qa_target}, batch['next_observations'], selected['actions'][:, :self.config.prefix_dim]).min(0)
        target = jax.lax.stop_gradient(target)
        valid = batch['valids']
        def loss(params):
            q = self.qa.apply_fn({'params': params}, batch['observations'], batch['actions'][:, :self.config.prefix_dim])
            return jnp.sum((q - target[None]) ** 2 * valid[None]) / jnp.maximum(2 * valid.sum(), 1), q
        (value, q), grads = jax.value_and_grad(loss, has_aux=True)(self.qa.params)
        qa = jax.lax.cond(valid.sum() > 0, lambda _: self.qa.apply_gradients(grads=grads), lambda _: self.qa, None)
        target_params = jax.lax.cond(valid.sum() > 0,
            lambda _: ema(self.qa_target, qa.params, self.config.qa_target_beta), lambda _: self.qa_target, None)
        return self.replace(qa=qa, qa_target=target_params, rng=rng), {
            'qa/loss': value, 'qa/grad_norm': optax.global_norm(grads), 'qa/q_mean': q.mean(),
            'qa/target_mean': target.mean(), 'qa/valid_fraction': valid.mean(), 'qa/updated': valid.sum() > 0,
            'finite': all_finite((value, grads, qa.params, qa.opt_state, target_params, target))}, selected

    @jax.jit
    def update_qz(self, mapping):
        teacher = jax.lax.stop_gradient(self.qa.apply_fn({'params': self.qa.params},
            mapping['observations'], mapping['actions'][:, :self.config.prefix_dim]))
        def loss(params):
            q = self.qz.apply_fn({'params': params}, mapping['observations'], mapping['noise'])
            return jnp.mean((q - teacher) ** 2), q
        (value, q), grads = jax.value_and_grad(loss, has_aux=True)(self.qz.params)
        qz = self.qz.apply_gradients(grads=grads)
        return self.replace(qz=qz, qz_ema=ema(self.qz_ema, qz.params, self.config.qz_ema_beta)), {
            'qz/loss': value, 'qz/grad_norm': optax.global_norm(grads), 'qz/teacher_mean': teacher.mean(),
            'qz/disagreement': jnp.abs(q[0] - q[1]).mean(),
            'finite': all_finite((value, grads, qz.params, qz.opt_state))}

    @jax.jit
    def update_actor(self, observations):
        rng, key = jax.random.split(self.rng)
        eps = jax.random.normal(key, (len(observations), self.config.latent_dim))
        alpha = jnp.exp(self.temperature.params['log_alpha'])
        def loss(params):
            mean, std = self.actor.apply_fn({'params': params}, observations)
            u, logp = sample_tanh_normal(mean, std, eps)
            q = self.qz.apply_fn({'params': self.qz.params}, observations, self.config.noise_scale * u).min(0)
            return jnp.mean(jax.lax.stop_gradient(alpha) * logp - q), (logp, u, std)
        (value, (logp, u, std)), grads = jax.value_and_grad(loss, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        entropy = -logp.mean()
        def alpha_loss(params):
            return params['log_alpha'] * jax.lax.stop_gradient(entropy - self.config.target_entropy)
        temp_loss, temp_grads = jax.value_and_grad(alpha_loss)(self.temperature.params)
        temperature = self.temperature.apply_gradients(grads=temp_grads)
        return self.replace(actor=actor, actor_ema=ema(self.actor_ema, actor.params, self.config.actor_ema_beta),
                            temperature=temperature, rng=rng), {
            'actor/loss': value, 'actor/entropy_u': entropy, 'actor/alpha': alpha,
            'actor/grad_norm': optax.global_norm(grads), 'actor/saturation_fraction': (jnp.abs(u) > .99).mean(),
            'actor/log_std_mean': std.mean(), 'alpha/loss': temp_loss,
            'finite': all_finite((value, grads, actor.params, actor.opt_state,
                                   temperature.params, temperature.opt_state, alpha))}

    @jax.jit
    def fresh_mapping(self, observations):
        rng, key = jax.random.split(self.rng)
        bank = self.candidates(observations, key, learned=1, gaussian=1)
        ids = jnp.arange(len(observations)) % 2
        noise = bank[jnp.arange(len(observations)), ids]
        return self.replace(rng=rng), {'observations': observations, 'noise': noise,
                                      'actions': self.decode(observations, noise), 'source': ids}

    def finish_calibration(self):
        return self.replace(qz_ema=self.qz.params, actor_ema=self.actor.params)
