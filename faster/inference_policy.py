"""Frozen, shape-specialized deployment for CachedDSRL and decoder-surrogate.

No optimizer, parameter update, output cache, precision change, or RNG change.
The time-conditioning cache contains only the frozen decoder's ten embeddings.
This module is intentionally independent of the experiment controller.
"""
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from faster.networks.diffusion import (
    DiffusionMLP, DiffusionMLPResNet, FourierFeatures, MLPResNetBlock,
    vp_beta_schedule,
)


class FrozenInferencePolicy:
    """Prepare once; call compiled(batch)(GPU observations, GPU RNG) repeatedly.

    mode: surrogate | native | learned_bon | gaussian_bon.
    variant: thin | time_cache | time_cache_unroll | affine_cache_unroll.
    trace=True additionally returns selected candidate index and scores for audit.
    RNG is split once per decision, with the existing fold_in(source,index) bank.
    """

    def __init__(self, agent, student=None, *, candidates=8, mode='surrogate', variant='time_cache'):
        if candidates not in (8,16,32):
            raise ValueError('Supported static candidate counts:8,16,32')
        if mode not in ('surrogate','native','learned_bon','gaussian_bon'):
            raise ValueError(mode)
        if variant not in ('thin','time_cache','time_cache_unroll','affine_cache_unroll'):
            raise ValueError(variant)
        if not agent.decoder_frozen:
            raise ValueError('Deployment requires a frozen decoder')
        if mode=='surrogate' and student is None:
            raise ValueError('Surrogate selector requires its trained student')
        self.config=agent.config
        self.candidates=candidates;self.mode=mode;self.variant=variant
        # Only live inference weights; never retain or pass TrainState/Adam/EMA.
        self.actor_params=jax.device_put(agent.actor.params)
        self.critic_params=jax.device_put(agent.qz.params if mode=='native' else agent.qa.params)
        self.decoder_params=jax.device_put(agent.decoder)
        self.student_params=jax.device_put(student.params) if mode=='surrogate' else None
        self.actor_apply=agent.actor.apply_fn;self.critic_apply=(agent.qz if mode=='native' else agent.qa).apply_fn
        self.decoder_apply=agent.bc.apply_fn
        self.student_apply=student.apply_fn if mode=='surrogate' else None
        self._compiled={};self._conditioning={}
        jax.block_until_ready((self.actor_params,self.critic_params,self.decoder_params,self.student_params))

    def _time_conditioning(self, rows):
        if rows in self._conditioning:return self._conditioning[rows]
        config=self.config;params=self.decoder_params
        fourier=FourierFeatures(output_size=128)
        encoder=DiffusionMLP(hidden_dims=config.bc_cond_dims,activations=nn.swish)
        @jax.jit
        def condition(params,t):
            time=jnp.full((rows,1),t)
            features=fourier.apply({'params':params['FourierFeatures_0']},time)
            return encoder.apply({'params':params['DiffusionMLP_0']},features)
        # Use exactly the original row shape during preparation, then retain the
        # one common embedding per t. No state/action-dependent value is cached.
        values=[condition(params,jnp.asarray(t,dtype=jnp.int32))[0] for t in range(config.denoise_steps)]
        cache=jnp.stack(values);jax.block_until_ready(cache)
        self._conditioning[rows]=cache
        return cache

    def _decode(self, params, observations, noise, conditioning):
        config=self.config;hats=jnp.cumprod(1-vp_beta_schedule(config.denoise_steps))
        reverse=DiffusionMLPResNet(num_blocks=config.bc_blocks,out_dim=config.latent_dim,
            hidden_dim=config.bc_width,use_layer_norm=True,activations=nn.swish)
        reverse_params=params['DiffusionMLPResNet_0']
        affine=self.variant=='affine_cache_unroll'
        if affine:
            first=reverse_params['Dense_0'];latent=config.latent_dim;obs_dim=config.observation_dim
            state_projection=observations@first['kernel'][latent:latent+obs_dim]
            time_projection=conditioning@first['kernel'][latent+obs_dim:]
        def epsilon(x,t):
            if self.variant=='thin':
                return self.decoder_apply({'params':params},observations,x,jnp.full((len(x),1),t))
            if not affine:
                cond=jnp.broadcast_to(conditioning[t],(len(x),conditioning.shape[-1]))
                return reverse.apply({'params':reverse_params},jnp.concatenate((x,observations,cond),-1))
            hidden=x@first['kernel'][:latent]+state_projection+time_projection[t]+first['bias']
            for i in range(config.bc_blocks):
                hidden=MLPResNetBlock(config.bc_width,act=nn.swish,use_layer_norm=True).apply(
                    {'params':reverse_params[f'MLPResNetBlock_{i}']},hidden)
            hidden=nn.swish(hidden)
            final=reverse_params['Dense_1']
            return hidden@final['kernel']+final['bias']
        def step(x,t):
            eps=epsilon(x,t)
            x0=(x-jnp.sqrt(1-hats[t])*eps)/jnp.sqrt(hats[t])
            previous=jnp.where(t>0,hats[jnp.maximum(t-1,0)],1.)
            return jnp.clip(jnp.sqrt(previous)*x0+jnp.sqrt(1-previous)*eps,-1.,1.),None
        unroll=config.denoise_steps if self.variant.endswith('unroll') else 1
        return jax.lax.scan(step,noise,jnp.arange(config.denoise_steps-1,-1,-1),unroll=unroll)[0]

    def compiled(self,batch,*,trace=False):
        key=(batch,trace)
        if key in self._compiled:return self._compiled[key]
        count=self.candidates;config=self.config;mode=self.mode
        decode_rows=batch*count if mode.endswith('bon') else batch
        conditioning=self._time_conditioning(decode_rows) if self.variant!='thin' else None
        params=(self.actor_params,self.critic_params,self.decoder_params,self.student_params,conditioning)
        flat,treedef=jax.tree_util.tree_flatten(params)
        def infer(observations,rng,*live_weights):
            actor_params,critic_params,decoder_params,student_params,condition=treedef.unflatten(live_weights)
            next_rng,draw_key=jax.random.split(rng)
            if mode=='gaussian_bon':
                keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.fold_in(draw_key,1),i))(jnp.arange(count))
                bank=jax.vmap(lambda k:jax.random.normal(k,(batch,config.latent_dim)))(keys).swapaxes(0,1)
            else:
                mean,log_std=self.actor_apply({'params':actor_params},observations)
                keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.fold_in(draw_key,0),i))(jnp.arange(count))
                eps=jax.vmap(lambda k:jax.random.normal(k,mean.shape))(keys).swapaxes(0,1)
                bank=config.noise_scale*jnp.tanh(mean[:,None]+jnp.exp(log_std[:,None])*eps)
            repeated=jnp.repeat(observations,count,axis=0);noise=bank.reshape(-1,config.latent_dim)
            if mode.endswith('bon'):
                decoded=self._decode(decoder_params,repeated,noise,condition)
                scores=self.critic_apply({'params':critic_params},repeated,decoded[:,:config.prefix_dim]).min(0)
            elif mode=='surrogate':
                predicted=self.student_apply({'params':student_params},repeated,noise)
                scores=self.critic_apply({'params':critic_params},repeated,predicted[:,:config.prefix_dim]).min(0)
            else:
                scores=self.critic_apply({'params':critic_params},repeated,noise).min(0)
            scores=scores.reshape(batch,count);winner=scores.argmax(-1)
            if mode.endswith('bon'):
                actions=decoded.reshape(batch,count,-1)[jnp.arange(batch),winner]
            else:
                selected=bank[jnp.arange(batch),winner]
                actions=self._decode(decoder_params,observations,selected,condition)
            return (next_rng,actions,winner,scores) if trace else (next_rng,actions)
        # Keep numerical weights as dynamic device arguments so XLA cannot fold
        # LayerNorm scales/biases into a different arithmetic expression. Flatten
        # the live weights once; no learner/optimizer tree walk on each call.
        compiled=jax.jit(infer).lower(jnp.zeros((batch,config.observation_dim),dtype=jnp.float32),
                                    jax.random.PRNGKey(0),*flat).compile()
        def call(observations,rng):return compiled(observations,rng,*flat)
        self._compiled[key]=call
        return call
