"""FASTER-DSRL action-predictor selection and hard action-critic updates.

The learned proposal supplies eight noise candidates. The current action critic
scores predicted chunks, then the frozen original diffusion decoder executes
only the selected candidate. The noise critic remains the actor objective.

The numerical kernels are preserved from the reported experiment runtime.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from faster.agents.cached_dsrl_learner import all_finite, ema
from faster.agents.action_predictor import predict

def fixed_logp(mean, log_std, pre):
    """Normalized tanh density at fixed pre-tanh samples; never invert tanh."""
    normal = -.5 * jnp.sum(((pre-mean)*jnp.exp(-log_std))**2 +
                           2*log_std + jnp.log(2*jnp.pi), axis=-1)
    jac = jnp.sum(2*(jnp.log(2.)-pre-jax.nn.softplus(-2*pre)), axis=-1)
    return normal-jac


@partial(jax.jit, static_argnames=('surrogate_enabled',))
def collect_set(agent, student, observations, key, surrogate_enabled=True):
    """Original iid N8 stream; a single true decoder call for the winner."""
    mean, log_std = agent.actor.apply_fn({'params':agent.actor.params}, observations)
    keys = jax.vmap(lambda i:jax.random.fold_in(jax.random.fold_in(key,0),i))(jnp.arange(8))
    eps = jax.vmap(lambda k:jax.random.normal(k,mean.shape))(keys).swapaxes(0,1)
    pre = mean[:,None]+jnp.exp(log_std[:,None])*eps
    bank = agent.config.noise_scale*jnp.tanh(pre)
    repeated = jnp.repeat(observations,8,axis=0)
    noise = bank.reshape(-1,agent.config.latent_dim)
    if surrogate_enabled:
        actions = predict(student,repeated,noise)
        scores = agent.qa.apply_fn({'params':agent.qa.params},repeated,actions[:,:agent.config.prefix_dim]).min(0)
    else:
        scores = agent.qz.apply_fn({'params':agent.qz.params},repeated,noise).min(0)
    scores = scores.reshape(len(observations),8)
    winner = scores.argmax(-1)
    selected = bank[jnp.arange(len(observations)),winner]
    actions = agent.decode(observations,selected)
    logp = fixed_logp(mean[:,None],log_std[:,None],pre)
    mapping = dict(observations=observations,noise=selected,actions=actions,
        selected_learned=jnp.ones(len(observations),dtype=bool),selected_index=winner,
        selected_q=scores[jnp.arange(len(observations)),winner])
    trace = dict(pre=pre,old_logp=logp.sum(-1),old_per_logp=logp,candidate_q=scores,
        baseline=jnp.zeros(len(observations)),
        independent_baseline_available=jnp.zeros(len(observations),dtype=bool),
        scored_candidates=jnp.full((len(observations),),8))
    return mapping,jax.tree_util.tree_map(jax.lax.stop_gradient,trace)


def selector_diagnostics(agent, student, observations, key):
    """Read-only current-policy bank checks against exact QA(D), not true return.

    All eight candidates are decoded only for this diagnostic. These decoder
    rows must be reported separately from the one-call execution policy.
    """
    bank=agent.candidates(observations,key,8,0)
    repeated=jnp.repeat(observations,8,axis=0)
    noise=bank.reshape(-1,agent.config.latent_dim)
    actions=agent.decode(repeated,noise)
    exact=agent.qa.apply_fn({'params':agent.qa.params},repeated,
        actions[:,:agent.config.prefix_dim]).min(0).reshape(len(observations),8)
    qz=agent.qz.apply_fn({'params':agent.qz.params},repeated,noise).min(0).reshape(exact.shape)
    exact,qz=np.asarray(exact),np.asarray(qz)
    if not np.isfinite(exact).all() or not np.isfinite(qz).all():
        raise FloatingPointError('Nonfinite selector diagnostic')
    result=dict(states=len(observations),candidates=8,diagnostic_decoder_rows=len(noise),
        reference='Current learned QA(s,D(s,z)); not environment-return ground truth')
    def compare(scores):
        selected=scores.argmax(-1)
        regret=exact.max(-1)-exact[np.arange(len(exact)),selected]
        left,right=np.triu_indices(8,1)
        truth=exact[:,left]-exact[:,right]
        proposed=scores[:,left]-scores[:,right]
        informative=np.abs(truth)>1e-8
        return dict(top1_agreement=float(np.mean(selected==exact.argmax(-1))),
            mean_QA_regret=float(regret.mean()),max_QA_regret=float(regret.max()),
            ordering_agreement=float(np.mean(np.sign(truth[informative])==np.sign(proposed[informative])))
                if informative.any() else 1.,score_mse=float(np.mean((scores-exact)**2)))
    result['qz']=compare(qz)
    if student is not None:
        predicted=predict(student,repeated,noise)
        scores=agent.qa.apply_fn({'params':agent.qa.params},repeated,
            predicted[:,:agent.config.prefix_dim]).min(0).reshape(exact.shape)
        error=np.asarray(predicted-actions)
        if not np.isfinite(error).all() or not np.isfinite(np.asarray(scores)).all():
            raise FloatingPointError('Nonfinite student diagnostic')
        result['student']=dict(compare(np.asarray(scores)),action_mse=float(np.mean(error**2)),
            action_mae=float(np.mean(np.abs(error))),action_max_abs_error=float(np.max(np.abs(error))))
    return result


@jax.jit
def update_qa_surrogate(agent, student, batch):
    if not agent.decoder_frozen:
        raise ValueError('Freeze BC before RL')
    rng,key = jax.random.split(agent.rng)
    selected,_ = collect_set(agent,student,batch['next_observations'],key,True)
    target = batch['rewards']+batch['discounts']*batch['masks']*agent.qa.apply_fn(
        {'params':agent.qa_target},batch['next_observations'],selected['actions'][:,:agent.config.prefix_dim]).min(0)
    target = jax.lax.stop_gradient(target)
    valid = batch['valids']
    def loss(params):
        q = agent.qa.apply_fn({'params':params},batch['observations'],batch['actions'][:,:agent.config.prefix_dim])
        return jnp.sum((q-target[None])**2*valid[None])/jnp.maximum(2*valid.sum(),1),q
    (value,q),grads = jax.value_and_grad(loss,has_aux=True)(agent.qa.params)
    qa = jax.lax.cond(valid.sum()>0,lambda _:agent.qa.apply_gradients(grads=grads),lambda _:agent.qa,None)
    target_params = jax.lax.cond(valid.sum()>0,
        lambda _:ema(agent.qa_target,qa.params,agent.config.qa_target_beta),lambda _:agent.qa_target,None)
    return agent.replace(qa=qa,qa_target=target_params,rng=rng),dict(
        **{'qa/loss':value,'qa/grad_norm':optax.global_norm(grads),'qa/q_mean':q.mean(),
           'qa/target_mean':target.mean(),'qa/valid_fraction':valid.mean(),'qa/updated':valid.sum()>0},
        finite=all_finite((value,grads,qa.params,qa.opt_state,target_params,target))),selected
