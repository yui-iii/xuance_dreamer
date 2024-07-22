"""
Advantage Actor-Critic (A2C)
Implementation: TensorFlow2
"""
from argparse import Namespace
from xuance.common import Optional
from xuance.tensorflow import tf, tk, Module
from xuance.tensorflow.learners import Learner


class A2C_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: Module):
        super(A2C_Learner, self).__init__(config, policy)
        if ("macOS" in self.os_name) and ("arm" in self.os_name):  # For macOS with Apple's M-series chips.
            self.optimizer = tk.optimizers.legacy.Adam(config.learning_rate)
        else:
            self.optimizer = tk.optimizers.Adam(config.learning_rate)
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef

    @tf.function
    def learn(self, obs_batch, act_batch, ret_batch, adv_batch):
        with tf.GradientTape() as tape:
            outputs, _, v_pred = self.policy(obs_batch)
            a_dist = self.policy.actor.dist
            log_prob = a_dist.log_prob(act_batch)

            a_loss = -tf.reduce_mean(adv_batch * log_prob)
            c_loss = tk.losses.mean_squared_error(ret_batch, v_pred)
            e_loss = tf.reduce_mean(a_dist.entropy())

            loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients([
                (tf.clip_by_norm(grad, self.grad_clip_norm), var)
                for (grad, var) in zip(gradients, self.policy.trainable_variables)
                if grad is not None
            ])
        return a_loss, c_loss, e_loss, v_pred

    def update(self, **samples):
        self.iterations += 1
        obs_batch = samples['obs']
        act_batch = samples['actions']
        ret_batch = samples['returns']
        adv_batch = samples['advantages']
        a_loss, c_loss, e_loss, v_pred = self.learn(obs_batch, act_batch, ret_batch, adv_batch)

        info = {
            "actor-loss": a_loss.numpy(),
            "critic-loss": c_loss.numpy(),
            "entropy": e_loss.numpy(),
            "predict_value": tf.math.reduce_mean(v_pred).numpy()
        }

        return info
