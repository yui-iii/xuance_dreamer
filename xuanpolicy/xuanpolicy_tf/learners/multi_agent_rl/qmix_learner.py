"""
Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning
Paper link:
http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf
Implementation: TensorFlow 2.X
"""
from xuanpolicy.xuanpolicy_tf.learners import *


class QMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: tk.Model,
                 optimizer: tk.optimizers.Optimizer,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: str = "cpu:0",
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        super(QMIX_Learner, self).__init__(config, policy, optimizer, summary_writer, device, modeldir)

    def update(self, sample):
        self.iterations += 1
        with tf.device(self.device):
            state = tf.convert_to_tensor(sample['state'])
            state_next = tf.convert_to_tensor(sample['state_next'])
            obs = tf.convert_to_tensor(sample['obs'])
            actions = tf.convert_to_tensor(sample['actions'], dtype=tf.int64)
            obs_next = tf.convert_to_tensor(sample['obs_next'])
            rewards = tf.reduce_mean(tf.convert_to_tensor(sample['rewards']), axis=1)
            terminals = tf.reshape(tf.convert_to_tensor(sample['terminals'], dtype=tf.float32), [-1, self.n_agents, 1])
            agent_mask = tf.reshape(tf.convert_to_tensor(sample['agent_mask'], dtype=tf.float32), [-1, self.n_agents, 1])
            IDs = tf.tile(tf.expand_dims(tf.eye(self.n_agents), axis=0), multiples=(self.args.batch_size, 1, 1))
            batch_size = obs.shape[0]

            with tf.GradientTape() as tape:
                inputs_policy = {"obs": obs, "ids": IDs}
                _, _, q_eval = self.policy(inputs_policy)
                q_eval_a = tf.gather(q_eval, tf.reshape(actions, [self.args.batch_size, self.n_agents, 1]), axis=-1, batch_dims=-1)
                q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask, state)
                inputs_target = {"obs": obs_next, "ids": IDs}
                q_next = self.policy.target_Q(inputs_target)

                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(inputs_target)
                    action_next_greedy = tf.reshape(tf.cast(action_next_greedy, dtype=tf.int64),
                                                    [batch_size, self.n_agents, 1])
                    q_next_a = tf.gather(q_next, action_next_greedy, axis=-1, batch_dims=-1)
                else:
                    q_next_a = tf.reduce_max(q_next, axis=-1, keepdims=True)
                q_tot_next = self.policy.target_Q_tot(q_next_a * agent_mask, state_next)
                if self.args.consider_terminal_states:
                    q_tot_target = rewards + (1-terminals) * self.args.gamma * q_tot_next
                else:
                    q_tot_target = rewards + self.args.gamma * q_tot_next

                # calculate the loss function
                q_tot_eval = tf.reshape(q_tot_eval, [-1])
                q_tot_target = tf.stop_gradient(tf.reshape(q_tot_target, [-1]))
                loss = tk.losses.mean_squared_error(q_tot_target, q_tot_eval)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients([
                    (grad, var)
                    for (grad, var) in zip(gradients, self.policy.trainable_variables)
                    if grad is not None
                ])

            if self.iterations % self.sync_frequency == 0:
                self.policy.copy_target()

            lr = self.optimizer._decayed_lr(tf.float32)
            self.writer.add_scalar("learning_rate", lr.numpy(), self.iterations)
            self.writer.add_scalar("loss_Q", loss.numpy(), self.iterations)
            self.writer.add_scalar("predictQ", tf.math.reduce_mean(q_eval_a).numpy(), self.iterations)
