"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
from xuanpolicy.xuanpolicy_torch.learners import *


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer: Optional[SummaryWriter] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 modeldir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(IQL_Learner, self).__init__(config, policy, optimizer, scheduler, summary_writer, device, modeldir)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().view(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().view(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        _, _, q_eval = self.policy(obs, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().view([self.args.batch_size, self.n_agents, 1]))
        q_next = self.policy.target_Q(obs_next, IDs)

        if self.args.double_q:
            _, action_next_greedy, q_next_eval = self.policy(obs_next, IDs)
            q_next_a = q_next.gather(-1, action_next_greedy.unsqueeze(-1).long())
        else:
            q_next_a = q_next.max(dim=-1, keepdim=True).values

        if self.args.consider_terminal_states:
            q_target = rewards + (1-terminals) * self.args.gamma * q_next_a
        else:
            q_target = rewards + self.args.gamma * q_next_a

        # calculate the loss function
        q_eval_a *= agent_mask
        q_target *= agent_mask
        loss = self.mse_loss(q_eval_a, q_target.detach())
        self.optimizer.zero_grad()  
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.writer.add_scalar("learning_rate", lr, self.iterations)
        self.writer.add_scalar("loss_Q", loss.item(), self.iterations)
        self.writer.add_scalar("predictQ", q_eval_a.mean().item(), self.iterations)
