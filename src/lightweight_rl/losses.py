##Joint distillation loss module for lightweight RL agents
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    ##Combined Knowledge Distillation Loss: L(Total) = alpha * L_KL(Student_logits/T, Teacher_logits/T) + (1-alpha)*L_MSE(student_q, teacher_q)
    def __init__(self, temperature: float = 2.0, alpha:float=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.mse = nn.MSELoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor)-> torch.Tensor:
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)

        ##Policy Distribution Loss(KL Divergence scaled to t^2)
        distill_loss = self.kl_div(soft_prob, soft_targets) * (self.temperature ** 2)
        q_loss = self.mse(student_logits, teacher_logits) # Direct Q-Value Regression Loss (MSE)
        return (self.alpha * distill_loss) + ((1.0 - self.alpha)*q_loss)
