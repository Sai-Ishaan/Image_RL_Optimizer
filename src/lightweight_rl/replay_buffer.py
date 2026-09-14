import random
import torch

class TrajectoryReplayBuffer:
    ##Trajectory-aware Replay Buffer for sequence models (SSMs, Transformers, LNNs)
    #Stores transtions and samples contiguous sequences of transitions for training sequence models
    def __init__(self, capacity: int=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: torch.Tensor, teacher_logits: torch.Tensor, done: bool=False):
        ##Stores a single step. 
        ##State: Tensor of shape (1,4,84,84) or (4, 84, 84)
        #teacher_logits: Tensor of shape (num_action, ) or (1, num_actions)

        s = state.squeeze(0).cpu() if state.ndim == 4 else state.cpu()
        l =(
            teacher_logits.squeeze(0).cpu()
            if teacher_logits.ndim == 2
            else teacher_logits.cpu()
        )
        item = (s, l, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample_trajectories(self, batch_size: int, seq_len: int):
        ###Sample a batch of contiguous sequences of transitions of length seq_len
        # returns: states_batch: Tensor of shape (batch_size, seq_len, 4, 84, 84)
        #          logits_batch: Tensor of shape (batch_size, seq_len, num_actions)
        max_start = len(self.buffer) - seq_len
        if max_start <= 0 :
            raise ValueError(f"Buffer size {len(self.buffer)} is too small to sample sequences of length {seq_len}.") 

        valid_starts = []
        for i in range(max_start):
            has_boundary = any(self.buffer[j][2] for j in range(i, i+seq_len - 1))
            if not has_boundary:
                valid_starts.append(i)

        if not valid_starts:
            valid_starts = list(range(max_start))

        sampled_starts = random.sample(valid_starts, min(batch_size, len(valid_starts)))

        batch_states = []
        batch_logits = []

        for start in sampled_starts:
            seq_states = [ self.buffer[j][0] for j in range(start, start + seq_len)]
            seq_logits = [ self.buffer[j][1] for j in range(start, start + seq_len)]

            batch_states.append(torch.stack(seq_states))
            batch_logits.append(torch.stack(seq_logits))

        return torch.stack(batch_states), torch.stack(batch_logits)

    def __len__(self):
        return len(self.buffer)