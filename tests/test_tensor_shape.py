from src.lightweight_rl.replay_buffer import TrajectoryReplayBuffer
import torch

buffer = TrajectoryReplayBuffer(capacity=100)

##Mock frame test
for i in range(50):
    mock_state = torch.randn(1,4,84,84)
    mock_logits = torch.randn(1,6)

    buffer.push(mock_state, mock_logits, done=(i==25))

states, logits = buffer.sample_trajectories(batch_size=8, seq_len=16)

print(f"Sampled States shape: {states.shape}")
print(f"Sampled Logits shape: {logits.shape}")
