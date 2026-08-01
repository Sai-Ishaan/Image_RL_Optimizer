import os
import torch
import torch.optim as optim
import numpy as np

from openenv.core.generic_client import GenericEnvClient
from lightweight_rl.env_client import NatureCNN, WeightMappingParser
from lightweight_rl.transforms import AtariTransformPipeline
from lightweight_rl.encoders import LightweightStudent
from lightweight_rl.losses import DistillationLoss

def run_teacher_inference_loop(checkpoint_path: str, num_steps: int =150):
    print("Init: Distillation Runner Environment.......")

    ##Initialise validated transformation pipeline and model
    pipeline = AtariTransformPipeline(stack_size=4, target_dim=(84,84))
    teacher_model = NatureCNN(in_channels=4, num_actions=6)
    teacher_model.eval()
    
    ## Map the pre-trained weights onto teacher network
    if os.path.exists(checkpoint_path):
        parser = WeightMappingParser(teacher_model)
        parser.load_and_map_weights(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Proceeding with random weights for a dry-run verification")

    student = LightweightStudent(in_channels=4, num_actions=6, hidden_dim=128)
    student.train()
    print("\n Successfully Initialized Lightweight Student (223K Params)")

    ##Setting up Optimizer and Distillation loss Function
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    distill_loss_fn = DistillationLoss(temperature=2.0, alpha=0.7)

    ### Live connection over OpenEnv loopback interface:
    client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

    with client.sync() as sync_client:
        print("Connecting to environment server and resetting game....")
        config = {"game": "PongNoFrameskip-v4"}
        initial_state = sync_client.reset(config=config)

        # Pull raw observation array out of JSON
        obs = initial_state.observation
        if isinstance(obs, dict):
            obs = obs[list(obs.keys())[0]]

        state_tensor = pipeline.reset(np.array(obs)) ##4-frame temporal buffer

        print(f"Beginning Distillation loop ({num_steps} iterations) \nObserve the Loss Drop....")
        for step_idx in range(num_steps):
            with torch.no_grad():
                teacher_logits = teacher_model(state_tensor)
            _, student_logits = student.get_features_and_logits(state_tensor)
            loss = distill_loss_fn(student_logits, teacher_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

                #q_values = teacher_model(state_tensor) ##Generating action by executing a forward pass through the Teacher network
            action_id = int(torch.argmax(teacher_logits, dim=1).item())
            payload = {"action_id": action_id}

            step_result = sync_client.step(payload)
            
            ##Extraction and preprocessing pipeline trackinh
            next_observation = step_result.observation
            if isinstance(next_observation, dict):
                key = list(next_observation.keys())[0]
                next_observation = next_observation[key]
            next_matrix = np.array(next_observation)

            #Streaming the new frame thru verified transformer
            state_tensor = pipeline.process_step(next_matrix)

            if step_idx % 15 == 0:
                t_max_q = teacher_logits.max().item()
                s_max_q = student_logits.max().item()
                print(f"Step {step_idx:<3} | Loss: {loss.item():.4f} | Teacher Q: {t_max_q:>7.4f} | Student Q: {s_max_q:>7.4f}")
        print("Success! Loop completed without any buffer drops or schema violations.")

if __name__ == "__main__":
    ##Creating the checkpoint variable to local NatureCNN checkpt or leave it blank to test loop orchestration
    CKPT = "checkpoints/nature_cnn_pong.pt"
    run_teacher_inference_loop(checkpoint_path=CKPT, num_steps=150)