import time
import os
import torch
import psutil
import numpy as np
from openenv.core.generic_client import GenericEnvClient
from lightweight_rl.transforms import AtariTransformPipeline
from lightweight_rl.env_client import NatureCNN, WeightMappingParser
from lightweight_rl.encoders import LightweightStudent
from lightweight_rl.visualizer import Visualizer

def run_benchmark(num_steps: int=200):
    print("=" *40)
    print(" ------ Lightweight RL Toolkit: Hardware and Performance Benchmark ------")
    print("="*40)

    process = psutil.Process(os.getpid())
    pipeline = AtariTransformPipeline(stack_size=4, target_dim=(84,84))
    visualizer = Visualizer(frame_size=(360,360))

    ##Initialise networks
    teacher = NatureCNN(in_channels=4, num_actions=6).eval()
    parser = WeightMappingParser(teacher)
    parser.load_and_map_weights("checkpoints/nature_cnn_pong.pt")

    student = LightweightStudent(in_channels=4, num_actions=6, hidden_dim=128).eval()
    student_ckpt = os.path.join("checkpoints", "student_pong.pt")

    if os.path.exists(student_ckpt):
        student.load_state_dict(torch.load(student_ckpt))
        print(f"Successfully Loaded Student model checkpoint from {student_ckpt}")
    else:
        print(f"Warning: Student checkpoint not found at {student_ckpt}. Proceeding with random weights.")
    client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation") ##OpenEnv Setup

    teacher_latencies = []
    student_latencies = []

    with client.sync() as sync_client:
        state = sync_client.reset(config={"game": "PongNoFrameskip-v4"})
        obs = state.observation
        if isinstance(obs, dict):
            obs = obs[list(obs.keys())[0]]

        raw_frame = np.array(obs)
        state_tensor = pipeline.reset(raw_frame)

        print("\n Starting Live Telemetry Benchmark Window: Press Ctrl+C in terminal to exit...")

        for _ in range(num_steps):
            ##Measure Teacher Latency
            t0 = time.perf_counter() ##Measuring Teacher latency
            with torch.no_grad():
                teacher_logits = teacher(state_tensor)
            t_lat = (time.perf_counter() - t0) *1000.0
            teacher_latencies.append(t_lat)

            ##Measure Student latency
            t1 = time.perf_counter()
            with torch.no_grad():
                student_logits = student(state_tensor)
            s_lat = (time.perf_counter() - t1) * 1000.0
            student_latencies.append(s_lat)  

            ##Extraction Q_VALUES/Actions from logits
            t_act = int(torch.argmax(teacher_logits, dim=1).item())
            s_act = int(torch.argmax(student_logits, dim=1).item())

            t_q = teacher_logits.max().item()
            s_q = student_logits.max().item()

            ## Render Side By Side with OpenCV HUD
            visualizer.render_side_by_side(
                teacher_frame = raw_frame,
                student_frame = raw_frame,
                t_action=t_act, t_q=t_q, t_lat=t_lat,
                s_action=s_act, s_q=s_q, s_lat=s_lat
            )

            step_res = sync_client.step({"action_id": t_act})
            next_obs = step_res.observation
            if isinstance(next_obs, dict):
                next_obs = next_obs[list(next_obs.keys())[0]]
            raw_frame = np.array(next_obs)

            state_tensor = pipeline.process_step(raw_frame)

    ram_mb = process.memory_info().rss / (1024 *1024)
    avg_t_lat = np.mean(teacher_latencies)
    avg_s_lat = np.mean(student_latencies)
    speedup = avg_t_lat / avg_s_lat if avg_s_lat >0 else 1.0

    print("\n", "="*60 )
    print("---------- BENCHMARK EVALUATION ----------")
    print("="*60)

    print(f" ->RAM Footprint :{ram_mb:.2f} MB")
    print(f" ->Teacher Avg Latency :{avg_t_lat:.2f} ms")
    print(f" ->Student Avg Latency :{avg_s_lat:.2f} ms")
    print(f" -> Inference Speedup (Teacher/Student): {speedup:.2f}x Faster")
    print("="*60)

if __name__ == "__main__":
    run_benchmark(num_steps=200)    