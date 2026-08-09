# Light-weight Image-Based RL Toolkit for Low-End Hardware

* **Status**: Draft - Pre-Issue Prototyping
* **Target Architecture**: External Companion Toolkit for `huggingface/OpenEnv` (`envs/atari_env`) or a plug-and-play system for OpenEnv
* **Last Updated**: 2026-08-01

---

## Summary of Problem Statement

### Challenge

Image-Based Reinforcement Learning (RL) projects often take long training times, slow iterations, thousands of epochs of training runs, and large wall-clock times, especially on low-end GPUs such as Intel Iris Xe, Nvidia MX450, and AMD Radeon RX 6400 LP. This makes experimentation, development, and benchmark validation slow and costly. The following problems are a result of the hardware bottleneck:

* **Protracted Iteration loops and Slow Feedback Loop:** Long wall-clock training prevents rapid prototyping and tuning of architectures and hyperparameters, causing a slow feedback loop.
* **Low Sample Efficiency:** Heavy models need more environment steps to reach good rewards and extract useful spatial representations, compounding the compute deficit.
* **Network & IPC Overhead:** When using distributed or containerized environment interfaces like **OpenEnv**, transferring high-resolution raw pixel tensors ($84 \times 84 \times 4$ frame stacks) over a client-server boundary such as HTTP/WebSocket loopback introduces substantial Inter-Process Communication (**IPC**) latency that can easily dwarf model forward-pass times on low-end Host CPUs.

### Proposed Solution

The following toolkit acts as a proof of concept for a highly optimized reproducible workflow. Inspired by *Software-based Optimization* methods, this toolkit aims to lower the barrier of entry for image-based RL. Instead of training complex agents from scratch on constrained devices, this project introduces a standalone companion toolkit that pairs Knowledge Distillation (KD) with hardware-aware structural optimizations. By leveraging a high-performing, pre-trained teacher model, we distill its policy, value outputs, and internal representations into an ultra-lightweight student-network optimized specifically for low-VRAM, lower compute target devices.

---

## Technical and Design Decisions

### (I) Environment Interoperability: How does OpenEnv and atari_env fit in?

OpenEnv acts **strictly** as an *environment interface* and *network abstraction layer* rather than a training library. It isolates the environment runtime inside a Docker Container and exposes state interactions asynchronously. In order to maximize throughput on low-end devices, this project interfaces with **envs/atari_env** under **HuggingFace/OpenEnv** using the following strategies:

* **Server-Side minimization:** We configure `atari_env` container to handle downsampling and grayscale conversion natively, rather than pulling heavy RGB frames across the container side and processing them on the client side. This can be achieved natively using environment variables (`ATARI_OBS_TYPE=grayscale`, `ATARI_FRAMESKIP=4`). This drastically reduces the network serialization payload per step.
* **Idiomatic Custom Transforms:** Client-side frame stacking and final tensor formatting are implemented by extending `openenv.core.Transform`. This ensures seamless integration with OpenEnv's data structures while keeping local client runtime lean.

### (II) Procurement of Teacher Policy

To avoid spending critical hardware cycles and falling into a compute trap, the teacher model will **not** be trained from scratch. Instead, the toolkit pulls verified, high-performance pre-trained weights (for example, Standard NatureCNN checkpoints from open-source baselines like CleanRL or Stable-Baselines3). A structural adapter maps these static state-dicts to our local PyTorch environment, providing an immediate, high-fidelity training signal for behavioral cloning and distillation.

### (III) Student Backbone Selection and Hardware Safeguards

* Although *MobileNetV3-Small* serves as the default student encoder due to its minimal parameter footprint, its heavy reliance on *depthwise-separable convolutions* can occasionally encounter memory-bandwidth bottlenecks on low-end desktop or integrated laptop GPUs.
* Alongside *MobileNetV3*, the toolkit implements a highly optimized *Shallow ResNet (Mini-ResNet)* config as a fast ablation alternate (controlled removal or modification of a component in the Neural Network). This allows users to benchmark which layer composition achieves superior hardware utilization on their specific silicon.

### (IV) Distillation Loss Formulation

The student network is optimized via a joint loss function that enforces alignment across action selection, value estimation, and latent space representations:

Where:

* $L_{\text{RL}}$ is the *Standard Environment Task loss* (e.g., PPO, or PPO clipped objectives).
* $D_{\text{KL}}(\pi_{T} \parallel \pi_{S})$ represents the Kullback-Leibler divergence between the teacher and student policy distributions, scaled by the temperature parameter, $T$.
* $\Vert{}V_{T} - V_{S}\Vert{}_2^2$ is the $L_2$ loss matching the student's value network predictions directly to the teacher's state-value estimates.
* $L_{\text{repr}}$ is an optional MSE (Mean Squared Error) loss enforcing alignment between projected latent features of both encoders.

### (V) Quantization and Inference Export

Post-Training Quantization (PTQ) directly into INT8 using standard PyTorch frameworks is primarily optimized for x86 CPUs and often experiences limited performance gains or runtime bugs on low-end GPUs. To resolve this, the toolkit exports the fully distilled PyTorch student model to the **ONNX Runtime** ecosystem, applying device-targeted quantization or FP16 execution explicitly optimized for low-end graphics cards.

---

## Project Setup, Environment, and Dependencies

This project is structured as an independent companion repository that consumes the `openenv-core` API Client.

### Core Stack and Version Matrix

| Library / Dependency | Version | Purpose |
| --- | --- | --- |
| **Python** | `3.11.x` | Base language runtime offering optimal performance and ecosystem stability. |
| **torch** | `2.5.1` | Core ML framework utilized for student network execution and joint loss computation. |
| **torchvision** | `0.20.1` | Source library for standard MobileNetV3 backbones. |
| **openenv-core** | `0.2.0` | Client-side environment communication protocol layer. |
| **atari-env** | `0.2.0` | Client bindings specifically targeting the Arcade Learning Environment. |
| **stable-baselines3** | `2.4.0` | Utilized strictly for sourcing and mapping pre-trained NatureCNN teacher checkpoints. |
| **onnx** | `1.17.0` | Open Neural Network Exchange graph serialization format. |
| **onnxruntime-gpu** | `1.20.0` | High-efficiency deployment engine optimized for CPU and low-end GPU targets. |
| **ruff** | `0.9.1` | Fast linter and code formatter matching standard upstream constraints. |

---

## Staged Implementation Plan

### Phase 0: Environment and IPC Overhead Benchmarking

* Run the local `atari_env` Docker container image.
* Implement a diagnostic test script to measure a baseline step latency using a completely random/no-op policy.
* Calculate and record pure network Round-Trip Time (**RTT** / IPC Overhead) to serve as a baseline value that can be cleanly separated from the model compute metrics in subsequent phases.

#### Logical and Mathematical Trace: Phase 0

1. **Random Agent Operator:**
* At this stage, we have no neural network. The agent selects an action $a_t$ at time $t$, by sampling uniformly from the discrete Atari action space ($A$):

$$a_t \sim \mathcal{U}(A)$$




2. **The Processing Pipeline:**
* **Client Request:** Client sends an action $a_t$ via HTTP/WebSocket to the `atari_env` Docker container.
* **Environment Step:** The Emulator (ALE) transitions from a state $s_t$ to $s_{t+1}$, generating a reward $r_{t+1}$ and a boolean flag indicating if the episode is done.
* **Server-Side Downsampling:** This is the first lightweight optimization. Instead of sending RAW RGB frames, the server processes the image based on our environment variables (`ATARI_OBS_TYPE=grayscale`, `ATARI_FRAMESKIP=4`).
* **Client Receives data:** The tuple $(s_{t+1}, r_{t+1}, \text{done})$ is serialized, transmitted, and deserialized into Python objects.



#### Math of 'Making it Lightweight'

* Raw RGB Frame is represented as: $210 \times 160 \times 3 \text{ channels} = 100,800 \text{ Bytes}$.
* Grayscale Downsampled frame is represented as: $84 \times 84 \times 1 \text{ channel} = 7,056 \text{ Bytes}$.
* We achieve an approximate **14x reduction** in network payload size per environment step by moving the preprocessing to the server.
* For the Baseline IPC Benchmark, we calculate the total time taken for one step:

Since the agent chooses a random integer, $T_{\text{client\_compute}} \approx 0$. Hence, any latency measured in the test script isolates exactly how long the OpenEnv network and emulator overhead takes. This acts as our **Zero Point** benchmark.

**Breakdown of the IPC Connection:**

#### Issues in Phase 0:

Multiple issues related to client script and running processes being out of sync regarding how they format the data packets:

1. **Shape Mystery (Got (100800,)):**
* Dimension calculation: $210 \times 160 \times 3 = 100,800$.
* This reveals two critical issues:
* The server completely ignored our `ATARI_OBS_TYPE=grayscale` environment flags set inside PyTest. Since the server process was executed manually using `uv run` in a separate shell, it cannot see the environment variables injected into the PyTest terminal.
* The server is transmitting the raw uncompressed, full color RGB Frame as a flattened 1D array of 100,800 numbers over the WebSocket.




2. **Action Formatter Crash (`'int' object is not iterable`):**
* The Stack trace inside `openenv/core/generic_client.py` revealed the internal data normalization contract:
```python
return dict(action)  # Crashed here when given an integer.

```


* The `GenericEnvClient` requires actions to be in a dictionary wrapper, but the server expects internal mapping payload keys to be exact.



#### Solutions:

* **Duck-Typing the Client Constraint:** We can use a simple duck-typing pattern to satisfy both layers. By wrapping the action in a lightweight mock object that implements a `.model_dump()` method, we pass right through the client's filter and deliver raw integer scalar directly to the wire socket.
* **Handling flattening and color space directly in test script:** By dynamically parsing the array shape, we can reshape our flattened array of 100,800 elements to its true raw dimensions $(210, 160, 3)$ or downsample it.
* **Align the dictionary action wrapper:** The generic client requires a dict wrapper; hence, we shall use the standard structural frame key wrapper `{"value": 1}` or generic format, ensuring it passes the internal `isinstance(action, dict)` gate safely.

#### Phase 0 Status:

Connection established! We have completely bypassed the Pydantic validation wall. The test has successfully executed 10 full benchmark rounds plus warmup iterations without throwing a single server error or dropping the connection.

**Test Log:**

```text
pytest .\tests\test_ipc_baseline.py --benchmark-sort=mean
============================= test session starts =============================
platform win32 -- Python 3.11.5, pytest-8.2.0, pluggy-1.6.0
benchmark: 5.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: \Image_RL_Optimizer
configfile: pyproject.toml
plugins: anyio-4.14.0, benchmark-5.2.3
collecting 2 items                                                            collected 2 items                                                              

te.                                            [100%]

============================== warnings summary ===============================
venv\Lib\site-packages\opentelemetry\util\_importlib_metadata.py:32
  \Image_RL_Optimizer\venv\Lib\site-packages\opentelemetry\util\_importlib_metadata.py:32: DeprecationWarning: SelectableGroups dict interface is deprecated.Use select.
    return EntryPoints(ep for group_eps in eps.values() for ep in group_eps)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

----------------------------------------------------- benchmark: 1 tests -----------------------------------------------------
Name (time in ms)                     Min       Max     Mean   StdDev   Median    IQR  Outliers      OPS  Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------
test_random_agent_ipc_latency     44.3017  115.3127  63.2626  25.0615  51.3271 21.0647       2;2  15.8071      10           1
------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
======================== 2 passed, 1 warning in 9.53s =========================

```

#### Data Breakdown:

* **Mean Latency (63.26ms):** On average, a single `step()` takes about 63ms to travel to the server, execute the Pong env, and return the New State.
* **Variance/Stability (StdDev = 25.06ms):** There seems to be a significant jitter here. The minimum time was 44.3ms, but spiked up to 115.3ms. This suggests network stack overhead, garbage collection, or unpredictable serialization times getting in the way.
* **Throughput/OPS (15.8):** Currently operating at 15.8 Operations / Environment steps / Second.

Although the connection is stable, **15.8 steps/second** is still a severe bottleneck for Reinforcement Learning. For perspective, a standard native Gym instance of Atari Pong can easily push about 1000–2000 steps per second. At ~15 OPS, training a high-performance agent will take an eternity. This massive overhead is due to JSON serializing and deserializing the raw $210 \times 160 \times 3$ (100,800 elements) observation array over the loopback interface on every step.

---

### Phase 1: Teacher Weight Mapping

* Download pre-trained NatureCNN weights for `PongNoFrameskip-v4` and `BreakoutNoFrameskip-v4`.
* Construct a lightweight state-dict parser to load the weights into a local PyTorch module.
* Run evaluation rollouts against the `atari_env` client wrapper to verify performance parity with published baselines.

#### Approach for Phase 1:

**Strategy — Optimize the Pipeline without changing the payload:**

* In order to lower the bar for target consumer systems while preserving the vanilla $210 \times 160 \times 3$ image payload, we must refine how the data is moved.
* Currently, our CLI Binary in the virtual env is likely serializing raw observations into massive JSON lists. JSON parsing of a 100,800 element array in Python is notoriously slow and CPU-bound.
* Before building the entire toolkit, we propose adding our *Phase 1 Optimization* by moving away from JSON list serialization to a compact binary or zero-copy protocol (like Protobuf, Msgpack, or Shared Memory IPC) within the OpenEnv communication layer. This strips the serialization overhead without touching the underlying game states.

#### Phase Breakdown:

The Architecture is divided into 3 layers: **IO Layer**, **Processing Layer**, and **Orchestration Layer**:

1. **IO Layer:** Under `src/lightweight_rl`, we created `env_client.py` & `env_utils.py`. By keeping these separate, we can optimize serialization (the previous 63ms bottleneck) in `env_utils` without breaking the encoders or `train_distill` logic.
2. **Processing Layer:** `encoders.py` and `transforms.py` act as our Processing Layer. Here, the $210 \times 160 \times 3$ frame becomes an $84 \times 84 \times 4$ tensor. Keeping transforms distinct is crucial for performance.
3. **Orchestration Layer:** `train_distill.py` and `serve_student.py` separate the training process from the inference/serving process.

#### Transforms Breakdown:

* Currently, the benchmark shows the server emits raw observation data (which the test script reshapes to $210 \times 160 \times 3$ or a flat array). However, the NatureCNN model strictly expects a frame-stacked tensor of shape $(1, 4, 84, 84)$.
* To bridge this gap, `transforms.py` handles:
* **Downsampling and Cropping:** Converting the $210 \times 160$ frame down to a standard $84 \times 84$ matrix.
* **Normalization:** Converting integers $[0, 255]$ to floating point values $[0.0, 1.0]$.
* **Frame Stacking:** Maintaining a rolling history of the last 4 consecutive frames to capture temporal motion (velocity/acceleration), standard for published Atari baselines.


* **Benefits:**
* **No OpenCV/Pillow Overhead:** Utilizing basic NumPy indexing for downsampling keeps frame preprocessing within the microsecond range, preventing degradation of loopback latency baselines.
* **Autonomous Memory Management:** Using `deque()`, we replace manual tracking array multiplication. Old steps are dropped instantly as new ones arrive.
* **PyTorch Integration:** The output shape is fully prepared as a floating point tensor $(1, 4, 84, 84)$ ready to be passed directly into `NatureCNN.forward()`.



#### Train Distill:

Once components are verified, `train_distill.py` serves as the **Orchestrator**, tying together:

* Initializing the live `GenericEnvClient` loop.
* Instantiating NatureCNN and parsing its pre-trained weights via `WeightMappingParser`.
* Funneling live frames straight from the server through the `AtariTransformPipeline`.
* Running live policy/teacher inferences.

#### Observations:

```text
Init: Distillation Runner Environment.......
Warning: Checkpoint not found at checkpoints/nature_cnn_pong.pt. Proceeding with random weights for a dry-run verification
Connecting to environment server and resetting game....
Beginning active verification loop (50 iterations)....
0/50 completed
Selected Action: 4 | Highest Estimated Q_VALUE : 0.0218
10/50 completed
Selected Action: 4 | Highest Estimated Q_VALUE : 0.0226
20/50 completed
Selected Action: 4 | Highest Estimated Q_VALUE : 0.0226
30/50 completed
Selected Action: 4 | Highest Estimated Q_VALUE : 0.0226
40/50 completed
Selected Action: 4 | Highest Estimated Q_VALUE : 0.0224
Success! Loop completed without any buffer drops or schema violations.

```

Analyzing the pattern in this test log, we notice stagnation where the agent chooses Action 4 repeatedly and Q-Values barely move from $0.0226$:

1. **Random Weights Factor:** The console warns that no checkpoint was found, so NatureCNN executed with randomly initialized weights.
* A random neural network maps incoming frame stacks to a static, slightly biased set of outputs.
* Since weights are locked in `.eval()` mode without backpropagation updates, internal mapping parameters never change during these steps.


2. **Atari Pong's Frame Stagnation (Environment Factor):**
* When Atari Pong resets, it starts with several frames of an entirely static screen (such as green/orange background before the ball is served).
* Since the raw input matrix is identical for those first few frames, the output from NatureCNN will be mathematically identical down to the last decimal point. Once the ball actually moves, a random network's output will fluctuate slightly.



#### Log Conclusion:

The dry run achieved the following goals:

* **Zero Buffer Drops:** The loop executed 50 rapid steps over loopback without dropping frame sequences.
* **Schema Integrity:** Server and client communicated without validation syntax errors.
* **Pipeline Fluidity:** Raw frames transformed into $(1, 4, 84, 84)$ shape, fed into the forward pass, and extracted as `action_id` successfully.
* **Random Network Initialization:** Re-running the dry-run yields distinct runs back-to-back, confirming random initialization across instances and deterministic behavior during the environment loop.

#### Update in Phase 1:

We are taking an external pre-trained model (trained on Atari Pong by CleanRL) and translating its internal parameter dictionary (`state_dict`) into the naming scheme expected by the NatureCNN Architecture by:

1. **Loading the Compiled Model:** Loading `agent.pt` as a TorchScript module to bypass Windows/PyTorch security checks and deserialization errors.
2. **Extracting the Parameter Dictionary:** Pulling the key-value dictionary directly out of the TorchScript object using `.state_dict()`.
3. **Key Inspection and Translation:** Inspecting raw keys and mapping weight tensors to local variable names (e.g., mapping `conv1.weight` to `conv0.weight`).
4. **Saving a Clean Checkpoint:** Exporting a clean state dictionary into the `checkpoints` directory using `torch.save`.

**Why before moving to `encoders.py`?**

* **Teacher Capability:** Knowledge Distillation requires a teacher network that outputs meaningful Q-Values rather than random noise.
* **Establishing Baselines:** A true baseline teacher is needed to prove performance.
* **End-to-End Pipeline Verification:** Ensures teacher processing of live frames yields dynamic Q-Values before introducing student loss loops.

#### Observations in Phase 1 Mapping:

```text
1. Downloading raw PyTorch state dict from HuggingFace Hub({filename})
dqn_atari.cleanrl_model: 
....Downloaded to: XXXX\XXX\.cache\huggingface\hub\models--cleanrl--PongNoFrameskip-v4-dqn_atari-seed1\snapshots\7abf8b0255fad67c925b9bf4b28cbaa5691228ee\dqn_atari.cleanrl_model

2. Loading State Dictionary....

3. Mapping CleanRL network keys to NatureCNN...
 -> Successfully mapped network.0.weight to conv.0.weight | Shape: [32, 4, 8, 8]
 -> Successfully mapped network.0.bias to conv.0.bias | Shape: [32]
 -> Successfully mapped network.2.weight to conv.2.weight | Shape: [64, 32, 4, 4]
 -> Successfully mapped network.2.bias to conv.2.bias | Shape: [64]
 -> Successfully mapped network.4.weight to conv.4.weight | Shape: [64, 64, 3, 3]
 -> Successfully mapped network.4.bias to conv.4.bias | Shape: [64]
 -> Successfully mapped network.7.weight to fc.0.weight | Shape: [512, 3136]
 -> Successfully mapped network.7.bias to fc.0.bias | Shape: [512]
 -> Successfully mapped network.9.weight to fc.2.weight | Shape: [6, 512]
 -> Successfully mapped network.9.bias to fc.2.bias | Shape: [6]

[SUCCESS] Checkpoint saved to checkpoints/nature_cnn_pong.pt

```

**Analysis of Successful Output:**

* **Q-Values reflect real RL Math:** In Deep Q-learning, the network outputs expected cumulative discounted future rewards $Q(s,a)$. Real teacher weights evaluate the early Pong screen at Q-values between **1.5087** and **1.3657**.
* With a standard discount factor ($\gamma = 0.99$), expected returns typically fall between $-2.0$ and $+2.0$. Outputs in this range confirm active teacher weights.
* **Environment Responsiveness:** Q-values estimate the discounted sum of expected future rewards:

* During steps 0–30 (static serve screen), teacher outputs remained steady at $1.3657$. At step 40, when the ball is served and detected via the 4-frame temporal pipeline, risk is re-evaluated (dropping Q-Value to $-0.7301$), switching action to $5$ to react to the incoming ball.
* **Structural Layer Parity:** All 10 core weight and bias tensors of the standard NatureCNN architecture were aligned without truncation or shape errors:

$$\text{Conv1} \rightarrow \text{Conv2} \rightarrow \text{Conv3} \rightarrow \text{Linear1} \rightarrow \text{Linear2}$$



*(3 Convolutional Weights, 3 Convolutional Biases, 2 Fully-Connected Weights, 2 Fully-Connected Biases)*

---

### Phase 2: Student Architecture and Distillation Loss

Once the teacher outputs intelligent predictions, the Student Network and Distillation Loss Function are initialized. Knowledge Distillation combines two target losses:

1. **KL Divergence Loss (Policy Distillation):** Forces student action probability distribution to match teacher's temperature-scaled output.
2. **MSE Feature Loss (Representation Matching):** Encourages the student encoder to extract spatial features similar to those extracted by the teacher.

#### Phase 2 Observations:

Inline verification script confirming student network shape parity and parameter count savings:

```bash
$env:PYTHONPATH="src"; python -c "import torch; from lightweight_rl.encoders import LightweightStudent, StudentEncoder; s = LightweightStudent(); x = torch.randn(1, 4, 84, 84); feat, q = s.get_features_and_logits(x); print(f'Features: {feat.shape}, Q-Values: {q.shape}, Total Params: {sum(p.numel() for p in s.parameters()):,}')"

```

**Results:**

```text
Features: torch.Size([1, 1568]), Q-Values: torch.Size([1, 6]), Total Params: 223,190

```

* **Spatial Feature Bottleneck:** Student latent footprint is `[1, 1568]` compared to Teacher's (NatureCNN) `[1, 3136]` — a **50% smaller** footprint.
* **Action Output Dimensions:** Both Student and NatureCNN output `(1, 6)`, maintaining **100% Schema Parity**.
* **Total Parameters:** Reduced by **~86%** (from 1,684,230 parameters in NatureCNN to 223,190 in the Student Network).

#### Updates in Phase 2:

The `train_distill.py` main loop was updated from a read-only verification loop into an **active learning loop**.

**Distillation Training Output:**

```text
Init: Distillation Runner Environment.......        
Parsing checkpoint: checkpoints/nature_cnn_pong.pt  
Successfully mapped 10 layers into Teacher Network  
                          
Successfully Initialized Lightweight Student (223K Params)                 
Connecting to environment server and resetting game....
Beginning Distillation loop (150 iterations) 

Step 0   | Loss: 0.6788 | Teacher Q:  1.5087 | Student Q:  0.0790
Step 15  | Loss: 0.0114 | Teacher Q:  1.2700 | Student Q:  1.1941
Step 30  | Loss: 0.0028 | Teacher Q:  1.3657 | Student Q:  1.4717
Step 45  | Loss: 0.4302 | Teacher Q: -0.7354 | Student Q:  0.1321
Step 60  | Loss: 0.1042 | Teacher Q:  0.7371 | Student Q:  0.0699
Step 75  | Loss: 0.3152 | Teacher Q: -0.5212 | Student Q:  0.4301
Step 90  | Loss: 0.4771 | Teacher Q:  1.3657 | Student Q:  0.1265
Step 105 | Loss: 0.0922 | Teacher Q:  1.3657 | Student Q:  1.9654
Step 120 | Loss: 0.3465 | Teacher Q:  1.3657 | Student Q:  0.3859
Step 135 | Loss: 0.0630 | Teacher Q:  1.3657 | Student Q:  0.9900
Success! Loop completed without any buffer drops or schema violations.

```

#### Log Breakdown:

* **Rapid Fitting Phase (Steps 0 $\rightarrow$ 30):**
* Step 0: Initial loss is high ($0.6788$). Student starts with random weights ($0.0790$) vs Teacher ($1.5087$).
* Steps 15–30: Loss drops to $0.0028$. Student Q-value rapidly climbs ($0.0790 \rightarrow 1.1941 \rightarrow 1.4717$) matching Teacher predictions.


* **Dynamic State-Shift Spike:**
* Step 45: Ball is served; Teacher detects risk and flips Q-Value to $-0.7354$.
* Loss jumps to $0.4302$ as the Student sees the high-velocity ball image for the first time. This proves the Teacher's target distribution shift forced gradient updates in the Student.


* **Online Adaptation & Recovery (Steps 60 $\rightarrow$ 135):**
* Without a replay buffer, the Student adapts online, bringing loss back down from $0.4771$ to $0.0630$ by Step 135.



#### Phase 2 Conclusions:

By reducing parameter count by 86%, memory requirements for matrix multiplication drop significantly, leading to:

* Higher Frame Inference Speeds (**FPS**) on standard CPUs.
* **Minimal memory consumption** during containerized deployments.
* **Preserved spatial resolution** ($84 \times 84$ frame stacks) allowing the student to accurately track the ball and paddle.

---
### Phase 3: DeepMind-Style Visualizer and Hardware Profiler
  We proceed to build the visualization and benchmarking tools to watch both networks play the Atari Pong game side-by-side with Real-Time HUD overlays (FPS, Latency, RAM and Action outputs).

#### Approach for Phase 3:
  ->We build our visualizer to handle side-by-side frame stitching and HUD text overlays using OpenCV.

  -> We create a benchmark script to profile execution latecy, throughput (FPS), memory footprint and display the live visualizer window.

  -> Making sure to load and map the weights onto the nature_cnn_pong 

#### Phase 3 Observations

<video controls src="Screen Recording 2026-08-06 185211.mp4" title="Title"></video>
  ============================================================
---------- BENCHMARK EVALUATION SUMMARY ----------
============================================================
 -> RAM Footprint                   : ~348 MB
 -> Teacher Avg Latency             : ~1.30 ms - 1.33 ms (~750 FPS)
 -> Student Avg Latency             : ~0.49 ms - 0.59 ms (~1,700 - 2,000 FPS)
 -> Inference Speedup               : 2.21x - 2.69x Faster
============================================================

Key Engineering Feats achieved here include:
  - **Latency Reduction**: Frame inference time dropped from 1.33ms down to 0.49ms, enabling a throughput of nearly 2000FPS on CPU.
  - **Compact Memory Profile**: Total memory consumption remains lightweight at **~348MB**, making it suitable for edge deployment or containerized environment.

Observe the following Paddle Behavior Carefully:
   
There are three key reasons for the strange movement of the second paddle:

  - **Benchmark Execution loop Architecture:** In benchmark script, both visualizer panels show the exact same game env frame. The environment is driven strictly by the Teacher's action, t_act to ensure a fair latency and Q-Value evaluation under identical visual inputs. The Student is running passive inference side by side to compare its predicted Q-Values against Teacher in real-time.

  - **Untrained Student Checkpoint:** Looking closely at the HUD overlay, the Student outputs near-zero Q-Values (0.04) because *benchmark.py* currently instantiates a fresh *LightweightStudent* instance with uninitialized weights. The trained weights from train_distill were not saved to disk or loaded into benchmark.

  - **Built-In Atari Opponent Mechanics:** In Atari Pong, the **left paddle** is controlled by the built-in **ALE** (Arcade Learning Environment), while the **right paddle** is controlled by the RL model. This built-in bot is hardcoded with slight reaction delays and deliberate imperfections, causing to occasionally miss serves or hesitate.


## Future Plans and Features

* Building visualization and benchmarking tools to monitor both networks playing side-by-side with real-time HUD overlays (FPS, Latency, RAM, and Action outputs).

---

## Setup and Run Instructions

1. **Clone the Repository:**
```bash
git clone https://github.com/huggingface/OpenEnv.git

```


2. **Setup Virtual Environment:**
* **Windows:**
```cmd
python -m venv venv

```


* **macOS / Linux:**
```bash
python3.11 -m venv venv

```




3. **Activate Virtual Environment:**
* **Windows:**
```cmd
.\venv\Scripts\activate

```


* **macOS / Linux:**
```bash
source venv/bin/activate

```




4. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


5. **Start Server (Terminal 1):**
```bash
uv run --python 3.11 --project . server --port 8000

```


6. **Run Phase 0 Test (Terminal 2):**
* **Windows:**
```cmd
pytest .\tests\test_ipc_baseline.py --benchmark-sort=mean

```


* **macOS / Linux:**
```bash
pytest ./tests/test_ipc_baseline.py --benchmark-sort=mean

```




7. **Run Pipeline Parity Test:**
* **Windows:**
```cmd
$env:PYTHONPATH="src"; pytest .\tests\test_pipeline_parity.py -v -s

```


* **macOS / Linux:**
```bash
PYTHONPATH=src pytest ./tests/test_pipeline_parity.py -v -s

```




8. **Run Distillation Training Loop:**
* **Windows:**
```cmd
$env:PYTHONPATH="src"; python .\src\lightweight_rl\train_distill.py

```


* **macOS / Linux:**
```bash
PYTHONPATH=src python ./src/lightweight_rl/train_distill.py

```

9. **Execute the benchmarking script alongside the visualizer**:
* **Windows:**
```bash
$env:PYTHONPATH="src"; python .\scripts\benchmark.py

```


* **macOS/Linux**
```bash
PYTHONPATH=src python ./scripts/benchmark.py

```