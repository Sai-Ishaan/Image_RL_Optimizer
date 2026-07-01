##PyTest for IPC Baseline(Phase 0 benchmarking)
import os
import numpy as np
import pytest
from openenv.core.generic_client import GenericEnvClient
#from atari_env import AtariEnvConfig

### MONKEY-PATCH: Revealing the hidden Pydantic Validation Schema
og_send_and_receive = GenericEnvClient._send_and_receive

os.environ["ATARI_OBS_TYPE"] = "grayscale"
os.environ["ATARI_FRAMESKIP"] = "4"

async def debug_send_and_receive(self, message):
    await self._send(message)
    response = await self._receive()

    if response.get("type") == "error":
        error_data = response.get("data", {})
        details = error_data.get("details", error_data)
        raise RuntimeError(f"Server Error: {error_data.get('message',' Unknown Error')}"
                        f"(Code: {error_data.get('code', 'UNKNOWN')})\n"
                        f"Schema Details: {details}"
        )
    return response
GenericEnvClient._send_and_receive = debug_send_and_receive

# class DiscreteAction: ## Tiny duck-tyed container to bypass client validation and ship raw integer payloads
#     def __init__(self, value: int):
#         self.value = value

#     def model_dump(self):
#         return self.value    

@pytest.fixture(scope="module")
def env_client():
    #config = {"game": "PongNoFrameskip-v4"}
    client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")
    with client.sync() as sync_client:
        yield sync_client

def test_environment_connection(env_client):
    ## Sanity check to ensure connection and reset works
    config = {"game": "PongNoFrameskip-v4"}
    result = env_client.reset(config=config)

    assert result is not None, "Failed to receive initial state from server"
    
    obs = result.observation ## Extract raw observation data from JSON Wrapper
    if isinstance(obs, dict):
        key = list(obs.keys())[0] #Extract the 1st valid data key available in dict
        obs = obs[key] ##Extract the actual observation data
    
    obs_matrix = np.array(obs)
    if obs_matrix.shape == (100800,):
        obs_matrix = obs_matrix.reshape(210,160,3)

    assert len(obs_matrix.shape) >=2, f"Failed to parse array structure. Shape is: {obs_matrix.shape}"
    print(f"\n Captured Action Frame Shape: {obs_matrix.shape}")

def test_random_agent_ipc_latency(benchmark, env_client):
    ##Measures minimum time reqd for a single env step over the OpenEnv loopback interface
    config = {"game": "PongNoFrameskip-v4"}
    env_client.reset(config=config)
    ## Probing standard Gym, Array and Agentic tool schemas
    candidates = [
        {"action":1},
        {"action":[1]},
        {"tool_name": "step", "arguments":{"action":1}},
        {"command":"step", "params":{"action":1}},
        {"type":"step", "action":1},
    ]
    valid_payload = {"action_id": 1}
    last_error = None
    print("\n Probing server for valid action scheme:")
    # for payload in candidates:
    #     try:
    #         env_client.step(payload)
    #         valid_payload = payload
    #         if isinstance(valid_payload, DiscreteAction):
    #             print(f" Valid payload found(Raw Scalae) via DiscreteAction: {payload.value}")
    #         else: 
    #             print(f" Valid payload found: {valid_payload}")
    #         break
    #     except Exception as e:
    #         last_error = e
    #         print(f" Payload {payload} failed with error: {e}")
    #         continue

    # if not valid_payload:
    #     print(f"\n Fatal! All candidate schemes rejected. \n Error trace: {last_error}")
    #     pytest.fail("No valid action payload scheme found.")
    def step_env():
        # action_payload = DiscreteAction(1)
        return env_client.step(valid_payload)
    ##Benchmark will run the function thousands of times and output min/max/mean latency
    benchmark(step_env)