from pyvirtualdisplay import Display
import gymnasium as gym
from google.colab import files

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env = gym.make("LunarLander-v2")
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())

# Create the environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")

model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

# Train it for 1,000,000 timesteps
model.learn(total_timesteps=1000000)
# Specify file name for model and save the model to file
model_name = "ppo-LunarLander-v2"
model.save(model_name)

env_id = "LunarLander-v2"
video_folder = "./"
video_length = 500

# TODO: Evaluate the agent
# Create a new environment for evaluation
eval_env = Monitor(gym.make("LunarLander-v2"))
model = PPO.load(model_name, env=env)
# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

# Print the results
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_id}")

vec_env.reset()
for _ in range(video_length + 1):
  action = [vec_env.action_space.sample()]
  obs, _, _, _ = vec_env.step(action)
# Save the video
vec_env.close()

# Download the video
files.download('/content/random-agent-LunarLander-v2-step-0-to-step-500.mp4')

