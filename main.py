from src.agents.CEM import CEM
from src.agents.RENYI import RENYI

# obj = CEM(env_id="Pendulum-v1", alpha=0, zeta=1, epoch_nr=3, out_path="results/Pendulum/CEM")
# obj.train()

# obj = CEM(env_id="CartPole-v1", alpha=0, zeta=1, epoch_nr=3, out_path="results/CartPole/CEM")
# obj.train()

obj = CEM(env_id="MountainCarContinuous-v0", alpha=0, zeta=1, epoch_nr=300, out_path="results/MountainCarContinuous/CEM")
obj.train()

# obj = RENYI(env_id="SafetyPointGoal1-v0", alpha=0, zeta=1, epoch_nr=3, out_path="results/SafetyPointGoal1/CEM")
# obj.train()

# obj = RENYI(env_id="Pendulum-v1", alpha=0, zeta=1, epoch_nr=3, out_path="results/Pendulum/RENYI")
# obj.train()

# obj = RENYI(env_id="CartPole-v1", alpha=0, zeta=1, epoch_nr=3, out_path="results/CartPole/RENYI")
# obj.train()

# obj = RENYI(env_id="MountainCarContinuous-v0", alpha=0, zeta=1, epoch_nr=3, out_path="results/MountainCarContinuous/RENYI")
# obj.train()

# obj = RENYI(env_id="SafetyPointGoal1-v0", alpha=0, zeta=1, epoch_nr=3, out_path="results/SafetyPointGoal1/RENYI")
# obj.train()
