from src.models.diffusionhead import DiffusionPolicyAction
import torch

model = DiffusionPolicyAction()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action = torch.randn(2, 8, 3)
context = torch.randn(2, 512)
# timesteps = torch.tensor([[1], [2]]).long()
timesteps = torch.tensor([1, 5]).long()
# print(timesteps.shape)
# print(model.add_noise(action, timesteps))

# print(action.shape)
# timesteps = torch.tensor([10]).long()

# print(model(action, timesteps, context))

with torch.no_grad():
        generated_actions = model.sample(context, device)

print(generated_actions)