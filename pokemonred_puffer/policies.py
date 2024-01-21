import pufferlib.models
import torch
from pokemonred_puffer.resnet import BasicBlock, ResNet 


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(pufferlib.models.Policy):
    def __init__(
        self,
        env,
        *args,
        framestack,
        flat_size,
        input_size=512,
        hidden_size=512,
        output_size=512,
        channels_last=True,
        downsample=1,
        **kwargs,
    ):
        """The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without."""
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample

        self.network = torch.nn.Sequential(
            ResNet(BasicBlock, [3, 4, 6, 3], input_channels=2, num_classes=1000),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(output_size),
            torch.nn.ReLU()
        )

        self.actor = pufferlib.pytorch.layer_init(
            torch.nn.Linear(output_size, self.num_actions), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(torch.nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, :: self.downsample, :: self.downsample]
        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
