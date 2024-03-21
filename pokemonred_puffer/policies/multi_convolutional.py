import torch
from torch import nn

import pufferlib.models
from pufferlib.emulation import unpack_batched_obs

unpack_batched_obs = torch.compiler.disable(unpack_batched_obs)

# Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0


def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class RecurrentMultiConvolutionalWrapper(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class MultiConvolutionalPolicy(pufferlib.models.Policy):
    def __init__(
        self,
        env,
        screen_framestack: int = 3,
        global_map_frame_stack: int = 1,
        screen_flat_size: int = 2433,  # 14341,
        global_map_flat_size: int = 1600,
        input_size: int = 512,
        framestack: int = 1,
        flat_size: int = 1,
        hidden_size=512,
        output_size=512,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(screen_framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        """
        self.global_map_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(global_map_frame_stack, 32, 16, stride=8)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        """

        self.encode_linear = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(
                    screen_flat_size,
                    hidden_size,
                ),
            ),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def encode_observations(self, observations):
        observations = unpack_batched_obs(observations, self.unflatten_context)

        output = []
        for okey, network in zip(
            ("screen",),
            (self.screen_network,),
        ):
            observation = observations[okey]
            if self.channels_last:
                observation = observation.permute(0, 3, 1, 2)
            if self.downsample > 1:
                observation = observation[:, :, :: self.downsample, :: self.downsample]
            output.append(network(observation.float() / 255.0).squeeze(1))
        return self.encode_linear(
            torch.cat(
                (
                    *output,
                    one_hot(observations["direction"].long(), 4).float().squeeze(1),
                    # one_hot(observations["reset_map_id"].long(), 0xF7).float().squeeze(1),
                    one_hot(observations["battle_type"].long(), 4).float().squeeze(1),
                    # observations["cut_in_party"].float(),
                    # observations["x"].float(),
                    # observations["y"].float(),
                    # one_hot(observations["map_id"].long(), 0xF7).float().squeeze(1),
                    # one_hot(observations["badges"].long(), 8).float().squeeze(1),
                ),
                dim=-1,
            )
        ), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
