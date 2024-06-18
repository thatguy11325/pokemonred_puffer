import torch
from torch import nn

import pufferlib.models
from pufferlib.emulation import unpack_batched_obs

from pokemonred_puffer.data.items import Items
from pokemonred_puffer.environment import PIXEL_VALUES

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
        hidden_size=512,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__(env)
        self.num_actions = self.action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.global_map_network = nn.Sequential(
            nn.LazyConv2d(32, 8, stride=4),
            nn.ReLU(),
            nn.LazyConv2d(64, 4, stride=2),
            nn.ReLU(),
            nn.LazyConv2d(64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.encode_linear = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.LazyLinear(self.num_actions)
        self.value_fn = nn.LazyLinear(1)

        self.two_bit = env.unwrapped.env.two_bit
        self.use_global_map = env.unwrapped.env.use_global_map

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
        )
        self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)

        # pokemon has 0xF7 map ids
        # Lets start with 4 dims for now. Could try 8
        self.map_embeddings = torch.nn.Embedding(0xF7, 4, dtype=torch.float32)
        # N.B. This is an overestimate
        item_count = max(Items._value2member_map_.keys())
        self.item_embeddings = torch.nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )

    def encode_observations(self, observations):
        observations = unpack_batched_obs(observations, self.unflatten_context)

        screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        if self.use_global_map:
            global_map = observations["global_map"]
            restored_global_map_shape = (
                global_map.shape[0],
                global_map.shape[1],
                global_map.shape[2] * 4,
                global_map.shape[3],
            )

        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
            if self.use_global_map:
                global_map = torch.index_select(
                    self.linear_buckets,
                    0,
                    ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                    .flatten()
                    .int(),
                ).reshape(restored_global_map_shape)
        badges = self.badge_buffer <= observations["badges"]
        map_id = self.map_embeddings(observations["map_id"].long())
        blackout_map_id = self.map_embeddings(observations["blackout_map_id"].long())
        # The bag quantity can be a value between 1 and 99
        # TODO: Should items be positionally encoded? I dont think it matters
        items = self.item_embeddings(observations["bag_items"].squeeze(1).long()).float() * (
            observations["bag_quantity"].squeeze(1).float().unsqueeze(-1) / 100.0
        )

        # image_observation = torch.cat((screen, visited_mask, global_map), dim=-1)
        image_observation = torch.cat((screen, visited_mask), dim=-1)
        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
            if self.use_global_map:
                global_map = global_map.permute(0, 3, 1, 2)
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        cat_obs = torch.cat(
            (
                self.screen_network(image_observation.float() / 255.0).squeeze(1),
                one_hot(observations["direction"].long(), 4).float().squeeze(1),
                # one_hot(observations["reset_map_id"].long(), 0xF7).float().squeeze(1),
                one_hot(observations["battle_type"].long(), 4).float().squeeze(1),
                observations["cut_event"].float(),
                observations["cut_in_party"].float(),
                # observations["x"].float(),
                # observations["y"].float(),
                # one_hot(observations["map_id"].long(), 0xF7).float().squeeze(1),
                badges.float().squeeze(1),
                map_id.squeeze(1),
                blackout_map_id.squeeze(1),
                observations["wJoyIgnore"].float(),
                items.flatten(start_dim=1),
            ),
            dim=-1,
        )
        if self.use_global_map:
            cat_obs = torch.cat(
                (
                    cat_obs,
                    self.global_map_network(global_map.float() / 255.0).squeeze(1),
                ),
                dim=-1,
            )
        return self.encode_linear(cat_obs), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
