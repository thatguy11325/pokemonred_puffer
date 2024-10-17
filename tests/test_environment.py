from typing import Generator
from unittest.mock import Mock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

import pokemonred_puffer.environment
from pokemonred_puffer.environment import RedGymEnv


@pytest.fixture()
def environment_fixture() -> Generator[RedGymEnv, None, None]:
    with patch.object(pokemonred_puffer.environment, "PyBoy", autospec=True) as pyboy_mock:
        pyboy_mock.return_value.symbol_lookup.return_value = (1, 2)
        env_config: DictConfig = OmegaConf.load("config.yaml").env
        env_config.gb_path = ""
        yield RedGymEnv(env_config=env_config)


@pytest.mark.parametrize(
    "max_opponent_level,party_count,levels,expected",
    (
        (0, 0, [], 0),
        (0, 0, [1], 0),
        (1, 0, [0], 1),
        (1, 1, [3], 3),
        (1, 2, [3, 4], 4),
        (1, 2, [3, 4, 5], 4),
    ),
)
def test_update_max_op_level(
    environment_fixture: RedGymEnv,
    max_opponent_level: int,
    party_count: int,
    levels: list[int],
    expected: int,
):
    environment_fixture.read_m = Mock()
    environment_fixture.read_m.side_effect = [party_count] + levels
    environment_fixture.max_opponent_level = max_opponent_level
    assert environment_fixture.update_max_op_level() == expected
