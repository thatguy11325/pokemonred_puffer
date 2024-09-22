from ctypes import LittleEndianStructure, Union, c_uint8

from pyboy import PyBoy


class FlagsBits(LittleEndianStructure):
    _fields_ = [
        # d728 - StatusFlags1
        ("BIT_STRENGTH_ACTIVE", c_uint8, 1),
        ("BIT_SURF_ALLOWED", c_uint8, 1),
        ("BIT_UNUSED_1", c_uint8, 1),
        ("BIT_GOT_OLD_ROD", c_uint8, 1),
        ("BIT_GOT_GOOD_ROD", c_uint8, 1),
        ("BIT_GOT_SUPER_ROD", c_uint8, 1),
        ("BIT_GAVE_SAFFRON_GUARDS_DRINK", c_uint8, 1),
        ("BIT_UNUSED_CARD_KEY", c_uint8, 1),
        # d729
        ("BYTE_UNUSED_1", c_uint8, 8),
        # d72a - wBeatGymFlags
        ("BIT_BOULDERBADGE", c_uint8, 1),
        ("BIT_CASCADEBADGE", c_uint8, 1),
        ("BIT_THUNDERBADGE", c_uint8, 1),
        ("BIT_RAINBOWBADGE", c_uint8, 1),
        ("BIT_SOULBADGE", c_uint8, 1),
        ("BIT_MARSHBADGE", c_uint8, 1),
        ("BIT_VOLCANOBADGE", c_uint8, 1),
        ("BIT_EARTHBADGE", c_uint8, 1),
        # d72b - unused_2
        ("BYTE_UNUSED_2", c_uint8, 8),
        # d72c - wStatusFlags2
        ("BIT_WILD_ENCOUNTER_COOLDOWN", c_uint8, 1),
        ("BIT_NO_AUDIO_FADE_OUT", c_uint8, 1),
        ("BIT_UNUSED_2", c_uint8, 1),
        ("BIT_UNUSED_3", c_uint8, 1),
        ("BIT_UNUSED_4", c_uint8, 1),
        ("BIT_UNUSED_5", c_uint8, 1),
        ("BIT_UNUSED_6", c_uint8, 1),
        ("BIT_UNUSED_7", c_uint8, 1),
        # d72d - wStatusFlags3
        ("BIT_INIT_TRADE_CENTER_FACING", c_uint8, 1),
        ("BIT_UNUSED_8", c_uint8, 1),
        ("BIT_UNUSED_9", c_uint8, 1),
        ("BIT_WARP_FROM_CUR_SCRIPT", c_uint8, 1),
        ("BIT_ON_DUNGEON_WARP", c_uint8, 1),
        ("BIT_NO_NPC_FACE_PLAYER", c_uint8, 1),
        ("BIT_TALKED_TO_TRAINER", c_uint8, 1),
        ("BIT_PRINT_END_BATTLE_TEXT", c_uint8, 1),
        # d72e - wStatusFlags4
        ("BIT_GOT_LAPRAS", c_uint8, 1),
        ("BIT_UNKNOWN_4_1", c_uint8, 1),
        ("BIT_USED_POKECENTER", c_uint8, 1),
        ("BIT_GOT_STARTER", c_uint8, 1),
        ("BIT_NO_BATTLES", c_uint8, 1),
        ("BIT_BATTLE_OVER_OR_BLACKOUT", c_uint8, 1),
        ("BIT_LINK_CONNECTED", c_uint8, 1),
        ("BIT_INIT_SCRIPTED_MOVEMENT", c_uint8, 1),
        # d73f - unused_3
        ("BYTE_UNUSED_3", c_uint8, 8),
        # d730 - wStatusFlags5
        ("BIT_SCRIPTED_NPC_MOVEMENT", c_uint8, 1),
        ("BIT_UNKNOWN_5_1", c_uint8, 1),
        ("BIT_UNKNOWN_5_2", c_uint8, 1),
        ("BIT_UNUSED_10", c_uint8, 1),
        ("BIT_UNKNOWN_5_4", c_uint8, 1),
        ("BIT_DISABLE_JOYPAD", c_uint8, 1),
        ("BIT_NO_TEXT_DELAY", c_uint8, 1),
        ("BIT_SCRIPTED_MOVEMENT_STATE", c_uint8, 1),
        # d731 - unused_4
        ("BYTE_UNUSED_4", c_uint8, 8),
        # d732 - wStatusFlags6
        ("BIT_GAME_TIMER_COUNTING", c_uint8, 1),
        ("BIT_DEBUG_MODE", c_uint8, 1),
        ("BIT_FLY_OR_DUNGEON_WARP", c_uint8, 1),
        ("BIT_FLY_WARP", c_uint8, 1),
        ("BIT_DUNGEON_WARP", c_uint8, 1),
        ("BIT_ALWAYS_ON_BIKE", c_uint8, 1),
        ("BIT_ESCAPE_WARP", c_uint8, 1),
        ("BIT_UNUSED_11", c_uint8, 1),
        # d733 - wStatusFlags7
        ("BIT_TEST_BATTLE", c_uint8, 1),
        ("BIT_NO_MAP_MUSIC", c_uint8, 1),
        ("BIT_FORCED_WARP", c_uint8, 1),
        ("BIT_TRAINER_BATTLE", c_uint8, 1),
        ("BIT_USE_CUR_MAP_SCRIPT", c_uint8, 1),
        ("BIT_UNUSED_12", c_uint8, 1),
        ("BIT_UNUSED_13", c_uint8, 1),
        ("BIT_USED_FLY", c_uint8, 1),
        # d734 - wElite4Flags
        ("BIT_UNUSED_BEAT_ELITE_4", c_uint8, 1),
        ("BIT_STARTED_ELITE_4", c_uint8, 1),
        ("BIT_UNUSED_14", c_uint8, 1),
        ("BIT_UNUSED_15", c_uint8, 1),
        ("BIT_UNUSED_16", c_uint8, 1),
        ("BIT_UNUSED_17", c_uint8, 1),
        ("BIT_UNUSED_18", c_uint8, 1),
        ("BIT_UNUSED_19", c_uint8, 1),
    ]


class Flags(Union):
    _fields_ = [("b", FlagsBits), ("asbytes", c_uint8 * 13)]

    def __init__(self, emu: PyBoy):
        super().__init__()
        self.emu = emu
        self.asbytes = (c_uint8 * 13)(
            *emu.memory[
                emu.symbol_lookup("wStatusFlags1")[1] : (emu.symbol_lookup("wElite4Flags")[1] + 1)
            ]
        )

    def get_bit(self, name: str) -> bool:
        return bool(getattr(self.b, name))

    def set_bit(self, name: str, value: bool):
        # This is O(N) but it's so rare that I'm not too worried about it
        idx = [x[0] for x in self.b._fields_].index(name)
        addr = self.emu.symbol_lookup("wStatusFlags1")[1] + idx // 8
        bit = idx % 8
        mask = int(value) << bit

        self.emu.memory[addr] = (self.emu.memory[addr] & ~mask) | mask
        setattr(self.b, name, int(value))
