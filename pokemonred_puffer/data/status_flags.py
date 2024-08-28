from ctypes import LittleEndianStructure, Union, c_uint8

from pyboy import PyBoy


class StatusFlags1Bits(LittleEndianStructure):
    _fields_ = [
        ("USING_STRENGTH_OUTSIDE_OF_BATTLE", c_uint8, 1),
        ("IS_SURFING_ALLOWED", c_uint8, 1),
        ("UNUSED_0", c_uint8, 1),
        ("RECEIVED_OLD_ROD", c_uint8, 1),
        ("RECEIVED_GOOD_ROD", c_uint8, 1),
        ("RECEIVED_SUPER_ROD", c_uint8, 1),
        ("GAVE_SAFFRON_GUARD_DRINK", c_uint8, 1),
        ("UNUSED_2", c_uint8, 1),
    ]


class StatusFlags1(Union):
    _fields_ = [("b", StatusFlags1Bits), ("asbytes", c_uint8)]

    def __init__(self, emu: PyBoy):
        super().__init__()
        self.asbytes = (c_uint8)(emu.memory[emu.symbol_lookup("wStatusFlags1")[1]])

    def get_bit(self, bit: str) -> bool:
        return bool(getattr(self.b, bit))
