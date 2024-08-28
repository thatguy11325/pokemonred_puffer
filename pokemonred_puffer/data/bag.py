from ctypes import LittleEndianStructure, Union, c_uint8

from pyboy import PyBoy

from pokemonred_puffer.data.items import Items


BAG_CAPACITY = 20
BAG_LENGTH_BYTES = 2 * BAG_CAPACITY


class BagItem(LittleEndianStructure):
    _pack_ = 1
    _fields = [("Item", c_uint8), ("Quantity", c_uint8)]


class Bag(Union):
    _pack_ = 1
    _fields = [("bag", BAG_CAPACITY * BagItem), ("asbytes", c_uint8 * BAG_LENGTH_BYTES)]

    def __init__(self, emu: PyBoy):
        _, self.wBagItems = self.pyboy.symbol_lookup("wBagItems")
        _, self.wNumBagItems = self.pyboy.symbol_lookup("wNumBagItems")
        self.numBagItems = emu.memory[self.wNumBagItems]
        self.as_bytes = (c_uint8 * BAG_LENGTH_BYTES)(
            *emu.memory[self.wBagItems : self.wBagItems + BAG_LENGTH_BYTES]
        )
        self.emu = emu

    def add(self, item: Items, quantity: int) -> bool:
        if self.numBagItems >= BAG_CAPACITY:
            return False
        try:
            idx = self.as_bytes[::2].index(item.value)
            self.as_bytes[idx + 1] += min(99, quantity + self.as_bytes[idx + 1])
        except ValueError:
            self.as_bytes[self.numBagItems] = item.value
            self.as_bytes[self.numBagItems + 1] = quantity
            self.numBagItems += 1
        self.emu.memory[self.wNumBagItems] = self.numBagItems
        self.emu.memory[self.wBagItems : self.wBagItems + BAG_LENGTH_BYTES] = self.as_bytes
        return True

    def remove(self, item: Items, quantity: int) -> bool:
        if self.numBagItems >= BAG_CAPACITY:
            return False
        try:
            idx = self.as_bytes[::2].index(item.value)
        except ValueError:
            return False
        new_quantity = max(0, self.as_bytes[idx + 1] - quantity)
        if new_quantity > 0:
            self.as_bytes[idx + 1] = new_quantity
        else:
            # Remove from bag
            self.as_bytes[idx:-2] = self.as_bytes[idx + 2 :]
            self.numBagItems -= 1
            self.as_bytes[2 * self.numBagItems :] = 0xFF
        self.emu.memory[self.wNumBagItems] = self.numBagItems
        self.emu.memory[self.wBagItems : self.wBagItems + BAG_LENGTH_BYTES] = self.as_bytes
        return True
