from enum import Enum


class MapIds(Enum):
    PALLET_TOWN = 0x00
    VIRIDIAN_CITY = 0x01
    PEWTER_CITY = 0x02
    CERULEAN_CITY = 0x03
    LAVENDER_TOWN = 0x04
    VERMILION_CITY = 0x05
    CELADON_CITY = 0x06
    FUCHSIA_CITY = 0x07
    CINNABAR_ISLAND = 0x08
    INDIGO_PLATEAU = 0x09
    SAFFRON_CITY = 0x0A
    # NUM_CITY_MAPS EQU const_value
    UNUSED_MAP_0B = 0x0B
    # FIRST_ROUTE_MAP EQU const_value
    ROUTE_1 = 0x0C
    ROUTE_2 = 0x0D
    ROUTE_3 = 0x0E
    ROUTE_4 = 0x0F
    ROUTE_5 = 0x10
    ROUTE_6 = 0x11
    ROUTE_7 = 0x12
    ROUTE_8 = 0x13
    ROUTE_9 = 0x14
    ROUTE_10 = 0x15
    ROUTE_11 = 0x16
    ROUTE_12 = 0x17
    ROUTE_13 = 0x18
    ROUTE_14 = 0x19
    ROUTE_15 = 0x1A
    ROUTE_16 = 0x1B
    ROUTE_17 = 0x1C
    ROUTE_18 = 0x1D
    ROUTE_19 = 0x1E
    ROUTE_20 = 0x1F
    ROUTE_21 = 0x20
    ROUTE_22 = 0x21
    ROUTE_23 = 0x22
    ROUTE_24 = 0x23
    ROUTE_25 = 0x24
    # FIRST_INDOOR_MAP EQU const_value
    REDS_HOUSE_1F = 0x25
    REDS_HOUSE_2F = 0x26
    BLUES_HOUSE = 0x27
    OAKS_LAB = 0x28
    VIRIDIAN_POKECENTER = 0x29
    VIRIDIAN_MART = 0x2A
    VIRIDIAN_SCHOOL_HOUSE = 0x2B
    VIRIDIAN_NICKNAME_HOUSE = 0x2C
    VIRIDIAN_GYM = 0x2D
    DIGLETTS_CAVE_ROUTE_2 = 0x2E
    VIRIDIAN_FOREST_NORTH_GATE = 0x2F
    ROUTE_2_TRADE_HOUSE = 0x30
    ROUTE_2_GATE = 0x31
    VIRIDIAN_FOREST_SOUTH_GATE = 0x32
    VIRIDIAN_FOREST = 0x33
    MUSEUM_1F = 0x34
    MUSEUM_2F = 0x35
    PEWTER_GYM = 0x36
    PEWTER_NIDORAN_HOUSE = 0x37
    PEWTER_MART = 0x38
    PEWTER_SPEECH_HOUSE = 0x39
    PEWTER_POKECENTER = 0x3A
    MT_MOON_1F = 0x3B
    MT_MOON_B1F = 0x3C
    MT_MOON_B2F = 0x3D
    CERULEAN_TRASHED_HOUSE = 0x3E
    CERULEAN_TRADE_HOUSE = 0x3F
    CERULEAN_POKECENTER = 0x40
    CERULEAN_GYM = 0x41
    BIKE_SHOP = 0x42
    CERULEAN_MART = 0x43
    MT_MOON_POKECENTER = 0x44
    CERULEAN_TRASHED_HOUSE_COPY = 0x45
    ROUTE_5_GATE = 0x46
    UNDERGROUND_PATH_ROUTE_5 = 0x47
    DAYCARE = 0x48
    ROUTE_6_GATE = 0x49
    UNDERGROUND_PATH_ROUTE_6 = 0x4A
    UNDERGROUND_PATH_ROUTE_6_COPY = 0x4B
    ROUTE_7_GATE = 0x4C
    UNDERGROUND_PATH_ROUTE_7 = 0x4D
    UNDERGROUND_PATH_ROUTE_7_COPY = 0x4E
    ROUTE_8_GATE = 0x4F
    UNDERGROUND_PATH_ROUTE_8 = 0x50
    ROCK_TUNNEL_POKECENTER = 0x51
    ROCK_TUNNEL_1F = 0x52
    POWER_PLANT = 0x53
    ROUTE_11_GATE_1F = 0x54
    DIGLETTS_CAVE_ROUTE_11 = 0x55
    ROUTE_11_GATE_2F = 0x56
    ROUTE_12_GATE_1F = 0x57
    BILLS_HOUSE = 0x58
    VERMILION_POKECENTER = 0x59
    POKEMON_FAN_CLUB = 0x5A
    VERMILION_MART = 0x5B
    VERMILION_GYM = 0x5C
    VERMILION_PIDGEY_HOUSE = 0x5D
    VERMILION_DOCK = 0x5E
    SS_ANNE_1F = 0x5F
    SS_ANNE_2F = 0x60
    SS_ANNE_3F = 0x61
    SS_ANNE_B1F = 0x62
    SS_ANNE_BOW = 0x63
    SS_ANNE_KITCHEN = 0x64
    SS_ANNE_CAPTAINS_ROOM = 0x65
    SS_ANNE_1F_ROOMS = 0x66
    SS_ANNE_2F_ROOMS = 0x67
    SS_ANNE_B1F_ROOMS = 0x68
    UNUSED_MAP_69 = 0x69
    UNUSED_MAP_6A = 0x6A
    UNUSED_MAP_6B = 0x6B
    VICTORY_ROAD_1F = 0x6C
    UNUSED_MAP_6D = 0x6D
    UNUSED_MAP_6E = 0x6E
    UNUSED_MAP_6F = 0x6F
    UNUSED_MAP_70 = 0x70
    LANCES_ROOM = 0x71
    UNUSED_MAP_72 = 0x72
    UNUSED_MAP_73 = 0x73
    UNUSED_MAP_74 = 0x74
    UNUSED_MAP_75 = 0x75
    HALL_OF_FAME = 0x76
    UNDERGROUND_PATH_NORTH_SOUTH = 0x77
    CHAMPIONS_ROOM = 0x78
    UNDERGROUND_PATH_WEST_EAST = 0x79
    CELADON_MART_1F = 0x7A
    CELADON_MART_2F = 0x7B
    CELADON_MART_3F = 0x7C
    CELADON_MART_4F = 0x7D
    CELADON_MART_ROOF = 0x7E
    CELADON_MART_ELEVATOR = 0x7F
    CELADON_MANSION_1F = 0x80
    CELADON_MANSION_2F = 0x81
    CELADON_MANSION_3F = 0x82
    CELADON_MANSION_ROOF = 0x83
    CELADON_MANSION_ROOF_HOUSE = 0x84
    CELADON_POKECENTER = 0x85
    CELADON_GYM = 0x86
    GAME_CORNER = 0x87
    CELADON_MART_5F = 0x88
    GAME_CORNER_PRIZE_ROOM = 0x89
    CELADON_DINER = 0x8A
    CELADON_CHIEF_HOUSE = 0x8B
    CELADON_HOTEL = 0x8C
    LAVENDER_POKECENTER = 0x8D
    POKEMON_TOWER_1F = 0x8E
    POKEMON_TOWER_2F = 0x8F
    POKEMON_TOWER_3F = 0x90
    POKEMON_TOWER_4F = 0x91
    POKEMON_TOWER_5F = 0x92
    POKEMON_TOWER_6F = 0x93
    POKEMON_TOWER_7F = 0x94
    MR_FUJIS_HOUSE = 0x95
    LAVENDER_MART = 0x96
    LAVENDER_CUBONE_HOUSE = 0x97
    FUCHSIA_MART = 0x98
    FUCHSIA_BILLS_GRANDPAS_HOUSE = 0x99
    FUCHSIA_POKECENTER = 0x9A
    WARDENS_HOUSE = 0x9B
    SAFARI_ZONE_GATE = 0x9C
    FUCHSIA_GYM = 0x9D
    FUCHSIA_MEETING_ROOM = 0x9E
    SEAFOAM_ISLANDS_B1F = 0x9F
    SEAFOAM_ISLANDS_B2F = 0xA0
    SEAFOAM_ISLANDS_B3F = 0xA1
    SEAFOAM_ISLANDS_B4F = 0xA2
    VERMILION_OLD_ROD_HOUSE = 0xA3
    FUCHSIA_GOOD_ROD_HOUSE = 0xA4
    POKEMON_MANSION_1F = 0xA5
    CINNABAR_GYM = 0xA6
    CINNABAR_LAB = 0xA7
    CINNABAR_LAB_TRADE_ROOM = 0xA8
    CINNABAR_LAB_METRONOME_ROOM = 0xA9
    CINNABAR_LAB_FOSSIL_ROOM = 0xAA
    CINNABAR_POKECENTER = 0xAB
    CINNABAR_MART = 0xAC
    CINNABAR_MART_COPY = 0xAD
    INDIGO_PLATEAU_LOBBY = 0xAE
    COPYCATS_HOUSE_1F = 0xAF
    COPYCATS_HOUSE_2F = 0xB0
    FIGHTING_DOJO = 0xB1
    SAFFRON_GYM = 0xB2
    SAFFRON_PIDGEY_HOUSE = 0xB3
    SAFFRON_MART = 0xB4
    SILPH_CO_1F = 0xB5
    SAFFRON_POKECENTER = 0xB6
    MR_PSYCHICS_HOUSE = 0xB7
    ROUTE_15_GATE_1F = 0xB8
    ROUTE_15_GATE_2F = 0xB9
    ROUTE_16_GATE_1F = 0xBA
    ROUTE_16_GATE_2F = 0xBB
    ROUTE_16_FLY_HOUSE = 0xBC
    ROUTE_12_SUPER_ROD_HOUSE = 0xBD
    ROUTE_18_GATE_1F = 0xBE
    ROUTE_18_GATE_2F = 0xBF
    SEAFOAM_ISLANDS_1F = 0xC0
    ROUTE_22_GATE = 0xC1
    VICTORY_ROAD_2F = 0xC2
    ROUTE_12_GATE_2F = 0xC3
    VERMILION_TRADE_HOUSE = 0xC4
    DIGLETTS_CAVE = 0xC5
    VICTORY_ROAD_3F = 0xC6
    ROCKET_HIDEOUT_B1F = 0xC7
    ROCKET_HIDEOUT_B2F = 0xC8
    ROCKET_HIDEOUT_B3F = 0xC9
    ROCKET_HIDEOUT_B4F = 0xCA
    ROCKET_HIDEOUT_ELEVATOR = 0xCB
    UNUSED_MAP_CC = 0xCC
    UNUSED_MAP_CD = 0xCD
    UNUSED_MAP_CE = 0xCE
    SILPH_CO_2F = 0xCF
    SILPH_CO_3F = 0xD0
    SILPH_CO_4F = 0xD1
    SILPH_CO_5F = 0xD2
    SILPH_CO_6F = 0xD3
    SILPH_CO_7F = 0xD4
    SILPH_CO_8F = 0xD5
    POKEMON_MANSION_2F = 0xD6
    POKEMON_MANSION_3F = 0xD7
    POKEMON_MANSION_B1F = 0xD8
    SAFARI_ZONE_EAST = 0xD9
    SAFARI_ZONE_NORTH = 0xDA
    SAFARI_ZONE_WEST = 0xDB
    SAFARI_ZONE_CENTER = 0xDC
    SAFARI_ZONE_CENTER_REST_HOUSE = 0xDD
    SAFARI_ZONE_SECRET_HOUSE = 0xDE
    SAFARI_ZONE_WEST_REST_HOUSE = 0xDF
    SAFARI_ZONE_EAST_REST_HOUSE = 0xE0
    SAFARI_ZONE_NORTH_REST_HOUSE = 0xE1
    CERULEAN_CAVE_2F = 0xE2
    CERULEAN_CAVE_B1F = 0xE3
    CERULEAN_CAVE_1F = 0xE4
    NAME_RATERS_HOUSE = 0xE5
    CERULEAN_BADGE_HOUSE = 0xE6
    UNUSED_MAP_E7 = 0xE7
    ROCK_TUNNEL_B1F = 0xE8
    SILPH_CO_9F = 0xE9
    SILPH_CO_10F = 0xEA
    SILPH_CO_11F = 0xEB
    SILPH_CO_ELEVATOR = 0xEC
    UNUSED_MAP_ED = 0xED
    UNUSED_MAP_EE = 0xEE
    TRADE_CENTER = 0xEF
    COLOSSEUM = 0xF0
    UNUSED_MAP_F1 = 0xF1
    UNUSED_MAP_F2 = 0xF2
    UNUSED_MAP_F3 = 0xF3
    UNUSED_MAP_F4 = 0xF4
    LORELEIS_ROOM = 0xF5
    BRUNOS_ROOM = 0xF6
    AGATHAS_ROOM = 0xF7


RESET_MAP_IDS = {
    0x0,  # Pallet Town
    0x1,  # Viridian City
    0x2,  # Pewter City
    0x3,  # Cerulean City
    0x4,  # Lavender Town
    0x5,  # Vermilion City
    0x6,  # Celadon City
    0x7,  # Fuchsia City
    0x8,  # Cinnabar Island
    0x9,  # Indigo Plateau
    0xA,  # Saffron City
    0xF,  # Route 4 (Mt Moon)
    0x10,  # Route 10 (Rock Tunnel)
    0xE9,  # Silph Co 9F (Heal station)
}

MAP_ID_COMPLETION_EVENTS = {
    MapIds.PEWTER_GYM: ["EVENT_BEAT_BROCK"],
    MapIds.CERULEAN_GYM: ["EVENT_BEAT_MISTY"],
    MapIds.VERMILION_GYM: ["EVENT_BEAT_LT_SURGE"],
    MapIds.CELADON_GYM: ["EVENT_BEAT_ERIKA"],
    MapIds.SAFFRON_GYM: ["EVENT_BEAT_SABRINA"],
    MapIds.FUCHSIA_GYM: ["EVENT_BEAT_KOGA"],
    MapIds.CINNABAR_GYM: ["EVENT_BEAT_BLAINE"],
    MapIds.VIRIDIAN_GYM: ["EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI"],
    MapIds.GAME_CORNER: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.ROCKET_HIDEOUT_B1F: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.ROCKET_HIDEOUT_B2F: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.ROCKET_HIDEOUT_B3F: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.ROCKET_HIDEOUT_B4F: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.ROCKET_HIDEOUT_ELEVATOR: ["EVENT_BEAT_ROCKET_HIDEOUT_GIOVANNI"],
    MapIds.SILPH_CO_1F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_2F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_3F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_4F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_5F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_6F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_7F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_8F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_9F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_10F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_11F: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.SILPH_CO_ELEVATOR: ["EVENT_BEAT_SILPH_CO_GIOVANNI"],
    MapIds.POKEMON_MANSION_1F: ["HS_POKEMON_MANSION_B1F_ITEM_5"],
    MapIds.POKEMON_MANSION_2F: ["HS_POKEMON_MANSION_B1F_ITEM_5"],
    MapIds.POKEMON_MANSION_3F: ["HS_POKEMON_MANSION_B1F_ITEM_5"],
    MapIds.POKEMON_MANSION_B1F: ["HS_POKEMON_MANSION_B1F_ITEM_5"],
    MapIds.POKEMON_TOWER_1F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_2F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_3F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_4F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_5F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_6F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.POKEMON_TOWER_7F: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.MR_FUJIS_HOUSE: ["EVENT_GOT_POKE_FLUTE"],
    MapIds.SAFARI_ZONE_CENTER: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_CENTER_REST_HOUSE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_EAST: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_EAST_REST_HOUSE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_GATE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_NORTH: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_NORTH_REST_HOUSE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_SECRET_HOUSE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_WEST: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
    MapIds.SAFARI_ZONE_WEST_REST_HOUSE: ["EVENT_GOT_HM03", "HS_SAFARI_ZONE_WEST_ITEM_4"],
}
