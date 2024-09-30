from enum import Enum
from pokemonred_puffer.data.species import Species


class TmHmMoves(Enum):
    MEGA_PUNCH = 0x5
    RAZOR_WIND = 0xD
    SWORDS_DANCE = 0xE
    WHIRLWIND = 0x12
    MEGA_KICK = 0x19
    TOXIC = 0x5C
    HORN_DRILL = 0x20
    BODY_SLAM = 0x22
    TAKE_DOWN = 0x24
    DOUBLE_EDGE = 0x26
    BUBBLE_BEAM = 0x3D
    WATER_GUN = 0x37
    ICE_BEAM = 0x3A
    BLIZZARD = 0x3B
    HYPER_BEAM = 0x3F
    PAY_DAY = 0x06
    SUBMISSION = 0x42
    COUNTER = 0x44
    SEISMIC_TOSS = 0x45
    RAGE = 0x63
    MEGA_DRAIN = 0x48
    SOLAR_BEAM = 0x4C
    DRAGON_RAGE = 0x52
    THUNDERBOLT = 0x55
    THUNDER = 0x57
    EARTHQUAKE = 0x59
    FISSURE = 0x5A
    DIG = 0x5B
    PSYCHIC = 0x5E
    TELEPORT = 0x64
    MIMIC = 0x66
    DOUBLE_TEAM = 0x68
    REFLECT = 0x73
    BIDE = 0x75
    METRONOME = 0x76
    SELFDESTRUCT = 0x78
    EGG_BOMB = 0x79
    FIRE_BLAST = 0x7E
    SWIFT = 0x81
    SKULL_BASH = 0x82
    SOFTBOILED = 0x87
    DREAM_EATER = 0x8A
    SKY_ATTACK = 0x8F
    REST = 0x9C
    THUNDER_WAVE = 0x56
    PSYWAVE = 0x95
    EXPLOSION = 0x99
    ROCK_SLIDE = 0x9D
    TRI_ATTACK = 0xA1
    SUBSTITUTE = 0xA4
    CUT = 0x0F
    FLY = 0x13
    SURF = 0x39
    STRENGTH = 0x46
    FLASH = 0x94


CUT_SPECIES_IDS = {
    Species.BULBASAUR.value,
    Species.IVYSAUR.value,
    Species.VENUSAUR.value,
    Species.CHARMANDER.value,
    Species.CHARMELEON.value,
    Species.CHARIZARD.value,
    Species.BEEDRILL.value,
    Species.SANDSHREW.value,
    Species.SANDSLASH.value,
    Species.ODDISH.value,
    Species.GLOOM.value,
    Species.VILEPLUME.value,
    Species.PARAS.value,
    Species.PARASECT.value,
    Species.BELLSPROUT.value,
    Species.WEEPINBELL.value,
    Species.VICTREEBEL.value,
    Species.TENTACOOL.value,
    Species.TENTACRUEL.value,
    Species.FARFETCHD.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.LICKITUNG.value,
    Species.TANGELA.value,
    Species.SCYTHER.value,
    Species.PINSIR.value,
    Species.MEW.value,
}

SURF_SPECIES_IDS = {
    Species.SQUIRTLE.value,
    Species.WARTORTLE.value,
    Species.BLASTOISE.value,
    Species.NIDOQUEEN.value,
    Species.NIDOKING.value,
    Species.PSYDUCK.value,
    Species.GOLDUCK.value,
    Species.POLIWAG.value,
    Species.POLIWHIRL.value,
    Species.POLIWRATH.value,
    Species.TENTACOOL.value,
    Species.TENTACRUEL.value,
    Species.SLOWPOKE.value,
    Species.SLOWBRO.value,
    Species.SEEL.value,
    Species.DEWGONG.value,
    Species.SHELLDER.value,
    Species.CLOYSTER.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.LICKITUNG.value,
    Species.RHYDON.value,
    Species.KANGASKHAN.value,
    Species.HORSEA.value,
    Species.SEADRA.value,
    Species.GOLDEEN.value,
    Species.SEAKING.value,
    Species.STARYU.value,
    Species.STARMIE.value,
    Species.GYARADOS.value,
    Species.LAPRAS.value,
    Species.VAPOREON.value,
    Species.OMANYTE.value,
    Species.OMASTAR.value,
    Species.KABUTO.value,
    Species.KABUTOPS.value,
    Species.SNORLAX.value,
    Species.DRATINI.value,
    Species.DRAGONAIR.value,
    Species.DRAGONITE.value,
    Species.MEW.value,
}

STRENGTH_SPECIES_IDS = {
    Species.CHARMANDER.value,
    Species.CHARMELEON.value,
    Species.CHARIZARD.value,
    Species.SQUIRTLE.value,
    Species.WARTORTLE.value,
    Species.BLASTOISE.value,
    Species.EKANS.value,
    Species.ARBOK.value,
    Species.SANDSHREW.value,
    Species.SANDSLASH.value,
    Species.NIDOQUEEN.value,
    Species.NIDOKING.value,
    Species.CLEFAIRY.value,
    Species.CLEFABLE.value,
    Species.JIGGLYPUFF.value,
    Species.WIGGLYTUFF.value,
    Species.PSYDUCK.value,
    Species.GOLDUCK.value,
    Species.MANKEY.value,
    Species.PRIMEAPE.value,
    Species.POLIWHIRL.value,
    Species.POLIWRATH.value,
    Species.MACHOP.value,
    Species.MACHOKE.value,
    Species.MACHAMP.value,
    Species.GEODUDE.value,
    Species.GRAVELER.value,
    Species.GOLEM.value,
    Species.SLOWPOKE.value,
    Species.SLOWBRO.value,
    Species.SEEL.value,
    Species.DEWGONG.value,
    Species.GENGAR.value,
    Species.ONIX.value,
    Species.KRABBY.value,
    Species.KINGLER.value,
    Species.EXEGGUTOR.value,
    Species.CUBONE.value,
    Species.MAROWAK.value,
    Species.HITMONLEE.value,
    Species.HITMONCHAN.value,
    Species.LICKITUNG.value,
    Species.RHYHORN.value,
    Species.RHYDON.value,
    Species.CHANSEY.value,
    Species.KANGASKHAN.value,
    Species.ELECTABUZZ.value,
    Species.MAGMAR.value,
    Species.PINSIR.value,
    Species.TAUROS.value,
    Species.GYARADOS.value,
    Species.LAPRAS.value,
    Species.SNORLAX.value,
    Species.DRAGONITE.value,
    Species.MEWTWO.value,
    Species.MEW.value,
}
