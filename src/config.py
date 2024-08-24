import torch

GAME_DIR = '~/Library/Containers/io.playcover.PlayCover/Applications/'
ICON_DIR = 'assets/'
AUDIO_DIR = 'assets/launch.mp3'

WIN = {
    'GAME': 'com.miHoYo.Nap',
    'ICON': 'zzz.png',
}

DRAW = {
    'GAME': 'com.miHoYo.hkrpg',
    'ICON': 'hsr.png',
    'FALLBACK': 'https://sr.mihoyo.com/cloud/#/'
}

LOSE = {
    'GAME': 'com.miHoYo.Yuanshen',
    'ICON': 'genshin.png',
    'FALLBACK': 'https://ys.mihoyo.com/cloud/#/'
}

NONE = {
    'GAME': 'games.Pigeon.Phigros',
    'ICON': 'phigros.png',
}

device = torch.device('mps')
batch_size = 32