import time
import random
from src.config import DRAW, WIN, LOSE, NONE
import cv2
from src.config import GAME_DIR, AUDIO_DIR
from src.datasets import dataset as data


def timer(sec: int = 3):
    for i in range(sec, 0, -1):
        print(f"{i}")
        time.sleep(1)
    print("Go!")
    time.sleep(0.1)


def random_choice(choices):
    if choices is None:
        choices = ['rock', 'paper', 'scissors']
    return random.choice(choices)


def select_image(choice: str):
    path = f'data/rps/{choice}/'
    import os
    files = list(filter(lambda x: not x.startswith('kaggle'), os.listdir(path)))
    return cv2.imread(path + random.choice(files))


def check_result(user: str, computer: str):
    if user == 'none':
        return NONE, 'No action!'
    elif user == computer:
        # Draw
        return DRAW, 'Draw!'
    elif (user == 'rock' and computer == 'scissors') or \
            (user == 'scissors' and computer == 'paper') or \
            (user == 'paper' and computer == 'rock'):
        # User wins
        return WIN, 'You win!'
    else:
        # Computer wins
        return LOSE, 'You lose!'


def launch_game(app: str, fallback: str):
    game = GAME_DIR + app + '.app'
    import os
    import pygame

    # Initialize the mixer module
    pygame.mixer.init()

    # Load the audio file
    pygame.mixer.music.load(AUDIO_DIR)

    try:
        os.system(f'open -a {game}')
        time.sleep(1.5)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except:
        os.system(f'open {fallback}')
