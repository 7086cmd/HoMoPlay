import cv2
import torch
from cv2.typing import MatLike
from src.datasets import eval_transform
from src.config import device
from src.model import model

camera: cv2.VideoCapture


def init_camera(video_capture=0):
    global camera
    camera = cv2.VideoCapture(video_capture)


def recognise_gesture(image: MatLike):
    image: torch.Tensor = eval_transform(image).float().to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        targets = ['none', 'paper', 'rock', 'scissors']
        return targets[predicted.item()]


def recognise_people_gesture(video_capture=0):
    global camera, frame
    recognised = {'none': 0, 'paper': 0, 'rock': 0, 'scissors': 0}
    for i in range(30):
        _, frame = camera.read()
        value = recognise_gesture(frame)
        recognised[value] += 1
    # Get the max one
    max_value = max(recognised.values())
    camera.release()
    for key, value in recognised.items():
        if value == max_value:
            return frame, key
    return frame, 'none'
