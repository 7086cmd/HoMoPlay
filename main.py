from src.utils import timer, random_choice, select_image, check_result, launch_game
from src.train import load_model, train
from src.recognize import recognise_people_gesture, init_camera
from src.plot import plot_image
from src.config import ICON_DIR


def main_evaluate():
    init_camera()
    load_model('assets/gesture_classifier.pth')
    timer(3)
    computer = random_choice(['rock', 'paper', 'scissors'])
    computer_frame = select_image(computer)
    user_frame, user = recognise_people_gesture()
    config, msg = check_result(user, computer)
    plot_image(msg, user_frame, computer_frame, user, computer, ICON_DIR + config['ICON'])
    launch_game(config['GAME'], config['FALLBACK'] if 'FALLBACK' in config else None)
    print(f'Computer: {computer}, User: {user}')


def main_training():
    load_model('assets/gesture_classifier.pth')
    train(epochs=10, save_model_to='assets/gesture_classifier.pth')

if __name__ == '__main__':
    main_evaluate()
