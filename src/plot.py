import cv2
import matplotlib.pyplot as plt
from cv2.typing import MatLike
import mediapipe as mp
from matplotlib.gridspec import GridSpec
from datetime import datetime


class PerformMosaic:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1,
                                                                   min_detection_confidence=0.5)


    def apply_mosaic(self, image: MatLike, x: int, y: int, w: int, h: int, size = 5):
        sub_face = image[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face, (size, size), interpolation = cv2.INTER_LINEAR)
        sub_face = cv2.resize(sub_face, (w, h), interpolation = cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = sub_face
        return image


    def __call__(self, image: MatLike):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_detection.process(image)
        if faces.detections:
            for detection in faces.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                image = self.apply_mosaic(image, x, y, w, h)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


transform = PerformMosaic()


def plot_image(msg: str, user: MatLike, computer: MatLike, user_action: str, computer_action: str, icon_dir: str):
    print(icon_dir)
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.2])

    # Add a title to the figure
    date_and_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f"{msg} | Combat between user and computer | at {date_and_time}", fontsize=16)

    # First subplot: User image with mosaic
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(transform(user), cv2.COLOR_RGB2BGR))
    ax1.set_title('User: ' + user_action)
    ax1.axis('off')

    # Second subplot: Computer image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(computer)
    ax2.set_title('Computer: ' + computer_action)
    ax2.axis('off')

    # Third subplot: Icon with comment spanning both columns
    ax3 = fig.add_subplot(gs[1, :])
    icon = cv2.cvtColor(cv2.imread(icon_dir), cv2.COLOR_BGR2RGB)
    ax3.imshow(icon)
    ax3.axis('off')
    ax3.text(0.5, -0.1, 'will be launched', transform=ax3.transAxes, ha='center', fontsize=12, color='black')

    # Adjust layout
    plt.tight_layout()
    plt.show()


