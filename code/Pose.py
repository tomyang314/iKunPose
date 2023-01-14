import cv2
import mediapipe as mp
import math
from time import sleep


mp_drawing_styles = mp.solutions.drawing_styles


class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True, detectionCon=0.5 , trackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     # model_complexity=1,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def find_pose(self, img, background, draw=True):
        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # BGR -> RGB
        self.results = self.pose.process(img_rgb)     # 检测姿势点
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(background, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        return background

    def find_position(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

        return self.lmList

    def point(self, img):
        x1, y1 = self.lmList[0][1:]
        x2, y2 = self.lmList[16][1:]
        x3, y3 = self.lmList[15][1:]

        cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)

        return img


def main():
    # url = "http://admin:admin@192.168.43.1:8081"
    video_xxx = "./ikun.mp4"
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(url)
    cap = cv2.VideoCapture(video_xxx)
    detector = PoseDetector()
    # iris = IrisTrack()
    print("wait...")
    # conn = esp8266_init()

    while True:
        try:
            success, img = cap.read()
            if not success:
                break
            img = cv2.resize(img, None, fx=0.75, fy=0.75)
            background = cv2.imread("./jntm.jpg")
            # img0 = iris.iris_track(img)
            img = cv2.flip(img, 1)
            img.flags.writeable = False
            img = detector.find_pose(img, background, draw=False)
            # img = iris.iris_track(img)
            lmList = detector.find_position(img)
            background = detector.point(background)

            cv2.imshow("Image", background)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except:
            pass

    cap.release()
    cv2.destroyAllWindows()
    print("end...")
    # conn.close()


if __name__ == '__main__':
    main()
