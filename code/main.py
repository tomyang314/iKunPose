import cv2 as cv
from Pose import PoseDetector


cap = cv.VideoCapture('../assets/ikun.mp4')
fourcc = cv.VideoWriter_fourcc(*'mp4v')

detector = PoseDetector()

width = int(cap.get(3))
height = int(cap.get(4))

out = cv.VideoWriter('../output/out.mp4', fourcc, 30.06, (width, height))
# fps=60,这里是视频的帧率,可以随意调整,大小只影响每张图片的播放速率
# (width,height)图片的大小,如果是用视频则是分辨率大小.这里的值需与写入的图片或视频保持一致


if __name__ == '__main__':
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('can not receive frame')
            break

        background = cv.imread("../assets/background.jpg")

        frame = detector.find_pose(frame, background)
        out.write(frame)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 800, 600)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

