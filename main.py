import cv2
import numpy as np
import dlib
from math import hypot

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def length(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def averagePoints(p1, p2):
    return (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2),
    )


def getEyePoints(gray, face):
    landmarks = predictor(gray, face)
    points = []
    for i in range(36, 48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x, y))
    return points


def drawEye(eyePoints):
    leftEye = eyePoints[:6]
    rightEye = eyePoints[6:]

    leftHorizontalP1, leftHorizontalP2 = leftEye[0], leftEye[3]
    rightHorizontalP1, rightHorizontalP2 = rightEye[0], rightEye[3]

    leftVerticalP1, leftVerticalP2 = (
        averagePoints(leftEye[1], leftEye[2]),
        averagePoints(leftEye[4], leftEye[5]),
    )

    rightVerticalP1, rightVerticalP2 = (
        averagePoints(rightEye[1], rightEye[2]),
        averagePoints(rightEye[4], rightEye[5]),
    )

    # print(leftVerticalP1, leftVerticalP2)

    leftRatio = length(leftHorizontalP1, leftHorizontalP2) / length(
        leftVerticalP1, leftVerticalP2
    )
    rightRatio = length(rightHorizontalP1, rightHorizontalP2) / length(
        rightVerticalP1, rightVerticalP2
    )

    diff = leftRatio - rightRatio
    # global yay
    # print(yay, diff, end=" ")
    # yay += 1
    print(diff)
    avgRatio = (leftRatio + rightRatio) / 2
    if avgRatio > 4.75:
        print("neutral")
    else:
        if diff > 0.25:
            print("left")
        elif diff < -0.25:
            print("right")
        else:
            print("front")
    
        

    cv2.line(frame, leftHorizontalP1, leftHorizontalP2, (170, 255, 34), 1)
    cv2.line(frame, rightHorizontalP1, rightHorizontalP2, (170, 255, 34), 1)

    cv2.line(frame, leftVerticalP1, leftVerticalP2, (170, 255, 34), 1)
    cv2.line(frame, rightVerticalP1, rightVerticalP2, (170, 255, 34), 1)


cap = cv2.VideoCapture(1)
detector = dlib.get_frontal_face_detector()

while True:
    _, frame = cap.read()

    if not _:
        print("Could not open camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (170, 255, 34), 2)

        eyePoints = getEyePoints(gray, face)
        drawEye(eyePoints)
        # for landmark in eyePoints:
        #     cv2.circle(frame, (x, y), 2, (170, 255, 34), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()