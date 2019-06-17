# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import win32api


args = {'shape_predictor': 'shape_predictor_68_face_landmarks.dat', 'picamera': 0}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    print(rects)

    # loop over the face detections
    image_points = None
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                 tuple(shape[48]), tuple(shape[54])])

        for (x, y) in image_points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        image_points = np.array([tuple(shape[30]), tuple(shape[8]), tuple(shape[36]), tuple(shape[45]),
                                 tuple(shape[48]), tuple(shape[54])], dtype='double')

    if len(rects) > 0:
        # cv2.putText(frame, "detected", (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        size = frame.shape

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype='double')

        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        mat = np.hstack((rotation_matrix, translation_vector))
        # homogeneous transformation matrix

        (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
        yaw = eulerAngles[1]
        pitch = abs(180 - eulerAngles[0]) * np.sign(eulerAngles[0])
        roll = eulerAngles[2]
        print(yaw, pitch, roll)

        x_now, y_now = win32api.GetCursorPos()
        x_move = np.sign(yaw) * 15 * (yaw / 40)**2
        y_move = np.sign(pitch) * 15 * (pitch / 40)**2
        win32api.SetCursorPos((int(x_now+x_move), int(y_now+y_move)))

        cv2.putText(frame, 'yaw' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'pitch' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'roll' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)


        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        print('reached')

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
