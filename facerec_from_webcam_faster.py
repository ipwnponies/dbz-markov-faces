import textwrap
from functools import lru_cache
from pathlib import Path

import face_recognition
import face_recognition.api
import cv2
import numpy

import markov_me

DEBUG = False

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)


def load_face(face_file):
    image = face_recognition.load_image_file(face_file)
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding


@lru_cache(maxsize=1)
def known_face_data():
    known_face_names, known_face_encodings = zip(*(
        (file.stem, load_face(file))
        for file in Path('./face-data/').glob('*.jpeg')
    ))

    return known_face_names, known_face_encodings


def process_frame(frame):
    scale_factor = 0.25
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    known_face_names, known_face_encodings = known_face_data()

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    scaled_up_face_locations = []
    for points in face_locations:
        scaled_up_face_locations.append(tuple(int(i / scale_factor) for i in points))

    return scaled_up_face_locations, face_names


def tag_name(frame, name, top, right, bottom, left):
    if DEBUG:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_DUPLEX

    # Make boundaries sticky, so they move less often
    def sticky(num, quantize=100):
        import math
        return math.ceil(num / quantize) * quantize

    top = sticky(top)
    right = sticky(right)
    bottom = sticky(bottom)
    left = sticky(left)

    for text in textwrap.wrap(name, 40):
        font_scale = 0.8
        thickness = 1
        padding = 6
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        cv2.rectangle(
            frame,
            (left - padding, bottom - text_height - padding),
            (left + text_width + padding, bottom + padding),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (left, bottom),
            font,
            font_scale,
            (0, 0, 0),
            thickness=thickness,
        )
        bottom += text_height + padding


@lru_cache(maxsize=1)
def get_saiyan_hair_img():
    return cv2.imread('assets/saiyan_hair.png', cv2.IMREAD_UNCHANGED)


def add_mustache(frame, face_locations):
    # Find all facial features in all the faces in the image
    raw_face_landmarks_list = face_recognition.api._raw_face_landmarks(frame, face_locations)
    face_landmarks_list = [[(p.x, p.y) for p in landmark.parts()] for landmark in raw_face_landmarks_list]

    # Draw mustache
    for face_landmarks in face_landmarks_list:

        # top lip
        mustache = tuple((x, y - 10) for x, y in face_landmarks[48:55])
        cv2.polylines(frame, [numpy.array(mustache, dtype=numpy.int32)], False, (0, 0, 0), 8)

    # Draw saiyan hair
    for index, face_landmarks in enumerate(face_landmarks_list):
        forehead_y = face_landmarks[27][1]

        s_img = get_saiyan_hair_img()
        (top, right, bottom, left) = face_locations[index]
        poofy_hair_factor = 1.5
        head_size = (int((right-left) * poofy_hair_factor), int((bottom - top) * poofy_hair_factor))
        s_img = cv2.resize(s_img, head_size)

        # assumes vertically alinged beccause i don't have time to linear algebra
        head_tip = forehead_y

        leftmost_face = face_landmarks[0][0] - int((s_img.shape[0] * 0.3))
        y1, y2 = head_tip - s_img.shape[0], head_tip
        x1, x2 = leftmost_face, leftmost_face + s_img.shape[1]

        if DEBUG:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

        # Don't draw if clipped
        # Becaseu i don't know how this math works and you obviously can't draw into negative space
        height, width = frame.shape[:2]
        if y1 > 0 and x1 > 0 and y2 < height and x2 < width:
            alpha_s = s_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])


def main():
    # Initialize some variables
    face_locations = []
    face_names = []
    process_this_frame = True
    dbz_speech = {}

    video_capture = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Mirror the frame so it looks better in frame
        frame = cv2.flip(frame, 1)

        # Only process every other frame of video to save time
        if process_this_frame:
            face_locations, face_names = process_frame(frame)
            for name in face_names:
                dbz_speech.setdefault(name, next(markov_me.generate_sentence()))

        process_this_frame = not process_this_frame

        add_mustache(frame, face_locations)
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            tag_name(frame, f'{name}: {dbz_speech[name]}', top, right, bottom, left)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == main():
    main()
