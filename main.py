import cv2
import numpy as np
import os
import mediapipe as mp
from datetime import datetime

# Folder for resources
folderPath = "Resources"
eraserIcon = cv2.imread(os.path.join(folderPath, "eraser.png"))
saveIcon = cv2.imread(os.path.join(folderPath, "save.png"))

# Resize icons for clarity
eraserIcon = cv2.resize(eraserIcon, (80, 80))
saveIcon = cv2.resize(saveIcon, (80, 80))

# Colors: BGR
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 0)]
colorNames = ["Pink", "Blue", "Green", "Yellow", "Eraser"]
colorIndex = 0
drawColor = colors[colorIndex]

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Brush thickness
brushThickness = 15
eraserThickness = 50

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Canvas
canvas = np.zeros((720, 1280, 3), np.uint8)

# Header bar layout
button_y1, button_y2 = 0, 100
button_positions = {
    "eraser": (20, 100),
    "color0": (140, 220),
    "color1": (260, 340),
    "color2": (380, 460),
    "color3": (500, 580),
    "save": (1180, 1260)
}

xp, yp = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # Draw header UI
    cv2.rectangle(img, (0, 0), (1280, 100), (50, 50, 50), -1)

    # Draw Eraser Button
    x1, x2 = button_positions["eraser"]
    img[10:90, x1:x2] = eraserIcon
    cv2.rectangle(img, (x1, 10), (x2, 90), (200, 200, 200), 2)

    # Draw Save Button
    x1, x2 = button_positions["save"]
    img[10:90, x1:x2] = saveIcon
    cv2.rectangle(img, (x1, 10), (x2, 90), (200, 200, 200), 2)

    # Draw Color Rectangles
    for i in range(4):
        x1, x2 = button_positions[f"color{i}"]
        cv2.rectangle(img, (x1, 10), (x2, 90), colors[i], -1)
        cv2.rectangle(img, (x1, 10), (x2, 90), (200, 200, 200), 2)

    # Hand Detection
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Index & Middle finger
            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]

            fingersUp = y2 < y1

            # Selection Mode
            if y1 < 100:
                for i in range(4):
                    bx1, bx2 = button_positions[f"color{i}"]
                    if bx1 < x1 < bx2:
                        colorIndex = i
                        drawColor = colors[i]
                        xp, yp = 0, 0

                # Eraser
                ex1, ex2 = button_positions["eraser"]
                if ex1 < x1 < ex2:
                    drawColor = (0, 0, 0)
                    xp, yp = 0, 0

                # Save
                sx1, sx2 = button_positions["save"]
                if sx1 < x1 < sx2:
                    filename = f'Drawing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                    cv2.imwrite(filename, canvas)
                    print(f"[INFO] Drawing saved as {filename}")

            # Drawing Mode
            if fingersUp:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    # Combine image and canvas
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Paint", img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
