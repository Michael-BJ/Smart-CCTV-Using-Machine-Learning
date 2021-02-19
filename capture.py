import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('take a picture ',frame)
    if cv2.waitKey(1) & 0xFF == ord('y') :
        cv2.imwrite('etcodetech.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()