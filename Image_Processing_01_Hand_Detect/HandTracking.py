import cv2
from my_class import HandDetector

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print('Failed to capture image')
            break
        img = detector.find_hands(img)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
