
import cv2

for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret and frame is not None:
                print(f"OK: camera={i}, backend={backend}")
                cv2.imshow("Camera test", frame)
                cv2.waitKey(2000)
            else:
                print(f"Mo duoc nhung khong doc duoc: {i}, {backend}")

        cap.release()
        cv2.destroyAllWindows()

print("Da kiem tra xong")
