import cv2
import os

name_exist = True
def cam_shot():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "allpeople/"+ name +"/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()

def run_train():
    print("train")
    os.system('python train_model.py')

while (name_exist):
    name = input("Enter your Name: ")
    parent_dir = 'D:/4_2/โครงงาน/deepface/facial-recognition-main/allpeople/'
    path = os.path.join(parent_dir, name) 
    try: 
        os.mkdir(path) 
        name_exist = False
        print("Take a photo 2-3 image")
        cam_shot()
    except OSError as error: 
        print(error)  

run_train()

