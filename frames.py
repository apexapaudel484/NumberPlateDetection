import cv2
import os
def frame(video,folder):
    cam = cv2.VideoCapture(video)
    fps=int(cam.get(cv2.CAP_PROP_FPS))
    print(fps)
    try:
        os.makedirs(folder, exist_ok=True)

    except:
        print('error making directory')

    currentframe=0
    # n=0
    while True:
        ret,frame=cam.read()
        if ret:
            # print(f'reading from frame{n}') 
            # if n%fps == 0:    # to extract 1 frame from 1 sec. #disabled for video output as each frame is required
            name= f'./{folder}/test' + str(currentframe) + '.jpg'
            print('Creating...'+name)
            cv2.imwrite(name,frame)
            currentframe +=1
            # n+=1

        else:
            break

    cam.release()
    cv2.destroyAllWindows()