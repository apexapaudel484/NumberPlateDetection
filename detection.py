import cv2
import torch 

from ultralytics import YOLO

model=YOLO()
device = "cuda" if torch.cuda.is_available() else "cpu"

def model_to_use():
    model=YOLO('exp-5-v1-e60-b16-p3.pt')
    return model

def video_functions(video):
#calculating the values of the video 
    cap =cv2.VideoCapture(video)#'stable_video.mp4')
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Box_color=(0,0,255)
    Text_color=(255,255,255)#white
    VIDEO_CODEC= 'mp4v'
    return fps , width , height , Box_color,Text_color,VIDEO_CODEC

def output(input_video,output_video):
    #creating video output displaying the detection.
    fps,width,height,Box_color,Text_color,VIDEO_CODEC = video_functions(input_video)
    model=model_to_use()
    out = cv2.VideoWriter(output_video,
                        cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                        fps,
                        (width,height))

    cam = cv2.VideoCapture(input_video)
    fps=int(cam.get(cv2.CAP_PROP_FPS))
    currentframe=0
    while True:
        ret,frame=cam.read()
        if ret:
            predicting=model(frame,save=False,imgsz=640,conf=0.5)
            num_detection=0
            for b in predicting:
                boxes = b.boxes
                # print(frame)
                # imgs=cv2.imread(frame)
                for box in boxes:
                    # Get box coordinates in (left, top, right, bottom) format
                    b = box.xyxy[0]
                    b=b.tolist()
                    # Get confidence score and class label
                    conf = box.conf
                    cls = box.cls
                    frame=cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color=Box_color,thickness=2)
                    class_name=model.names[int(cls)]
                    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    frame=cv2.rectangle(frame, (int(b[0]), int(b[1]) - int(1.5 * text_height)), (int(b[0]) + text_width,  int(b[1])), Box_color, -1)
                    cv2.putText(
                        frame,
                        text=class_name,
                        org=(int(b[0]), int(b[1]) - int(0.5 * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=Text_color,
                        lineType=cv2.LINE_AA)
                    num_detection+=1
                cv2.putText(
                    frame,
                    text=f'Number Plate Detected:{num_detection}',
                    org=(50,50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=Text_color,
                    lineType=cv2.LINE_AA
                )
                out.write(frame)
            currentframe +=1
            
        else:
            break

    out.release()
    cam.release()
    cv2.destroyAllWindows()
