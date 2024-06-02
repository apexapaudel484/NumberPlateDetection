import os
import time

import cv2
import streamlit as st

from final_model import output

st.title("Number Plate Detection")

video_source = st.text_input('Enter video source').strip()

def get_video_source():
    if len(video_source)<1:
        st.stop()
    
    st.session_state.video_source = int(video_source) if video_source.isnumeric() else video_source

def change_video_source():
    del st.session_state['cam']
    get_video_source()

def set_video_source():
    if 'cam' not in st.session_state:
        st.session_state['cam'] = cv2.VideoCapture(st.session_state.video_source)

get_video_source()

set_video_source()

output_name = st.text_input('Enter the output video name').strip()

@st.cache_resource
def result(output_name):
    output(st.session_state.video_source, output_name)
    
# result(output_name)

def display_video(output_filename):   
    # output_filename = os.path.join('data_used/',output_filename)
    video_file = open(output_filename,'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

display_video(st.session_state.video_source)

if 'img_container' not in st.session_state: 
    st.session_state.img_container = st.empty()

def display_result(filepath=output_name):
    cam = cv2.VideoCapture(filepath)

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        st.session_state.img_container.image(frame)
        time.sleep(1/29)


st.button("Display Result", on_click=display_result)
