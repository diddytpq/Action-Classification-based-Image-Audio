import os, sys
from utils.utils import extract_wav_from_mp4
import subprocess


video_path = "./videos/"
video_name = "test_11.mp4"

audio_path = "./audios/"
wav_name = video_name.split(".")[0] + ".wav"

command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path + video_name, audio_path + wav_name)
    
subprocess.call(command, shell=True)




