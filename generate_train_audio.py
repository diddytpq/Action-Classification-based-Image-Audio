import os, sys
from utils.utils import extract_wav_from_mp4
import subprocess

video_path = './dataset/train/videos/'
audio_path = './dataset/train/audios/'

video_list = os.listdir(video_path)

for video_name in video_list:
    # video_name = "normal.mp4"
    wav_name = video_name.split(".")[0] + ".wav"
    # command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path + video_name, audio_path + wav_name)
    command = f"ffmpeg -ab 160k -ac 2 -ar 44100 -vn -i {video_path + video_name} {audio_path + wav_name}"
        
    subprocess.call(command, shell=True)




