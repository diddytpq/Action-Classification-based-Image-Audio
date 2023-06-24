from moviepy.editor import VideoFileClip
import os, sys
from utils.utils import extract_wav_from_mp4
import subprocess



def split_video_into_frames(video_path, filename, num_frames):
    video = VideoFileClip(video_path + filename)
    total_duration = video.duration
    video_fps = video.fps
    duration_per_frame = 1 / video_fps

    video_num = int(total_duration * video_fps / num_frames)

    filename = filename.split(".")[0]
    
    for i in range(video_num):
        start_time = (i) * num_frames * duration_per_frame
        end_time = (i + 1) * num_frames * duration_per_frame

        frame = video.subclip(start_time, end_time)
        # frame.write_videofile(f'./dataset/train/split_videos/normal/'+ filename + '_' + str(i) + '.mp4', codec='libx264')
        frame.write_videofile(f'./dataset/train/split_videos/abnormal/'+ filename + '_' + str(i) + '.mp4', codec='libx264')
        # frame.write_videofile(f'./dataset/train/split_videos/abnormal/_{i}.mp4', codec='libx264')

    video.close()
    return True

# video_path = './dataset/train/videos/normal/'
video_path = './dataset/train/videos/abnormal/'

video_list = os.listdir(video_path)

for filename in video_list:
    # 영상 파일 경로
    # filename = './videos/normal.mp4'
    # filename = './videos/abnormal.mp4'

    # 한 영상을 10장의 프레임 영상으로 나누기
    num_frames = 30
    frame_videos = split_video_into_frames(video_path, filename, num_frames)


# video_path = './dataset/train/split_videos/normal/'
video_path = './dataset/train/split_videos/abnormal/'
audio_path = './dataset/train/audios/abnormal/'

video_list = os.listdir(video_path)

for video_name in video_list:
    # video_name = "normal.mp4"
    wav_name = video_name.split(".")[0] + ".wav"
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(video_path + video_name, audio_path + wav_name)
        
    subprocess.call(command, shell=True)




