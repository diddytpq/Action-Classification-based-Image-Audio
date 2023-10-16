from moviepy.editor import VideoFileClip

def split_video_into_frames(filename, num_frames):
    video = VideoFileClip(filename)
    total_duration = video.duration
    video_fps = video.fps
    duration_per_frame = 1 / video_fps

    video_num = int(total_duration * video_fps / num_frames)
    
    for i in range(video_num):
        start_time = (i) * num_frames * duration_per_frame
        end_time = (i + 1) * num_frames * duration_per_frame

        frame = video.subclip(start_time, end_time)
        frame.write_videofile(f'./dataset/train/videos/normal_{i}.mp4', codec='libx264')
    video.close()
    return True

# 영상 파일 경로
filename = './videos/normal.mp4'

# 한 영상을 10장의 프레임 영상으로 나누기
num_frames = 25
frame_videos = split_video_into_frames(filename, num_frames)

# 각 프레임 영상을 저장하기
# for i, frame_video in enumerate(frame_videos):
#     frame_video.write_videofile(f'./videos/clip_videos/video_{i}.mp4', codec='libx264')