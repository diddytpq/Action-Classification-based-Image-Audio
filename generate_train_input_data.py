from moviepy.editor import VideoFileClip
import os, sys
from utils.utils import extract_wav_from_mp4
import subprocess

import torch
import cv2
import torchvision
import time

from models.transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop,
                               ToFloat, ToUInt8)

from models.raft.raft_src.raft import RAFT, InputPadder

from custom_model import Img_Audio_Feature_Extraction


def split_video_into_frames(video_ori_path, video_split_path, filename, num_frames):
    video = VideoFileClip(os.path.join(video_ori_path, filename))
    total_duration = video.duration
    video_fps = video.fps
    duration_per_frame = 1 / video_fps

    video_num = int(total_duration * video_fps / num_frames)

    filename = filename.split(".")[0]
    
    for i in range(video_num):
        start_time = (i) * num_frames * duration_per_frame
        end_time = (i + 1) * num_frames * duration_per_frame

        frame = video.subclip(start_time, end_time)
        frame.write_videofile(os.path.join(video_split_path, filename) +  '_' + str(i) + '.mp4', codec='libx264')

    video.close()
    return True

if __name__ == "__main__":

    #feature 추출을 위한 모델 정의
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    I3D_weight_path = {"rgb" : "./models/i3d/checkpoints/i3d_rgb.pt",
                    "flow" : "./models/i3d/checkpoints/i3d_flow.pt"}

    RAFT_weight_path = "./models/raft/checkpoints/raft-sintel.pth"
    VGGISH_weight_path = "./models/vggish/checkpoints/vggish-10086976.pth"

    class_list = ["normal", "abnormal"]

    data_type = "train" #"test"

    split_video_path = f"./dataset/{data_type}/split_videos/"
    feature_save_path = f"./dataset/{data_type}/features/"

    min_side_size = 256

    resize_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        ResizeImproved(min_side_size),
                                                        PILToTensor(),
                                                        ToFloat(),])
    


    split_frame = 30

    video_ori_path = f'./dataset/{data_type}/videos/'
    video_split_path = f'./dataset/{data_type}/split_videos/'
    audio_path = f'./dataset/{data_type}/audios/'

    for class_name in class_list:
        os.makedirs(os.path.join(video_split_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(audio_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(feature_save_path, class_name), exist_ok=True)

        video_list = os.listdir(os.path.join(video_ori_path, class_name))

        # 한 영상을 30장의 프레임 영상으로 나누기
        for video_name in video_list:
            frame_videos = split_video_into_frames(os.path.join(video_ori_path, class_name),\
                                                    os.path.join(video_split_path, class_name), \
                                                    video_name, \
                                                    split_frame)

        # 분활된 영상에서 wav 추출
        video_list = os.listdir(os.path.join(video_split_path, class_name))
        for video_name in video_list:
            wav_name = video_name.split(".")[0] + ".wav"
            command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(os.path.join(video_split_path, class_name, video_name), \
                                                                            os.path.join(audio_path, class_name, wav_name))
                
            subprocess.call(command, shell=True)


        split_video_list = os.listdir(os.path.join(split_video_path, class_name))

        for video_name in split_video_list:
            print(video_name)
            audio_wav_name = video_name.split(".")[0] + ".wav"
            audio_wav_path = os.path.join(audio_path, class_name, audio_wav_name)

            cap = cv2.VideoCapture(os.path.join(split_video_path, class_name, video_name))
            frame_num = 0
            first_frame = True
            padder = None

            rgb_stack = []
            stack_size = 29

            model = Img_Audio_Feature_Extraction(I3D_weight_path, RAFT_weight_path, audio_path = audio_wav_path, img_stack_size = stack_size, device=device)

            while True:
                t0 = time.time()
                frame_exists, rgb_ori = cap.read()
                
                if frame_exists:
                    frame_num += 1

                    if first_frame:
                        first_frame = False
                        if frame_exists is False:
                            continue

                    rgb = cv2.cvtColor(rgb_ori, cv2.COLOR_BGR2RGB)
                    rgb = resize_transforms(rgb)
                    rgb = rgb.unsqueeze(0)

                    if padder is None:
                        padder = InputPadder(rgb.shape)
                        model.padder = padder

                    rgb_stack.append(rgb)

                    with torch.no_grad():
                        if len(rgb_stack) - 1 == stack_size:
                            rgb_stack_input = torch.cat(rgb_stack).to(device)
                            result = model(rgb_stack_input)

                            feature = [result[0].squeeze(0),result[1].squeeze(0),result[2].squeeze(0)] # feature : rgb, flow, audio

                            torch.save(feature, os.path.join(feature_save_path, class_name, "{}".format(video_name.split('.')[0]) + ".pt"))
                            print(os.path.join(feature_save_path, class_name, "{}".format(video_name.split('.')[0]) + ".pt"))
                            print("feature saved")

                            rgb_stack = rgb_stack[stack_size:]

                        print("FPS: ", 1/(time.time() - t0))

                    # cv2.imshow("img", rgb_ori)
                    cv2.waitKey(1)

                else:
                    print("End of video")
                    cap.release()
                    break