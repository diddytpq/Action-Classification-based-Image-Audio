import torch
import cv2
import torchvision
import time
import os

from models.transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop,
                               ToFloat, ToUInt8)

from models.raft.raft_src.raft import RAFT, InputPadder

from custom_model import Img_Audio_Feature_Extraction


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    video_path = "./dataset/train/videos/"
    
    video_list = os.listdir(video_path)

    I3D_weight_path = {"rgb" : "./models/i3d/checkpoints/i3d_rgb.pt",
                    "flow" : "./models/i3d/checkpoints/i3d_flow.pt"}
    
    RAFT_weight_path = "./models/raft/checkpoints/raft-sintel.pth"
    VGGISH_weight_path = "./models/vggish/checkpoints/vggish-10086976.pth"

    min_side_size = 256

    resize_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        ResizeImproved(min_side_size),
                                                        PILToTensor(),
                                                        ToFloat(),])

    for video_name in video_list:
        print(video_name)
        audio_wav_name = video_name.split(".")[0] + ".wav"
        audio_wav_path = "./dataset/train/audios/" + audio_wav_name

        cap = cv2.VideoCapture(video_path + video_name)

        frame_num = 0
        first_frame = True
        padder = None

        rgb_stack = []
        stack_size = 24

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
                        # model.show_optical_flow(rgb_stack_input)
                        feature = torch.cat((result[0],result[1],result[2]), 1)
                        # feature_save_path_name = "./dataset/train/feature/normal/{}".format(video_name) + ".pt"
                        feature_save_path_name = "./dataset/train/features/{}".format(video_name.split('.')[0]) + ".pt"
                        torch.save(feature, feature_save_path_name)
                        print("feature saved")

                        # rgb_stack = rgb_stack[1:]
                        rgb_stack = rgb_stack[stack_size:]

                print("FPS: ", 1/(time.time() - t0))

                # cv2.imshow("img", rgb_ori)
                cv2.waitKey(1)

            else:
                print("End of video")
                cap.release()
                break