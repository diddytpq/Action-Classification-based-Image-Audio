import numpy as np
import torch
import torchvision

from models.i3d.i3d_src.i3d_net import I3D
from models.raft.extract_raft import DATASET_to_RAFT_CKPT_PATHS
from models.raft.raft_src.raft import RAFT, InputPadder
from models.vggish.vggish_src.vggish_slim import VGGish

from models.transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop,
                               ToFloat, ToUInt8)

from utils.utils import dp_state_to_normal, extract_wav_from_mp4
import utils.flow_viz as flow_viz

import cv2
import time
import math

class Img_Audio_Feature_Extraction(torch.nn.Module):
    def __init__(self, I3D_weight_path, RAFT_weight_path, audio_path = None, img_stack_size = 30, device = 'cpu'):
        super().__init__()

        self.device = device
        self.padder = None
        self.stack_size = img_stack_size
        self.central_crop_size = 224
        self.audio_path = audio_path

        self.i3d_model_rgb = I3D(num_classes=400, modality='rgb')
        self.i3d_model_rgb.load_state_dict(torch.load(I3D_weight_path['rgb'], map_location='cpu'))
        self.i3d_model_rgb = self.i3d_model_rgb.to(self.device).eval()


        self.i3d_model_flow = I3D(num_classes=400, modality='flow')
        self.i3d_model_flow.load_state_dict(torch.load(I3D_weight_path['flow'], map_location='cpu'))
        self.i3d_model_flow = self.i3d_model_flow.to(self.device).eval()

        self.raft_model = RAFT()
        state_dict = dp_state_to_normal(torch.load(RAFT_weight_path, map_location='cpu'))
        self.raft_model.load_state_dict(state_dict)
        self.raft_model = self.raft_model.to(self.device).eval()

        self.vggish_model = VGGish().to(self.device).eval()

        self.i3d_transforms = {
            'rgb': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }

    def forward(self, x, audio_input = None):
        img_flow = self.raft_model(self.padder.pad(x)[:-1], self.padder.pad(x)[1:])

        rgb_input = self.i3d_transforms['rgb'](x[:-1])
        flow_input = self.i3d_transforms['flow'](img_flow)

        img_feature = self.i3d_model_rgb(rgb_input, features=True)
        flow_feature = self.i3d_model_flow(flow_input, features=True)

        if self.audio_path is not None:
            audio_feature = self.vggish_model(self.audio_path, self.device)
        
        else:
            audio_feature = self.vggish_model(audio_input, self.device, fs = 44100)
        
        return img_feature, flow_feature, audio_feature


    def show_optical_flow(self, x):
        img_flow = self.raft_model(self.padder.pad(x)[:-1], self.padder.pad(x)[1:])

        # # maybe un-padding only before saving because np.concat will not work if the img is unpadded
        if self.padder is not None:
            batch_feats = self.padder.unpad(img_flow)

        for idx, flow in enumerate(batch_feats):
            img = x[idx].permute(1, 2, 0).cpu().numpy()
            flow = flow.permute(1, 2, 0).cpu().numpy()
            flow = flow_viz.flow_to_image(flow)
            img_flow = np.concatenate([img, flow], axis=0)
            cv2.imshow('Press any key to see the next frame...', img_flow[:, :, [2, 1, 0]] / 255.0)
            cv2.waitKey()

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class Action_Classification_Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.block_1 = torch.nn.Sequential(
                        torch.nn.Linear(2176, 512),
                        torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        torch.nn.Dropout(p=0.2),
                        # torch.nn.Sigmoid(),
                        Swish(),

                        # torch.nn.Linear(2048, 1024),
                        # torch.nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(1024, 512),
                        # torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(512, 1024),
                        # torch.nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(1024, 512),
                        # torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(512, 512),
                        # torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(512, 512),
                        # torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(512, 512),
                        # torch.nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(512, 256),
                        # torch.nn.BatchNorm1d(256, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(256, 128),
                        # torch.nn.BatchNorm1d(128, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        # torch.nn.Linear(128, 64),
                        # torch.nn.BatchNorm1d(64, momentum=0.99, eps=1e-3),
                        # # torch.nn.Sigmoid(),
                        # Swish(),

                        torch.nn.Linear(512, 128),
                        torch.nn.BatchNorm1d(128, momentum=0.99, eps=1e-3),
                        # torch.nn.Dropout(p=0.2),
                        # torch.nn.Sigmoid(),
                        Swish(),

                        torch.nn.Linear(128, 64),
                        torch.nn.BatchNorm1d(64, momentum=0.99, eps=1e-3),
                        # torch.nn.Dropout(p=0.2),
                        # torch.nn.Sigmoid(),
                        Swish(),

                        torch.nn.Linear(64, 2),
                        torch.nn.Softmax(),
                        # torch.nn.Sigmoid(),
        )

        self.init_weights()

        self.block_rgb = torch.nn.Sequential(
                    torch.nn.Linear(1024, 512),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    # torch.nn.Linear(256, 2),
                    # torch.nn.Softmax(),
                    # torch.nn.Sigmoid(),
            )
        
        self.block_flow = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 2),
            # torch.nn.Softmax(),
            # torch.nn.Sigmoid(),
            )

        self.block_audio = torch.nn.Sequential(
                    torch.nn.Linear(128, 64),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.ReLU(),
                    # torch.nn.Linear(256, 2),
                    # torch.nn.Softmax(),
                    # torch.nn.Sigmoid(),
            )

        self.block_cls = torch.nn.Sequential(
                    torch.nn.Linear(64 * 3, 1),
                    # torch.nn.Softmax(),
                    torch.nn.Sigmoid(),
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                # torch.nn.init.uniform_(m.weight, -init_range, init_range)
                torch.nn.init.xavier_uniform_(m.weight)



    def forward(self, x):
        output_img = self.block_rgb(x[0].to(self.device))
        output_flow = self.block_flow(x[1].to(self.device))
        output_audio = self.block_audio(x[2].to(self.device))

        output = self.block_cls(torch.cat((output_img, output_flow, output_audio), dim=1))

        # output = self.block_2(x)

        return output 

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    video_path = "./dataset/train/videos/normal_0.mp4"
    audio_wav_path = "./dataset/train/audios/normal_0.wav"

    # video_path = "./videos/normal.mp4"
    # audio_wav_path = "./dataset/train/audios/normal.wav"

    # audio_wav_path, audio_aac_path = extract_wav_from_mp4(video_path, tmp_path = './tmp')

    cap = cv2.VideoCapture(video_path)

    I3D_weight_path = {"rgb" : "./models/i3d/checkpoints/i3d_rgb.pt",
                       "flow" : "./models/i3d/checkpoints/i3d_flow.pt"}
    
    RAFT_weight_path = "./models/raft/checkpoints/raft-sintel.pth"
    VGGISH_weight_path = "./models/vggish/checkpoints/vggish-10086976.pth"

    min_side_size = 256

    resize_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        ResizeImproved(min_side_size),
                                                        PILToTensor(),
                                                        ToFloat(),])

    rgb_stack = []
    stack_size = 24

    model = Img_Audio_Feature_Extraction(I3D_weight_path, RAFT_weight_path, audio_path = audio_wav_path, img_stack_size = stack_size, device=device)
    ac_model = Action_Classification_Model().to(device).eval()
    
    frame_num = 0
    first_frame = True
    padder = None
    while cap.isOpened():
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

            print(len(rgb_stack))

            with torch.no_grad():
                if len(rgb_stack) - 1 == stack_size:

                    rgb_stack_input = torch.cat(rgb_stack).to(device)
                    result = model(rgb_stack_input)
                    # model.show_optical_flow(rgb_stack_input)

                    print(result[0].shape, result[1].shape, result[2].shape)
                    feature = torch.cat((result[0],result[1],result[2]), 1)
                    print(feature.size())
                    output = ac_model(feature)
                    # feature_save_path_name = "./dataset/train/feature/normal/feature_0" + ".pt"
                    # torch.save(feature, feature_save_path_name)

                    # rgb_stack = rgb_stack[1:]
                    rgb_stack = rgb_stack[stack_size:]

            print("FPS: ", 1/(time.time() - t0))

            cv2.imshow("img", rgb_ori)
            cv2.waitKey(1)

        else:
            print("End of video")
            break