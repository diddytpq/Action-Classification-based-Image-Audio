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

from multiprocessing import Process, Pipe, Manager, Queue
from multiprocessing.managers import BaseManager

import numpy as np
import soundfile as sf

class Img_Buffer(object):
    def __init__(self):
        self.img_data = None
        self.img_success_flag = True
        self.img_buffer = []
        self.frame_num = 0
    def set_img_data(self, data):
        self.frame_num = data[0]
        self.img_data = data[1]
        self.img_buffer.append(data[2])

        if len(self.img_buffer) > 30:
            self.img_buffer.pop(0)

    def set_img_connect_flag(self, data):
        self.img_success_flag = data
    

    def get_img_data(self):
        return self.img_success_flag, self.frame_num, self.img_data, self.img_buffer
    
    def get_img_buffer(self):
        return self.img_buffer

def video_capture(img_buffer, source):
    resize_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        ResizeImproved(min_side_size),
                                                        PILToTensor(),
                                                        ToFloat(),])
    cap = cv2.VideoCapture(source)
    frame_num = 0

    while True:
        ret, rgb_ori = cap.read()
        
        frame_exists = True

        t0 = time.time()

        if ret:
            img_buffer.set_img_connect_flag(frame_exists)

            frame_num += 1
            
            rgb = cv2.cvtColor(rgb_ori, cv2.COLOR_BGR2RGB)
            rgb = resize_transforms(rgb)
            rgb = rgb.unsqueeze(0)

            img_buffer.set_img_data([frame_num, rgb_ori, rgb])

            cv2.imshow("img", rgb_ori)
            key = cv2.waitKey(5) # fps : 30
            
            print(1 / (time.time() - t0))

            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                frame_exists = False
                img_buffer.set_img_connect_flag(frame_exists)


        else:
            frame_exists = False
            img_buffer.set_img_connect_flag(frame_exists)
            print("End of video")
            cap.release()
            break

def wavfile_to_examples(wav_file, return_tensor=True):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.
    torch: Return data as a Pytorch tensor ready for VGGish

  Returns:
    See waveform_to_examples.
  """
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    return samples

def get_input_audio_buffer(audio_data, sampleing_rate = 44100):
    num_blocks = len(audio_data) // sampleing_rate
    audio_buffer = np.reshape(audio_data[:num_blocks * sampleing_rate], (num_blocks, sampleing_rate, 2))
    return audio_buffer

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    video_path = "./videos/test_video/test_8.mp4"
    audio_path = "./audios/test_8.wav"

    
    I3D_weight_path = {"rgb" : "./models/i3d/checkpoints/i3d_rgb.pt",
                    "flow" : "./models/i3d/checkpoints/i3d_flow.pt"}
    
    RAFT_weight_path = "./models/raft/checkpoints/raft-sintel.pth"
    VGGISH_weight_path = "./models/vggish/checkpoints/vggish-10086976.pth"

    min_side_size = 256

    audio_data = wavfile_to_examples(audio_path)

    audio_sample_rate = 44100
    video_fps = 30

    audio_data_per_frame = int(audio_sample_rate / video_fps)


    # model_feature = Img_Audio_Feature_Extraction(I3D_weight_path, RAFT_weight_path, audio_path = audio_wav_path, img_stack_size = stack_size, device=device)
    # model = Action_Classification_Model(device).to(device)

    BaseManager.register('img_buffer', Img_Buffer)
    manager = BaseManager()
    manager.start()

    img_buffer = manager.img_buffer()
    img_thread = Process(target=video_capture, args=[img_buffer, video_path])
    img_thread.start()

    # audio_wav_name = video_name.split(".")[0] + ".wav"
    # audio_wav_path = "./dataset/train/audios/normal/" + audio_wav_name
    # audio_wav_path = "./dataset/train/audios/abnormal/" + audio_wav_name

    frame_num = 0
    first_frame = True
    padder = None

    rgb_stack = []
    stack_size = 30

    frame_exist, frame_num, rgb_ori, input_data = img_buffer.get_img_data()

    if frame_exist:
        while True:
            t0 = time.time()
            frame_exist, frame_num, rgb_ori, input_data = img_buffer.get_img_data()

            if frame_exist == 0:
                print("End of video")
                break

            if len(input_data): 
                input_data_tensor = torch.cat(input_data)
                input_audio_data = audio_data[int((frame_num - 30) * audio_data_per_frame) : int((frame_num) * audio_data_per_frame)]
                # print(input_data_tensor.size())
                # print(input_audio_data.shape)
            

            # if padder is None:
            #     padder = InputPadder(input_data[0].shape)
            #     model_feature.padder = padder

            # with torch.no_grad():
            #     if len(input_data) == stack_size:
            #         rgb_stack_input = torch.cat(input_data).to(device)
            #         result = model_feature(rgb_stack_input)
            #          # feature : rgb, flow, audio
            #         feature = [result[0].squeeze(0),result[1].squeeze(0),result[2].squeeze(0)]

            #         y_pred = model([torch.unsqueeze(feature[0], dim = 0), torch.unsqueeze(feature[1], dim = 0), torch.unsqueeze(feature[2], dim = 0)])


                    # rgb_stack = rgb_stack[1:]
                    

            # print("FPS: ", 1/(time.time() - t0))

            # cv2.imshow("img", rgb_ori)
            # cv2.waitKey(1)

