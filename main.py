import torch
import cv2
import torchvision
import time
import os

from models.transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop,
                               ToFloat, ToUInt8)

from models.raft.raft_src.raft import RAFT, InputPadder
from models.vggish.vggish_src.vggish_input import waveform_to_examples

from custom_model import Img_Audio_Feature_Extraction, Action_Classification_Model

from multiprocessing import Process, Pipe, Manager, Queue
from multiprocessing.managers import BaseManager

import numpy as np
import soundfile as sf

class Img_Buffer(object):
    def __init__(self):
        self.img_data = None
        self.img_success_flag = True
        self.img_buffer = []
        self.audio_buffer = []
        self.frame_num = 0
        self.predict_result = 0

    def set_input_data(self, data):
        # self.frame_num = data[0]
        # self.img_data = data[1]
        # self.img_buffer.append(data[2])
        # self.audio_buffer = data[3]

        self.frame_num = data[0]
        # self.img_data = data[1]
        self.img_buffer.append(data[1])
        self.audio_buffer = data[2]


        if len(self.img_buffer) > 30:
            self.img_buffer.pop(0)

    def set_img_connect_flag(self, data):
        self.img_success_flag = data
    
    def get_img_data(self):
        # return self.img_success_flag, self.frame_num, self.img_data, self.img_buffer, self.audio_buffer
        return self.img_success_flag, self.frame_num, self.img_buffer, self.audio_buffer
    
    def get_img_buffer(self):
        return self.img_buffer

    def set_predict_result(self, data):
        self.predict_result = data

    def get_predict_result(self):
        return self.predict_result

def img_processing(pic):
    image = cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2BGR)

    img = torch.from_numpy(np.array(pic, copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))

def video_capture(data_buffer, video_source, audio_source):

    audio_data = wavfile_to_examples(audio_source)

    audio_sample_rate = 44100
    video_fps = 30

    audio_data_per_frame = int(audio_sample_rate / video_fps)


    cap = cv2.VideoCapture(video_source)
    frame_num = 0

    while True:
        ret, rgb_ori = cap.read()
        
        frame_exists = True

        t0 = time.time()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        normal = "Normal"
        abnormal = "Accident"

        font_scale = 1.5
        font_thickness = 2
        c1, c2 = (int(0), int(0)), (int(300), int(80))

        audio_buffer = []
        rgb = []

        if ret:
            data_buffer.set_img_connect_flag(frame_exists)

            frame_num += 1

            if frame_num >= 30:
                audio_buffer = audio_data[int((frame_num - 30) * audio_data_per_frame) : int((frame_num) * audio_data_per_frame)]

            data_buffer.set_input_data([frame_num, rgb_ori, audio_buffer])

            status = data_buffer.get_predict_result()

            if status > 0.9:
                cv2.rectangle(rgb_ori, c1, c2, (0, 0 ,255), thickness=-1, lineType=cv2.LINE_AA)

                cv2.putText(rgb_ori, abnormal,
                (int(c2[0]/6.5) , int(c2[1]/1.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),   # 글자 색상 (흰색)
                font_thickness)

            else:
                cv2.rectangle(rgb_ori, c1, c2, (0, 255 ,0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(rgb_ori, normal,
                (int(c2[0]/6.5) , int(c2[1]/1.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),   # 글자 색상 (흰색)
                font_thickness)

            # cv2.imshow("img", rgb_ori)
            cv2.imshow("img", cv2.resize(rgb_ori, (1280, 720)))

            key = cv2.waitKey(30) # fps : 30

            # print("frame_num : ", frame_num, "fps : ", 1 / (time.time() - t0))
            
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                frame_exists = False
                data_buffer.set_img_connect_flag(frame_exists)


        else:
            frame_exists = False
            data_buffer.set_img_connect_flag(frame_exists)
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

    video_name = "test_11"

    video_path = f"./videos/{video_name}.mp4"
    audio_path = f"./audios/{video_name}.wav"
    
    I3D_weight_path = {"rgb" : "./models/i3d/checkpoints/i3d_rgb.pt",
                    "flow" : "./models/i3d/checkpoints/i3d_flow.pt"}
    
    RAFT_weight_path = "./models/raft/checkpoints/raft-sintel.pth"
    VGGISH_weight_path = "./models/vggish/checkpoints/vggish-10086976.pth"

    ACM_weight_path = "./models/custom/e100_Adam_0.048791107.pt"

    stack_size = 30
    audio_sample_rate = 44100
    min_side_size = 256

    model_feature = Img_Audio_Feature_Extraction(I3D_weight_path, RAFT_weight_path, audio_path = None, img_stack_size = stack_size, device=device)
    # model_feature.eval()
    model = Action_Classification_Model(device).to(device)
    model.load_state_dict(torch.load(ACM_weight_path))
    model.eval()

    resize_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                    ResizeImproved(min_side_size),
                                                    PILToTensor(),
                                                    ToFloat(),])

    BaseManager.register('img_buffer', Img_Buffer)
    manager = BaseManager()
    manager.start()

    data_buffer = manager.img_buffer()
    img_thread = Process(target=video_capture, args=[data_buffer, video_path, audio_path])
    img_thread.start()

    # audio_wav_name = video_name.split(".")[0] + ".wav"
    # audio_wav_path = "./dataset/train/audios/normal/" + audio_wav_name
    # audio_wav_path = "./dataset/train/audios/abnormal/" + audio_wav_name

    frame_num = 0
    first_frame = True
    padder = None

    frame_exist, frame_num, image_buffer, audio_buffer = data_buffer.get_img_data()

    try:
        if frame_exist:
            while True:
                t0 = time.time()
                frame_exist, frame_num, image_buffer, audio_buffer = data_buffer.get_img_data()

                if frame_exist == 0:
                    print("End of video")
                    break

                if len(image_buffer) == 30:
                    for i in range(len(image_buffer)):
                        rgb = cv2.cvtColor(image_buffer[i], cv2.COLOR_BGR2RGB)
                        rgb = resize_transforms(rgb)
                        image_buffer[i] = rgb.unsqueeze(0)

                    if padder is None:
                        padder = InputPadder(image_buffer[0].shape)
                        model_feature.padder = padder

                    input_data_tensor = torch.cat(image_buffer).to(device)
                    with torch.no_grad():
                        result = model_feature(input_data_tensor, audio_buffer)
                        y_pred = model(result)

                        data_buffer.set_predict_result(y_pred.cpu())

                        # y_pred = model([torch.unsqueeze(result[0], dim = 0), torch.unsqueeze(result[1], dim = 0), torch.unsqueeze(result[2], dim = 0)])
                        print(y_pred)
                # print("FPS: ", 1/(time.time() - t0))

                # cv2.imshow("img", rgb_ori)
                # cv2.waitKey(1)

    except KeyboardInterrupt:
        del model_feature
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        img_thread.terminate()

        print('Ctrl + C 중지 메시지 출력')