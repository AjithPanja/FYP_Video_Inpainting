import torch
import cv2
import numpy as np
import torch.utils.data


class FlowInfer(torch.utils.data.Dataset):

    def __init__(self, list_file, size=None, isRGB=True, start_pos=0):
        super(FlowInfer, self).__init__()
        self.size = size
        txt_file = open(list_file, 'r')
        self.frame1_list = []
        self.frame2_list = []
        self.frame3_list = []
        self.frame4_list = []
        self.frame5_list = []
        self.output_list_1 = []
        self.output_list_2 = []
        self.output_list_3 = []
        self.output_list_4 = []
        self.isRGB = isRGB

        for line in txt_file:
            line = line.strip(' ')
            line = line.strip('\n')

            line_split = line.split(' ')
            self.frame1_list.append(line_split[0])
            self.frame2_list.append(line_split[1])
            self.frame3_list.append(line_split[2])
            self.frame4_list.append(line_split[3])
            self.frame5_list.append(line_split[4])
            self.output_list_1.append(line_split[5])
            self.output_list_2.append(line_split[6])
            self.output_list_3.append(line_split[7])
            self.output_list_4.append(line_split[8])

        if start_pos > 0:
            self.frame1_list = self.frame1_list[start_pos:]
            self.frame2_list = self.frame2_list[start_pos:]
            self.frame3_list = self.frame3_list[start_pos:]
            self.frame4_list = self.frame4_list[start_pos:]
            self.frame5_list = self.frame5_list[start_pos:]
            self.output_list_1 = self.output_list_1[start_pos:]
            self.output_list_2 = self.output_list_2[start_pos:]
            self.output_list_3 = self.output_list_3[start_pos:]
            self.output_list_4 = self.output_list_4[start_pos:]
        txt_file.close()

    def __len__(self):
        return len(self.frame1_list)

    def __getitem__(self, idx):
        frame1 = cv2.imread(self.frame1_list[idx])
        frame2 = cv2.imread(self.frame2_list[idx])
        frame3 = cv2.imread(self.frame3_list[idx])
        frame4 = cv2.imread(self.frame4_list[idx])
        frame5 = cv2.imread(self.frame5_list[idx])
        if self.isRGB:
            frame1 = frame1[:, :, ::-1]
            frame2 = frame2[:, :, ::-1]
            frame3 = frame3[:, :, ::-1]
            frame4 = frame4[:, :, ::-1]
            frame5 = frame5[:, :, ::-1]
        output_path_1 = self.output_list_1[idx]
        output_path_2 = self.output_list_2[idx]
        output_path_3 = self.output_list_3[idx]
        output_path_4 = self.output_list_4[idx]

        frame1 = self._img_tf(frame1)
        frame2 = self._img_tf(frame2)
        frame3 = self._img_tf(frame3)
        frame4 = self._img_tf(frame4)
        frame5 = self._img_tf(frame5)
        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).contiguous().float()
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).contiguous().float()
        frame3_tensor = torch.from_numpy(frame3).permute(2, 0, 1).contiguous().float()
        frame4_tensor = torch.from_numpy(frame4).permute(2, 0, 1).contiguous().float()
        frame5_tensor = torch.from_numpy(frame5).permute(2, 0, 1).contiguous().float()

        return frame1_tensor, frame2_tensor, frame3_tensor,frame4_tensor,frame5_tensor,output_path_1,output_path_2,output_path_3,output_path_4


    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))

        return img
