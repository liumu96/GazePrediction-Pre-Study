from torch import nn
import torch
from model import graph_convolution_network
import torch.nn.functional as F

class attended_hand_recognition(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.body_joint_number = opt.body_joint_number
        self.hand_joint_number = opt.hand_joint_number 
        self.joint_number = self.body_joint_number + self.hand_joint_number
        self.input_n = opt.seq_len
        gcn_latent_features = opt.gcn_latent_features
        residual_gcns_num = opt.residual_gcns_num
        gcn_dropout = opt.gcn_dropout
        head_cnn_channels = opt.head_cnn_channels
        recognition_cnn_channels = opt.recognition_cnn_channels

        # 1D CNN for extracting features from head directions
        in_channels_head = 3
        cnn_kernel_size = 3
        cnn_padding = (cnn_kernel_size - 1) // 2
        out_channels_1_head = head_cnn_channels
        out_channels_2_head = head_cnn_channels
        out_channels_head = head_cnn_channels

        self.head_cnn = nn.Sequential(
            nn.Conv1d(in_channels_head, out_channels_1_head, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_1_head, self.input_n]),
            nn.Tanh(),
            nn.Conv1d(out_channels_1_head, out_channels_2_head, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_2_head, self.input_n]),
            nn.Tanh(),
            nn.Conv1d(out_channels_2_head, out_channels_head, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.Tanh()
        )

        # GCN for extracting features from body and left hand joints
        self.left_hand_gcn = graph_convolution_network.graph_convolution_network(in_features=3,
                                                                                 latent_features=gcn_latent_features,
                                                                                 node_n=self.joint_number,
                                                                                 seq_len=self.input_n,
                                                                                 p_dropout=gcn_dropout,
                                                                                 residual_gcns_num=residual_gcns_num)
        # GCN for extracting features from right hand joints
        self.right_hand_gcn = graph_convolution_network.graph_convolution_network(in_features=3,
                                                                                  latent_features=gcn_latent_features,
                                                                                  node_n=self.joint_number,
                                                                                  seq_len=self.input_n,
                                                                                  p_dropout=gcn_dropout,
                                                                                  residual_gcns_num=residual_gcns_num)
        # 1D CNN for recognizing attended hand (left or right)
        in_channels_recognition = self.joint_number * gcn_latent_features * 2 + out_channels_head
        cnn_kernel_size = 3
        cnn_padding = (cnn_kernel_size - 1) // 2
        out_channels_1_recognition = recognition_cnn_channels
        out_channels_recognition = 2

        self.recognition_cnn = nn.Sequential(
            nn.Conv1d(in_channels_recognition, out_channels_1_recognition, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_1_recognition, self.input_n]),
            nn.Tanh(),
            nn.Conv1d(out_channels_1_recognition, out_channels_recognition, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
        )

    def forward(self, src, input_n=15):
        bs, seq_len, features = src.shape
        body_joints = src.clone()[:, :, :self.body_joint_number*3]
        left_hand_joints = src.clone()[:, :, self.body_joint_number*3:(self.body_joint_number+self.hand_joint_number)*3]
        right_hand_joints = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number)*3:(self.body_joint_number+self.hand_joint_number*2)*3]
        head_directions = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number * 2)*3:(self.body_joint_number+self.hand_joint_number * 2 + 1)*3]

        left_hand_joints = torch.cat((left_hand_joints, body_joints), dim=2)
        left_hand_joints = left_hand_joints.permute(0, 2, 1).reshape(bs, -1, 3, input_n).permute(0, 2, 1, 3)
        left_hand_features = self.left_hand_gcn(left_hand_joints)
        left_hand_features = left_hand_features.permute(0, 2, 1, 3).reshape(bs, -1, input_n)

        right_hand_joints = torch.cat((right_hand_joints, body_joints), dim=2)
        right_hand_joints = right_hand_joints.permute(0, 2, 1).reshape(bs, -1, 3, input_n).permute(0, 2, 1, 3)
        right_hand_features = self.right_hand_gcn(right_hand_joints)
        right_hand_features = right_hand_features.permute(0, 2, 1, 3).reshape(bs, -1, input_n)

        head_direction = head_directions.permute(0, 2, 1)
        head_features = self.head_gcn(head_direction)

        # fuse head and hand features
        features = torch.cat((left_hand_features, right_hand_features), dim=1)
        features = torch.cat((features, head_features), dim=1)
        # recognize attended hand from fused features
        recognition = self.recognition_cnn(features).permute(0, 2, 1)
        
        return recognition
