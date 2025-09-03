from torch import nn
import torch
from model import graph_convolution_network, transformer
import torch.nn.functional as F

class gaze_estimation(nn.Module):
    def __init__(self, opt):

        super().__init__()
        self.opt = opt
        self.body_joint_number = opt.body_joint_number
        self.hand_joint_number = opt.hand_joint_number
        self.input_n = opt.seq_len
        self.object_num = opt.object_num
        gcn_latent_features = opt.gcn_latent_features
        residual_gcns_num = opt.residual_gcns_num
        gcn_dropout = opt.gcn_dropout
        head_cnn_channels = opt.head_cnn_channels
        gaze_cnn_channels = opt.gaze_cnn_channels
        self.use_self_att = opt.use_self_att
        self_att_head_num = opt.self_att_head_num
        self_att_dropout = opt.self_att_dropout
        self.use_cross_att = opt.use_cross_att
        cross_att_head_num = opt.cross_att_head_num
        cross_att_dropout = opt.cross_att_dropout
        self.use_attended_hand = opt.use_attended_hand
        self.use_attended_hand_gt = opt.use_attended_hand_gt
        if self.use_attended_hand:
            self.joint_number = self.body_joint_number + self.hand_joint_number + self.object_num
        else:
            self.joint_number = self.body_joint_number + self.hand_joint_number * 2 + self.object_num * 2

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

        # GCN extracting features from hand joints, body joints, and scene objects
        self.hand_gcn = graph_convolution_network.graph_convolution_network(in_features=3,
                                                                            latent_features=gcn_latent_features,
                                                                            node_n=self.joint_number,
                                                                            seq_len=self.input_n,
                                                                            p_dropout=gcn_dropout,
                                                                            residual_num=residual_gcns_num)
        if self.use_self_att:
            self.head_self_att = transformer.Temporal_Self_Attention(out_channels_head, self_att_head_num, self_att_dropout)
            self.hand_self_att = transformer.Temporal_Self_Attention(self.joint_number*gcn_latent_features, self_att_head_num, self_att_dropout)

        if self.use_cross_att:
            self.head_hand_cross_att = transformer.Temporal_Cross_Attention(out_channels_head, self.joint_number*gcn_latent_features, cross_att_head_num, cross_att_dropout)
            self.hand_head_cross_att = transformer.Temporal_Cross_Attention(self.joint_number*gcn_latent_features, out_channels_head, cross_att_head_num, cross_att_dropout)

        # 1D CNN for estimating eye gaze
        in_channels_gaze = self.joint_number * gcn_latent_features + out_channels_head
        cnn_kernel_size = 3
        cnn_padding = (cnn_kernel_size - 1) // 2
        out_channels_1_gaze = gaze_cnn_channels
        out_channels_gaze = 3

        self.gaze_cnn = nn.Sequential(
            nn.Conv1d(in_channels_gaze, out_channels_1_gaze, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_1_gaze, self.input_n]),
            nn.Tanh(),
            nn.Conv1d(out_channels_1_gaze, out_channels_gaze, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
            nn.Tanh()
        )

    def forward(self, src, input_n=15):
        bs, seq_len, features = src.shape
        body_joints = src.clone()[:, :, :self.body_joint_number*3]
        left_hand_joints = src.clone()[:, :, self.body_joint_number*3:(self.body_joint_number+self.hand_joint_number)*3]
        right_hand_joints = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number)*3:(self.body_joint_number+self.hand_joint_number*2)*3]
        head_direction = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number*2)*3:(self.body_joint_number+self.hand_joint_number*2+1)*3]

        if self.object_num > 0:
            left_object_position = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number*2+1)*3:(self.body_joint_number+self.hand_joint_number*2+1+self.object_num * 8)*3]
            left_object_position = torch.mean(left_object_position.reshape(bs, seq_len, self.object_num, 8, 3), dim=3).reshape(bs, seq_len, self.object_num * 3)
            right_object_position = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number*2+1+self.object_num * 8)*3:(self.body_joint_number+self.hand_joint_number*2+1+self.object_num * 8 *2)*3]
            right_object_position = torch.mean(right_object_position.reshape(bs, seq_len, self.object_num, 8, 3), dim=3).reshape(bs, seq_len, self.object_num * 3)
        
        attended_hand_prd = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number*2+1+8*self.object_num*2)*3:(self.body_joint_number+self.hand_joint_number*2+1+8*self.object_num*2)*3+2]        
        left_hand_weights = torch.round(attended_hand_prd[:, :, 0:1])
        right_hand_weights = torch.round(attended_hand_prd[:, :, 1:2])                
        if self.use_attended_hand_gt:
            attended_hand_gt = src.clone()[:, :, (self.body_joint_number+self.hand_joint_number*2+1+8*self.object_num*2)*3+2:(self.body_joint_number+self.hand_joint_number*2+1+8*self.object_num*2)*3+3]                    
            left_hand_weights = 1-attended_hand_gt
            right_hand_weights = attended_hand_gt

        if self.use_attended_hand:
            hand_joints = left_hand_joints * left_hand_weights + right_hand_joints * right_hand_weights
        else:
            hand_joints = torch.cat((left_hand_joints, right_hand_joints), dim=2)
        hand_joints = torch.cat((hand_joints, body_joints), dim=2)

        if self.object_num > 0:
            if self.use_attended_hand:
                object_position = left_object_position*left_hand_weights + right_object_position*right_hand_weights
            else:
                object_position = torch.cat((left_object_position, right_object_position), dim=2)
            hand_joints = torch.cat((hand_joints, object_position), dim=2)
        
        hand_joints = hand_joints.permute(0, 2, 1).reshape(bs, -1, 3, input_n).permute(0, 2, 1, 3)
        hand_features = self.hand_gcn(hand_joints)
        hand_features = hand_features.permute(0, 2, 1, 3).reshape(bs, -1, input_n) 

        head_direction = head_direction.permute(0, 2, 1)
        head_features = self.head_cnn(head_direction)

        if self.use_self_att:
            head_features = self.head_self_att(head_features.permute(0, 2, 1)).permute(0, 2, 1)
            hand_features = self.hand_self_att(hand_features.permute(0, 2, 1)).permute(0, 2, 1)

        if self.use_cross_att:
            head_features_copy = head_features.clone()
            head_features = self.cross_att(head_features.permute(0, 2, 1), hand_features.permute(0, 2, 1)).permute(0, 2, 1)
            hand_features = self.cross_att(hand_features.permute(0, 2, 1), head_features_copy.permute(0, 2, 1)).permute(0, 2, 1)

        # fuse head and hand features
        features = torch.cat((hand_features, head_features), dim=1)
        # estimate eye gaze
        prediction = self.gaze_cnn(features).permute(0, 2, 1)
        # normalize to unit vectors
        prediction = F.normalize(prediction, dim=2)

        return prediction

