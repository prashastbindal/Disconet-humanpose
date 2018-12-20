from src.model import Linear
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Uniform
import torch
from torch.nn import functional as F
import ipdb

def loss_function(y, y_gt):
    y_len = len(y)

    loss_regen = 0
    for i in range(y_len):
        loss_regen += F.mse_loss(y[i], y_gt, size_average=False)

    loss_regen = loss_regen / y_len

    loss_dissim_vec = Variable(torch.zeros_like(y[0]))
    for i in range(y_len):
        for j in range(i + 1, y_len):
            if i != j:
                loss_dissim_vec += (y[i] - y[j]) ** 2

    loss_dissim = torch.sum(loss_dissim_vec / (y_len * (y_len - 1) / 2))

    L2 = loss_regen - 0.5 * loss_dissim

    return L2,  loss_regen, loss_dissim

class DiscoNetModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=1,
                 p_dropout=0.5):
        super(DiscoNetModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = 16 * 2
        # 3d joints
        self.output_size = 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.mid_process = nn.Linear(self.linear_size, self.linear_size // 2)

        self.linear_stages2 = []
        for l in range(num_stage):
            self.linear_stages2.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages2 = nn.ModuleList(self.linear_stages2)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        self.z_dist = torch.Tensor(2, 2).uniform_(0, 1)

    def reparameterize(self, mu, logvar):
        # logic is same at train and test time
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers


        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y_out = self.mid_process(y)
        pose_3d_samples = []
        for _ in range(10):
            z = torch.Tensor(y_out.size()).uniform_(0, 1).cuda()
            mid = torch.cat([y_out, z], dim=1)

            for i in range(self.num_stage):
                mid = self.linear_stages2[i](mid)

            pose_3d_samples.append( self.w2(mid))

        return pose_3d_samples
