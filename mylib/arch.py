import sys
from torch import torch, cat, add, nn


#kaiming init
def kaiming_w_init(layer, a=0, nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight)
    layer.bias.data.fill_(0.01)

#conv block 2x conv+BN+relu
class convBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU() 
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,  stride=1, padding=1, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(mid_channels) 
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,  stride=1, padding=1, padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(out_channels) 
        #weights initialization
        kaiming_w_init(self.conv1)
        kaiming_w_init(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


#MODEL FOR CARLA DATASET
class E0(nn.Module): 
    def __init__(self, in_channel_dim=[2, 3, 15]): 
        super().__init__()
        #OTHERS
        n_fmap_ch = [16, 32, 64, 128]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.drop = nn.Dropout(p=0.5)
        self.act_dep = nn.ReLU()
        self.act_seg = nn.Sigmoid()
        self.act_lidseg = nn.Sigmoid()
        self.act_bir = nn.Sigmoid()

        #DOWNBLOCK
        #BGR
        self.conv0_0_bgr_f = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_l = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_ri = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_r = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_bgr_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_bgr = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #DVS
        self.conv0_0_dvs_f = convBlock(in_channel_dim[0], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dvs_l = convBlock(in_channel_dim[0], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dvs_ri = convBlock(in_channel_dim[0], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dvs_r = convBlock(in_channel_dim[0], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_dvs_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dvs_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dvs_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dvs_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_dvs = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #LIDAR
        self.conv0_0_lid_t = convBlock(in_channel_dim[2], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_lid_t = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_lid = convBlock((1*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])

        #BOTTLENECK1
        self.conv3_0_bgrdvslid = convBlock((3*n_fmap_ch[2]), n_fmap_ch[3], n_fmap_ch[3])
        
        #DE
        self.conv2_1_dep = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_dep_f = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_l = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_ri = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_r = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_dep_f = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_l = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_ri = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_r = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        #SS
        self.conv2_1_seg = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_seg_f = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_l = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_ri = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_r = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_seg_f = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_l = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_ri = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_r = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        #LS
        self.conv2_1_lidseg = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_lidseg_t = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_lidseg_t = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        
        #FINAL POINTWISE CONV
        self.final_dep_f = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_l = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_ri = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_r = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_seg_f = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_l = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_ri = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_r = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_lidseg_t = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1) 

        #DOWNBLOCK2
        #DE
        self.conv0_0_dep_f = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_l = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_ri = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_r = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_dep_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_dep = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #SS
        self.conv0_0_seg_f = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_l = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_ri = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_r = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_seg_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_seg = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #LS
        self.conv0_0_lidseg_t = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_lidseg_t = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_lidseg = convBlock((1*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
     
        #BOTTLENECK2
        self.conv3_0_depseglidseg = convBlock((3*n_fmap_ch[2]), n_fmap_ch[3], n_fmap_ch[3])
        
        #UPBLOCK
        #BEVP
        self.conv2_1_bir = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_bir_t = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_bir_t = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])

        #FINAL POINTWISE CONV2
        self.final_bir_t = nn.Conv2d(n_fmap_ch[0], 9, kernel_size=1) 

        #INIT OTHER LAYERS
        kaiming_w_init(self.final_dep_f)
        kaiming_w_init(self.final_dep_l)
        kaiming_w_init(self.final_dep_ri)
        kaiming_w_init(self.final_dep_r)
        kaiming_w_init(self.final_seg_f)
        kaiming_w_init(self.final_seg_l)
        kaiming_w_init(self.final_seg_ri)
        kaiming_w_init(self.final_seg_r)
        kaiming_w_init(self.final_lidseg_t)
        kaiming_w_init(self.final_bir_t)
       

    def forward(self, input):
        #DOWNSAMPLING
        dvs_f_0_0 = self.conv0_0_dvs_f(input[0])
        dvs_l_0_0 = self.conv0_0_dvs_l(input[1])
        dvs_ri_0_0 = self.conv0_0_dvs_ri(input[2])
        dvs_r_0_0 = self.conv0_0_dvs_r(input[3])
        bgr_f_0_0 = self.conv0_0_bgr_f(input[4])
        bgr_l_0_0 = self.conv0_0_bgr_l(input[5])
        bgr_ri_0_0 = self.conv0_0_bgr_ri(input[6])
        bgr_r_0_0 = self.conv0_0_bgr_r(input[7])
        lid_t_0_0 = self.conv0_0_lid_t(input[8])
        down_bgr_f_0_0 = self.pool(bgr_f_0_0)
        down_bgr_l_0_0 = self.pool(bgr_l_0_0)
        down_bgr_ri_0_0 = self.pool(bgr_ri_0_0)
        down_bgr_r_0_0 = self.pool(bgr_r_0_0)
        down_dvs_f_0_0 = self.pool(dvs_f_0_0)
        down_dvs_l_0_0 = self.pool(dvs_l_0_0)
        down_dvs_ri_0_0 = self.pool(dvs_ri_0_0)
        down_dvs_r_0_0 = self.pool(dvs_r_0_0)
        down_lid_t_0_0 = self.pool(lid_t_0_0)

        bgr_f_1_0 = self.conv1_0_bgr_f(down_bgr_f_0_0)
        bgr_l_1_0 = self.conv1_0_bgr_l(down_bgr_l_0_0)
        bgr_ri_1_0 = self.conv1_0_bgr_ri(down_bgr_ri_0_0)
        bgr_r_1_0 = self.conv1_0_bgr_r(down_bgr_r_0_0)
        dvs_f_1_0 = self.conv1_0_dvs_f(down_dvs_f_0_0)
        dvs_l_1_0 = self.conv1_0_dvs_l(down_dvs_l_0_0)
        dvs_ri_1_0 = self.conv1_0_dvs_ri(down_dvs_ri_0_0)
        dvs_r_1_0 = self.conv1_0_dvs_r(down_dvs_r_0_0)
        lid_t_1_0 = self.conv1_0_lid_t(down_lid_t_0_0)
        down_bgr_f_1_0 = self.pool(bgr_f_1_0)
        down_bgr_l_1_0 = self.pool(bgr_l_1_0)
        down_bgr_ri_1_0 = self.pool(bgr_ri_1_0)
        down_bgr_r_1_0 = self.pool(bgr_r_1_0)
        down_dvs_f_1_0 = self.pool(dvs_f_1_0)
        down_dvs_l_1_0 = self.pool(dvs_l_1_0)
        down_dvs_ri_1_0 = self.pool(dvs_ri_1_0)
        down_dvs_r_1_0 = self.pool(dvs_r_1_0)
        down_lid_t_1_0 = self.pool(lid_t_1_0)

        #FUSE ALL FEATURES
        cat_in_bgr = cat([down_bgr_f_1_0, down_bgr_l_1_0, down_bgr_ri_1_0, down_bgr_r_1_0], dim=1)
        cat_in_dvs = cat([down_dvs_f_1_0, down_dvs_l_1_0, down_dvs_ri_1_0, down_dvs_r_1_0], dim=1)
        bgr_2_0 = self.drop(self.conv2_0_bgr(cat_in_bgr))
        dvs_2_0 = self.drop(self.conv2_0_dvs(cat_in_dvs))
        lid_2_0 = self.drop(self.conv2_0_lid(down_lid_t_1_0))
        down_bgr_2_0 = self.pool(bgr_2_0) 
        down_dvs_2_0 = self.pool(dvs_2_0)
        down_lid_2_0 = self.pool(lid_2_0) 

        #BOTTLENECK 1
        cat_in_bgrdvslid = cat([down_bgr_2_0, down_dvs_2_0, down_lid_2_0], dim=1)
        bgrdvslid_3_0 = self.drop(self.conv3_0_bgrdvslid(cat_in_bgrdvslid))
        up_bgrdvslid_3_0 = self.up(bgrdvslid_3_0)      

        #UPSAMPLING    
        dep_2_1 = self.drop(self.conv2_1_dep(up_bgrdvslid_3_0))
        seg_2_1 = self.drop(self.conv2_1_seg(up_bgrdvslid_3_0))
        lidseg_2_1 = self.drop(self.conv2_1_lidseg(up_bgrdvslid_3_0))
        up_dep_2_1 = self.up(dep_2_1)
        up_seg_2_1 = self.up(seg_2_1)
        up_lidseg_2_1 = self.up(lidseg_2_1)

        dep_f_1_2 = self.conv1_2_dep_f(cat([up_dep_2_1, dvs_f_1_0], dim=1))
        dep_l_1_2 = self.conv1_2_dep_l(cat([up_dep_2_1, dvs_l_1_0], dim=1))
        dep_ri_1_2 = self.conv1_2_dep_ri(cat([up_dep_2_1, dvs_ri_1_0], dim=1))
        dep_r_1_2 = self.conv1_2_dep_r(cat([up_dep_2_1, dvs_r_1_0], dim=1))
        seg_f_1_2 = self.conv1_2_seg_f(cat([up_seg_2_1, bgr_f_1_0], dim=1))
        seg_l_1_2 = self.conv1_2_seg_l(cat([up_seg_2_1, bgr_l_1_0], dim=1))
        seg_ri_1_2 = self.conv1_2_seg_ri(cat([up_seg_2_1, bgr_ri_1_0], dim=1))
        seg_r_1_2 = self.conv1_2_seg_r(cat([up_seg_2_1, bgr_r_1_0], dim=1))
        lidseg_t_1_2 = self.conv1_2_lidseg_t(cat([up_lidseg_2_1, lid_t_1_0], dim=1))
        up_dep_f_1_2 = self.up(dep_f_1_2)
        up_dep_l_1_2 = self.up(dep_l_1_2)
        up_dep_ri_1_2 = self.up(dep_ri_1_2)
        up_dep_r_1_2 = self.up(dep_r_1_2)
        up_seg_f_1_2 = self.up(seg_f_1_2)
        up_seg_l_1_2 = self.up(seg_l_1_2)
        up_seg_ri_1_2 = self.up(seg_ri_1_2)
        up_seg_r_1_2 = self.up(seg_r_1_2)
        up_lidseg_t_1_2 = self.up(lidseg_t_1_2)

        dep_f_0_3 = self.conv0_3_dep_f(cat([up_dep_f_1_2, dvs_f_0_0], dim=1))
        dep_l_0_3 = self.conv0_3_dep_l(cat([up_dep_l_1_2, dvs_l_0_0], dim=1))
        dep_ri_0_3 = self.conv0_3_dep_ri(cat([up_dep_ri_1_2, dvs_ri_0_0], dim=1))
        dep_r_0_3 = self.conv0_3_dep_r(cat([up_dep_r_1_2, dvs_r_0_0], dim=1))
        seg_f_0_3 = self.conv0_3_seg_f(cat([up_seg_f_1_2, bgr_f_0_0], dim=1))
        seg_l_0_3 = self.conv0_3_seg_l(cat([up_seg_l_1_2, bgr_l_0_0], dim=1))
        seg_ri_0_3 = self.conv0_3_seg_ri(cat([up_seg_ri_1_2, bgr_ri_0_0], dim=1))
        seg_r_0_3 = self.conv0_3_seg_r(cat([up_seg_r_1_2, bgr_r_0_0], dim=1))
        lidseg_t_0_3 = self.conv0_3_lidseg_t(cat([up_lidseg_t_1_2, lid_t_0_0], dim=1))
        out_dep_f = self.act_dep(self.final_dep_f(dep_f_0_3))
        out_dep_l = self.act_dep(self.final_dep_l(dep_l_0_3))
        out_dep_ri = self.act_dep(self.final_dep_ri(dep_ri_0_3))
        out_dep_r = self.act_dep(self.final_dep_r(dep_r_0_3))
        out_seg_f = self.act_seg(self.final_seg_f(seg_f_0_3))
        out_seg_l = self.act_seg(self.final_seg_l(seg_l_0_3))
        out_seg_ri = self.act_seg(self.final_seg_ri(seg_ri_0_3))
        out_seg_r = self.act_seg(self.final_seg_r(seg_r_0_3))
        out_lidseg_t = self.act_lidseg(self.final_lidseg_t(lidseg_t_0_3))

        #DOWNSAMPLING
        dep_f_0_0 = self.conv0_0_dep_f(out_dep_f)
        dep_l_0_0 = self.conv0_0_dep_l(out_dep_l)
        dep_ri_0_0 = self.conv0_0_dep_ri(out_dep_ri)
        dep_r_0_0 = self.conv0_0_dep_r(out_dep_r)
        seg_f_0_0 = self.conv0_0_seg_f(out_seg_f)
        seg_l_0_0 = self.conv0_0_seg_l(out_seg_l)
        seg_ri_0_0 = self.conv0_0_seg_ri(out_seg_ri)
        seg_r_0_0 = self.conv0_0_seg_r(out_seg_r)
        lidseg_t_0_0 = self.conv0_0_lidseg_t(out_lidseg_t) 
        down_dep_f_0_0 = self.pool(dep_f_0_0)
        down_dep_l_0_0 = self.pool(dep_l_0_0)
        down_dep_ri_0_0 = self.pool(dep_ri_0_0)
        down_dep_r_0_0 = self.pool(dep_r_0_0)
        down_seg_f_0_0 = self.pool(seg_f_0_0)
        down_seg_l_0_0 = self.pool(seg_l_0_0)
        down_seg_ri_0_0 = self.pool(seg_ri_0_0)
        down_seg_r_0_0 = self.pool(seg_r_0_0)
        down_lidseg_t_0_0 = self.pool(lidseg_t_0_0)

        dep_f_1_0 = self.conv1_0_dep_f(down_dep_f_0_0)
        dep_l_1_0 = self.conv1_0_dep_l(down_dep_l_0_0)
        dep_ri_1_0 = self.conv1_0_dep_ri(down_dep_ri_0_0)
        dep_r_1_0 = self.conv1_0_dep_r(down_dep_r_0_0)
        seg_f_1_0 = self.conv1_0_seg_f(down_seg_f_0_0)
        seg_l_1_0 = self.conv1_0_seg_l(down_seg_l_0_0)
        seg_ri_1_0 = self.conv1_0_seg_ri(down_seg_ri_0_0)
        seg_r_1_0 = self.conv1_0_seg_r(down_seg_r_0_0)
        lidseg_t_1_0 = self.conv1_0_lidseg_t(down_lidseg_t_0_0)
        down_dep_f_1_0 = self.pool(dep_f_1_0)
        down_dep_l_1_0 = self.pool(dep_l_1_0)
        down_dep_ri_1_0 = self.pool(dep_ri_1_0)
        down_dep_r_1_0 = self.pool(dep_r_1_0)
        down_seg_f_1_0 = self.pool(seg_f_1_0)
        down_seg_l_1_0 = self.pool(seg_l_1_0)
        down_seg_ri_1_0 = self.pool(seg_ri_1_0)
        down_seg_r_1_0 = self.pool(seg_r_1_0)
        down_lidseg_t_1_0 = self.pool(lidseg_t_1_0)

        #FUSE ALL FEATURES
        cat_in_dep = cat([down_dep_f_1_0, down_dep_l_1_0, down_dep_ri_1_0, down_dep_r_1_0], dim=1)
        cat_in_seg = cat([down_seg_f_1_0, down_seg_l_1_0, down_seg_ri_1_0, down_seg_r_1_0], dim=1)
        dep_2_0 = self.drop(self.conv2_0_dep(cat_in_dep))
        seg_2_0 = self.drop(self.conv2_0_seg(cat_in_seg))
        lidseg_2_0 = self.drop(self.conv2_0_lidseg(down_lidseg_t_1_0))
        down_dep_2_0 = self.pool(dep_2_0)
        down_seg_2_0 = self.pool(seg_2_0) 
        down_lidseg_2_0 = self.pool(lidseg_2_0) 

        #BOTTLENECK 2
        cat_in_depseglidseg = cat([down_dep_2_0, down_seg_2_0, down_lidseg_2_0], dim=1)
        depseglidseg_3_0 = self.drop(self.conv3_0_depseglidseg(cat_in_depseglidseg))
        up_depseglidseg_3_0 = self.up(depseglidseg_3_0)
       
        #UPSAMPLING
        bir_2_1 = self.drop(self.conv2_1_bir(up_depseglidseg_3_0))
        up_bir_2_1 = self.up(bir_2_1)

        bir_t_1_2 = self.conv1_2_bir_t(cat([up_bir_2_1, lidseg_t_1_0], dim=1))
        up_bir_t_1_2 = self.up(bir_t_1_2)

        bir_t_0_3 = self.conv0_3_bir_t(cat([up_bir_t_1_2, lidseg_t_0_0], dim=1))
        out_bir_t = self.act_bir(self.final_bir_t(bir_t_0_3))

        return [out_dep_f, out_dep_l, out_dep_ri, out_dep_r, out_seg_f, out_seg_l, out_seg_ri, out_seg_r, out_lidseg_t, out_bir_t]



#MODEL FOR NUSCENE DATASET
class E1(nn.Module): #
    def __init__(self, in_channel_dim=[2, 3, 15]):
        super().__init__()
        #OTHERS
        n_fmap_ch = [16, 32, 64, 128]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.drop = nn.Dropout(p=0.5)
        self.act_dep = nn.ReLU()
        self.act_seg = nn.Sigmoid()
        self.act_lidseg = nn.Sigmoid()
        self.act_bir = nn.Sigmoid()

        #DOWNBLOCK
        #BGR
        self.conv0_0_bgr_f = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_l = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_ri = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_bgr_r = convBlock(in_channel_dim[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_bgr_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_bgr_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_bgr = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #LIDAR
        self.conv0_0_lid_t = convBlock(in_channel_dim[2], n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_lid_t = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_lid = convBlock((1*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])

        #BOTTLENECK1
        self.conv3_0_bgrlid = convBlock((2*n_fmap_ch[2]), n_fmap_ch[3], n_fmap_ch[3])
        
        #DE
        self.conv2_1_dep = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_dep_f = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_l = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_ri = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_dep_r = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_dep_f = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_l = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_ri = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_dep_r = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        #SS
        self.conv2_1_seg = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_seg_f = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_l = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_ri = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_2_seg_r = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_seg_f = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_l = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_ri = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_3_seg_r = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        #LS
        self.conv2_1_lidseg = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_lidseg_t = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_lidseg_t = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])
        
        #FINAL POINTWISE CONV
        self.final_dep_f = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_l = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_ri = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_dep_r = nn.Conv2d(n_fmap_ch[0], 1, kernel_size=1)
        self.final_seg_f = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_l = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_ri = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_seg_r = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1)
        self.final_lidseg_t = nn.Conv2d(n_fmap_ch[0], 23, kernel_size=1) 

        #DOWNBLOCK2
        #DE
        self.conv0_0_dep_f = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_l = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_ri = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_dep_r = convBlock(1, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_dep_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_dep_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_dep = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #SS
        self.conv0_0_seg_f = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_l = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_ri = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv0_0_seg_r = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_seg_f = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_l = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_ri = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv1_0_seg_r = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_seg = convBlock((4*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
        #LS
        self.conv0_0_lidseg_t = convBlock(23, n_fmap_ch[0], n_fmap_ch[0])
        self.conv1_0_lidseg_t = convBlock(n_fmap_ch[0], n_fmap_ch[1], n_fmap_ch[1])
        self.conv2_0_lidseg = convBlock((1*n_fmap_ch[1]), n_fmap_ch[2], n_fmap_ch[2])
     
        #BOTTLENECK2
        self.conv3_0_depseglidseg = convBlock((3*n_fmap_ch[2]), n_fmap_ch[3], n_fmap_ch[3])
        
        #UPBLOCK
        #BEVP
        self.conv2_1_bir = convBlock(n_fmap_ch[3], n_fmap_ch[2], n_fmap_ch[2])
        self.conv1_2_bir_t = convBlock(n_fmap_ch[1]+n_fmap_ch[2], n_fmap_ch[1], n_fmap_ch[1])
        self.conv0_3_bir_t = convBlock(n_fmap_ch[0]+n_fmap_ch[1], n_fmap_ch[0], n_fmap_ch[0])

        #FINAL POINTWISE CONV2
        self.final_bir_t = nn.Conv2d(n_fmap_ch[0], 9, kernel_size=1) 

        #INIT OTHER LAYERS
        kaiming_w_init(self.final_dep_f)
        kaiming_w_init(self.final_dep_l)
        kaiming_w_init(self.final_dep_ri)
        kaiming_w_init(self.final_dep_r)
        kaiming_w_init(self.final_seg_f)
        kaiming_w_init(self.final_seg_l)
        kaiming_w_init(self.final_seg_ri)
        kaiming_w_init(self.final_seg_r)
        kaiming_w_init(self.final_lidseg_t)
        kaiming_w_init(self.final_bir_t)
       

    def forward(self, input):
        #DOWNSAMPLING
        # dvs_f_0_0 = self.conv0_0_dvs_f(input[0])
        # dvs_l_0_0 = self.conv0_0_dvs_l(input[1])
        # dvs_ri_0_0 = self.conv0_0_dvs_ri(input[2])
        # dvs_r_0_0 = self.conv0_0_dvs_r(input[3])
        bgr_f_0_0 = self.conv0_0_bgr_f(input[4])
        bgr_l_0_0 = self.conv0_0_bgr_l(input[5])
        bgr_ri_0_0 = self.conv0_0_bgr_ri(input[6])
        bgr_r_0_0 = self.conv0_0_bgr_r(input[7])
        lid_t_0_0 = self.conv0_0_lid_t(input[8])
        down_bgr_f_0_0 = self.pool(bgr_f_0_0)
        down_bgr_l_0_0 = self.pool(bgr_l_0_0)
        down_bgr_ri_0_0 = self.pool(bgr_ri_0_0)
        down_bgr_r_0_0 = self.pool(bgr_r_0_0)
        down_lid_t_0_0 = self.pool(lid_t_0_0)

        bgr_f_1_0 = self.conv1_0_bgr_f(down_bgr_f_0_0)
        bgr_l_1_0 = self.conv1_0_bgr_l(down_bgr_l_0_0)
        bgr_ri_1_0 = self.conv1_0_bgr_ri(down_bgr_ri_0_0)
        bgr_r_1_0 = self.conv1_0_bgr_r(down_bgr_r_0_0)
        lid_t_1_0 = self.conv1_0_lid_t(down_lid_t_0_0)
        down_bgr_f_1_0 = self.pool(bgr_f_1_0)
        down_bgr_l_1_0 = self.pool(bgr_l_1_0)
        down_bgr_ri_1_0 = self.pool(bgr_ri_1_0)
        down_bgr_r_1_0 = self.pool(bgr_r_1_0)
        down_lid_t_1_0 = self.pool(lid_t_1_0)

        #FUSE ALL FEATURES
        cat_in_bgr = cat([down_bgr_f_1_0, down_bgr_l_1_0, down_bgr_ri_1_0, down_bgr_r_1_0], dim=1)
        bgr_2_0 = self.drop(self.conv2_0_bgr(cat_in_bgr))
        lid_2_0 = self.drop(self.conv2_0_lid(down_lid_t_1_0))
        down_bgr_2_0 = self.pool(bgr_2_0) 
        down_lid_2_0 = self.pool(lid_2_0) 

        #BOTTLENECK 1
        cat_in_bgrlid = cat([down_bgr_2_0, down_lid_2_0], dim=1)
        bgrlid_3_0 = self.drop(self.conv3_0_bgrlid(cat_in_bgrlid))
        up_bgrlid_3_0 = self.up(bgrlid_3_0)      

        #UPSAMPLING    
        dep_2_1 = self.drop(self.conv2_1_dep(up_bgrlid_3_0))
        seg_2_1 = self.drop(self.conv2_1_seg(up_bgrlid_3_0))
        lidseg_2_1 = self.drop(self.conv2_1_lidseg(up_bgrlid_3_0))
        up_dep_2_1 = self.up(dep_2_1)
        up_seg_2_1 = self.up(seg_2_1)
        up_lidseg_2_1 = self.up(lidseg_2_1)

        dep_f_1_2 = self.conv1_2_dep_f(cat([up_dep_2_1, bgr_f_1_0], dim=1))
        dep_l_1_2 = self.conv1_2_dep_l(cat([up_dep_2_1, bgr_l_1_0], dim=1))
        dep_ri_1_2 = self.conv1_2_dep_ri(cat([up_dep_2_1, bgr_ri_1_0], dim=1))
        dep_r_1_2 = self.conv1_2_dep_r(cat([up_dep_2_1, bgr_r_1_0], dim=1))
        seg_f_1_2 = self.conv1_2_seg_f(cat([up_seg_2_1, bgr_f_1_0], dim=1))
        seg_l_1_2 = self.conv1_2_seg_l(cat([up_seg_2_1, bgr_l_1_0], dim=1))
        seg_ri_1_2 = self.conv1_2_seg_ri(cat([up_seg_2_1, bgr_ri_1_0], dim=1))
        seg_r_1_2 = self.conv1_2_seg_r(cat([up_seg_2_1, bgr_r_1_0], dim=1))
        lidseg_t_1_2 = self.conv1_2_lidseg_t(cat([up_lidseg_2_1, lid_t_1_0], dim=1))
        up_dep_f_1_2 = self.up(dep_f_1_2)
        up_dep_l_1_2 = self.up(dep_l_1_2)
        up_dep_ri_1_2 = self.up(dep_ri_1_2)
        up_dep_r_1_2 = self.up(dep_r_1_2)
        up_seg_f_1_2 = self.up(seg_f_1_2)
        up_seg_l_1_2 = self.up(seg_l_1_2)
        up_seg_ri_1_2 = self.up(seg_ri_1_2)
        up_seg_r_1_2 = self.up(seg_r_1_2)
        up_lidseg_t_1_2 = self.up(lidseg_t_1_2)

        dep_f_0_3 = self.conv0_3_dep_f(cat([up_dep_f_1_2, bgr_f_0_0], dim=1))
        dep_l_0_3 = self.conv0_3_dep_l(cat([up_dep_l_1_2, bgr_l_0_0], dim=1))
        dep_ri_0_3 = self.conv0_3_dep_ri(cat([up_dep_ri_1_2, bgr_ri_0_0], dim=1))
        dep_r_0_3 = self.conv0_3_dep_r(cat([up_dep_r_1_2, bgr_r_0_0], dim=1))
        seg_f_0_3 = self.conv0_3_seg_f(cat([up_seg_f_1_2, bgr_f_0_0], dim=1))
        seg_l_0_3 = self.conv0_3_seg_l(cat([up_seg_l_1_2, bgr_l_0_0], dim=1))
        seg_ri_0_3 = self.conv0_3_seg_ri(cat([up_seg_ri_1_2, bgr_ri_0_0], dim=1))
        seg_r_0_3 = self.conv0_3_seg_r(cat([up_seg_r_1_2, bgr_r_0_0], dim=1))
        lidseg_t_0_3 = self.conv0_3_lidseg_t(cat([up_lidseg_t_1_2, lid_t_0_0], dim=1))
        out_dep_f = self.act_dep(self.final_dep_f(dep_f_0_3))
        out_dep_l = self.act_dep(self.final_dep_l(dep_l_0_3))
        out_dep_ri = self.act_dep(self.final_dep_ri(dep_ri_0_3))
        out_dep_r = self.act_dep(self.final_dep_r(dep_r_0_3))
        out_seg_f = self.act_seg(self.final_seg_f(seg_f_0_3))
        out_seg_l = self.act_seg(self.final_seg_l(seg_l_0_3))
        out_seg_ri = self.act_seg(self.final_seg_ri(seg_ri_0_3))
        out_seg_r = self.act_seg(self.final_seg_r(seg_r_0_3))
        out_lidseg_t = self.act_lidseg(self.final_lidseg_t(lidseg_t_0_3))

        #DOWNSAMPLING
        dep_f_0_0 = self.conv0_0_dep_f(out_dep_f)
        dep_l_0_0 = self.conv0_0_dep_l(out_dep_l)
        dep_ri_0_0 = self.conv0_0_dep_ri(out_dep_ri)
        dep_r_0_0 = self.conv0_0_dep_r(out_dep_r)
        seg_f_0_0 = self.conv0_0_seg_f(out_seg_f)
        seg_l_0_0 = self.conv0_0_seg_l(out_seg_l)
        seg_ri_0_0 = self.conv0_0_seg_ri(out_seg_ri)
        seg_r_0_0 = self.conv0_0_seg_r(out_seg_r)
        lidseg_t_0_0 = self.conv0_0_lidseg_t(out_lidseg_t) 
        down_dep_f_0_0 = self.pool(dep_f_0_0)
        down_dep_l_0_0 = self.pool(dep_l_0_0)
        down_dep_ri_0_0 = self.pool(dep_ri_0_0)
        down_dep_r_0_0 = self.pool(dep_r_0_0)
        down_seg_f_0_0 = self.pool(seg_f_0_0)
        down_seg_l_0_0 = self.pool(seg_l_0_0)
        down_seg_ri_0_0 = self.pool(seg_ri_0_0)
        down_seg_r_0_0 = self.pool(seg_r_0_0)
        down_lidseg_t_0_0 = self.pool(lidseg_t_0_0)

        dep_f_1_0 = self.conv1_0_dep_f(down_dep_f_0_0)
        dep_l_1_0 = self.conv1_0_dep_l(down_dep_l_0_0)
        dep_ri_1_0 = self.conv1_0_dep_ri(down_dep_ri_0_0)
        dep_r_1_0 = self.conv1_0_dep_r(down_dep_r_0_0)
        seg_f_1_0 = self.conv1_0_seg_f(down_seg_f_0_0)
        seg_l_1_0 = self.conv1_0_seg_l(down_seg_l_0_0)
        seg_ri_1_0 = self.conv1_0_seg_ri(down_seg_ri_0_0)
        seg_r_1_0 = self.conv1_0_seg_r(down_seg_r_0_0)
        lidseg_t_1_0 = self.conv1_0_lidseg_t(down_lidseg_t_0_0)
        down_dep_f_1_0 = self.pool(dep_f_1_0)
        down_dep_l_1_0 = self.pool(dep_l_1_0)
        down_dep_ri_1_0 = self.pool(dep_ri_1_0)
        down_dep_r_1_0 = self.pool(dep_r_1_0)
        down_seg_f_1_0 = self.pool(seg_f_1_0)
        down_seg_l_1_0 = self.pool(seg_l_1_0)
        down_seg_ri_1_0 = self.pool(seg_ri_1_0)
        down_seg_r_1_0 = self.pool(seg_r_1_0)
        down_lidseg_t_1_0 = self.pool(lidseg_t_1_0)

        #FUSE ALL FEATURES
        cat_in_dep = cat([down_dep_f_1_0, down_dep_l_1_0, down_dep_ri_1_0, down_dep_r_1_0], dim=1)
        cat_in_seg = cat([down_seg_f_1_0, down_seg_l_1_0, down_seg_ri_1_0, down_seg_r_1_0], dim=1)
        dep_2_0 = self.drop(self.conv2_0_dep(cat_in_dep))
        seg_2_0 = self.drop(self.conv2_0_seg(cat_in_seg))
        lidseg_2_0 = self.drop(self.conv2_0_lidseg(down_lidseg_t_1_0))
        down_dep_2_0 = self.pool(dep_2_0)
        down_seg_2_0 = self.pool(seg_2_0) 
        down_lidseg_2_0 = self.pool(lidseg_2_0) 

        #BOTTLENECK 2
        cat_in_depseglidseg = cat([down_dep_2_0, down_seg_2_0, down_lidseg_2_0], dim=1)
        depseglidseg_3_0 = self.drop(self.conv3_0_depseglidseg(cat_in_depseglidseg))
        up_depseglidseg_3_0 = self.up(depseglidseg_3_0)
       
        #UPSAMPLING
        bir_2_1 = self.drop(self.conv2_1_bir(up_depseglidseg_3_0))
        up_bir_2_1 = self.up(bir_2_1)

        bir_t_1_2 = self.conv1_2_bir_t(cat([up_bir_2_1, lidseg_t_1_0], dim=1))
        up_bir_t_1_2 = self.up(bir_t_1_2)

        bir_t_0_3 = self.conv0_3_bir_t(cat([up_bir_t_1_2, lidseg_t_0_0], dim=1))
        out_bir_t = self.act_bir(self.final_bir_t(bir_t_0_3))

        return [out_dep_f, out_dep_l, out_dep_ri, out_dep_r, out_seg_f, out_seg_l, out_seg_ri, out_seg_r, out_lidseg_t, out_bir_t]


