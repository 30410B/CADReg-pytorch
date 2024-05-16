# python imports
from ssim3d import ssim3D
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
# internal imports
from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Models.STN import SpatialTransformer
from natsort import natsorted
from Models.network import CONFIGS as CONFIGS_TM
import Models.network as CADReg
import  Utils
from medpy.metric import binary
import time

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join('./Result', name))


def compute_label_dice(gt, pred,std_idx):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    dice_lst = []
    line = 'p_{}'.format(std_idx)
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
        line = line+','+str(dice)
    return np.mean(dice_lst),line


def train():

    # 创建需要的文件夹并指定gpu
    make_dirs()
    device = torch.device('cuda:{}'.format(1) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    csv_name = 'Ours'
    dict = Utils.process_label_lpba()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(csv_name, 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(54):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)
    # 读入fixed图像 [D, W, H] = 160×192×160
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()


    # 创建配准网络（net）和STN
    config = CONFIGS_TM['TransMorph']
    net = CADReg.CADReg(config)
    net.to(device)
    best_model = torch.load('./experiments/V124_2500_0.0001_1/dsc0.7260epoch243.pth.tar',map_location='cpu')['state_dict']
    net.load_state_dict(best_model)


    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    # UNet.train()


    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

    net.eval()
    STN.eval()
    STN_label.eval()
    DSC = []
    TIME = []
    HD95 = []
    SSIM = []
    eval_det = Utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for file in test_file_lst:
            fig_name = file[58:60]
            name = os.path.split(file)[1]
            # 读入moving图像
            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            # 读入moving图像对应的label
            label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
            #input_label = torch.from_numpy(input_label).to(device).float()

            # 获得配准后的图像和label
            x=torch.cat((input_fixed_eval,input_moving),dim=1)
            start = time.time()
            pred_flow = net(x)
            pred_img = STN(input_fixed_eval, pred_flow)
            TIME.append(time.time() - start)
            pred_flow1=pred_flow.permute(0,1,4,3,2)
            pred_label = STN_label(fixed_label, pred_flow)

            #SSIM.append(ssim3D(input_moving.double(),pred_img.double()).item())
            #hd95=binary.hd95(input_label[0, 0, ...], pred_label[0, 0, ...].cpu().detach().numpy())
            #HD95.append(hd95)
            dice,line= compute_label_dice(input_label, pred_label[0, 0, ...].cpu().detach().numpy(),stdy_idx)
            print("{0}" .format(dice))
            DSC.append(dice)
            tmpName = str(file[53:56])  # please check the tmpName when you run by yourself
            tar = input_moving.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = Utils.jacobian_determinant_vxm(pred_flow1.detach().cpu().numpy()[0, :, :, :, :])
            line = line +','+str(np.sum(jac_det <= 0) / np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), input_fixed_eval.size(0))
            stdy_idx+=1
            #print(pred_flow.shape)

        print(np.mean(DSC), np.std(DSC))    
        print(np.mean(TIME))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
#        print('Mean_hd: {}',np.mean(HD95)) 
#        print('Mean_SSIM: {}',np.mean(SSIM))         
'''         save_image(pred_img, f_img, tmpName + '_warpped.nii.gz')
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, tmpName + "_flow.nii.gz")
            save_image(pred_label, f_img, tmpName + "_label.nii.gz")
'''
def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
