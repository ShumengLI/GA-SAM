import os
import gc
import pdb
import sys
import shutil
import argparse
import logging
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage
from importlib import import_module
from tensorboardX import SummaryWriter
from fvcore.nn import FlopCountAnalysis

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from dataloaders.dataset_seg import *
from dataloaders.dataset_aug import get_StrongAug_pixel
from segment_anything_lora import sam_model_registry as sam_model_registry_lora
from segment_anything.utils.transforms import ResizeLongestSide
from sam_lora_image_encoder import LoRA_Sam
from utils import ramps, utils, losses

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Pancreas-CT/processed_h5/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='3d_sam_lora_t', help='model_name')
parser.add_argument('--num_classes', type=int, default=1, help='number of class')

parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--patch_size', type=int, default=128, help='shape of data')
parser.add_argument('--input_size', type=int, default=1024, help='shape of data')
parser.add_argument('--pretrain_model', type=str, default='vit_b', help='vit to select')
parser.add_argument('--strong_aug', action="store_true", help='load psdudo label')
parser.add_argument('--p_per_sample', type=float, default=0.7, help='p_per_sample')

parser.add_argument('--plabel_dir', type=str, default='../model_pancreas/pseudolabel/fsc3_post/', help='Name of Experiment')
parser.add_argument('--save_iter', type=int, default=500, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--save_img', type=int, default=100, help='img saving iterations')

parser.add_argument("--lr_sam", type=float, default=0.001, help="sam learning rate")
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--w_topo', type=float, default=0.1, help='lambda of topo')
parser.add_argument('--post', action="store_true", help='load psdudo label')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=20.0, help='consistency_rampup')
args = parser.parse_args()

root = "../"
train_data_path = args.root_path
snapshot_path = root + "model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
n_gpu = len(args.gpu.split(','))
print(batch_size)
max_iterations, input_size = args.max_iterations, args.input_size
patch_size = (args.patch_size, args.patch_size)
num_classes = args.num_classes
lr_sam = args.lr_sam

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    aug = transforms.Compose([RandomRotFlip(), Resample(patch_size), get_StrongAug_pixel(3, args.p_per_sample)])

    db_train = Pancreas_pseudo(base_dir=train_data_path, plabel_dir=args.plabel_dir, transform=aug)

    multimask_output = True if num_classes > 1 else False
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    print("total {} samples".format(len(db_train)))

    def create_model():
        if args.pretrain_model == "vit_b":
            model_sam, img_embedding_size = sam_model_registry_lora["vit_b"](image_size=args.input_size, num_classes=num_classes, checkpoint='pre_weight/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
        pkg = import_module(args.module)
        model_sam = pkg.LoRA_Sam(model_sam, args.rank).cuda()
        return model_sam

    model1 = create_model()
    model2 = create_model()
    model3 = create_model()
    models = [model1, model2, model3]

    base_lr_sam = lr_sam
    optimizers = [
        torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr_sam, betas=(0.9, 0.999), weight_decay=0.1)
        for model in models
    ]
    for model in models:
        model.train()

    # Set losses
    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = losses.DiceLoss(num_classes+1)
    mse_loss = torch.nn.MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    lr_ = base_lr_sam

    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            weight_batch = sampled_batch['weight'].cuda()
            volume_batch = torch.cat((volume_batch, volume_batch, volume_batch), 1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.long().cuda()

            input_batch1 = volume_batch[0].transpose(0, 1)
            input_batch2 = volume_batch[0].transpose(0, 1).transpose(0, 2)
            input_batch3 = volume_batch[0].transpose(0, 1).transpose(0, 3)

            transform = ResizeLongestSide(input_size)
            input_batches = [
                transform.apply_image_torch(input_batch1),
                transform.apply_image_torch(input_batch2),
                transform.apply_image_torch(input_batch3)
            ]
            output_transforms = [
                lambda x: x.transpose(0, 1).unsqueeze(0),                   # 对应 model1
                lambda x: x.transpose(0, 2).transpose(0, 1).unsqueeze(0),   # 对应 model2
                lambda x: x.transpose(0, 3).transpose(0, 1).unsqueeze(0)    # 对应 model3
            ]

            # 随机选择1个模型
            selected_indices = random.sample([0, 1, 2], 3)
            model_t = models[selected_indices[0]]
            model_a, model_b = models[selected_indices[1]], models[selected_indices[2]]
            input_t = input_batches[selected_indices[0]]
            input_a, input_b = input_batches[selected_indices[1]], input_batches[selected_indices[2]]
            output_transform_t = output_transforms[selected_indices[0]]
            output_transform_a, output_transform_b = output_transforms[selected_indices[1]], output_transforms[selected_indices[2]]
            optimizer_t = optimizers[selected_indices[0]]

            with torch.no_grad():
                outputs_a = model_a(input_a, multimask_output, input_size)
                output_masks_a, low_res_masks_a, _ = outputs_a['masks'], outputs_a['low_res_logits'], outputs_a['iou_predictions']
                output_masks_a = F.interpolate(low_res_masks_a, (args.patch_size, args.patch_size), mode="bilinear", align_corners=False)
                output_masks_a = output_transform_a(output_masks_a)
                output_soft_a = F.softmax(output_masks_a, dim=1)
                prediction_a = torch.argmax(output_soft_a, dim=1)

                outputs_b = model_b(input_b, multimask_output, input_size)
                output_masks_b, low_res_masks_b, _ = outputs_b['masks'], outputs_b['low_res_logits'], outputs_b['iou_predictions']
                output_masks_b = F.interpolate(low_res_masks_b, (args.patch_size, args.patch_size), mode="bilinear", align_corners=False)
                output_masks_b = output_transform_b(output_masks_b)
                output_soft_b = F.softmax(output_masks_b, dim=1)
                prediction_b = torch.argmax(output_soft_b, dim=1)

            # 前向传播
            outputs = model_t(input_t, multimask_output, input_size)
            output_masks, low_res_masks, _ = outputs['masks'], outputs['low_res_logits'], outputs['iou_predictions']
            output_masks = F.interpolate(low_res_masks, (args.patch_size, args.patch_size), mode="bilinear", align_corners=False)
            output_masks = output_transform_t(output_masks)
            output_soft = F.softmax(output_masks, dim=1)
            prediction = torch.argmax(output_soft, dim=1)

            loss_ce = losses.wce(output_masks, label_batch, weight_batch, batch_size, patch_size[0], patch_size[1], patch_size[2])
            loss_dice = losses.multi_dice_loss_weight(output_soft, label_batch, weight_batch, classnum=1)
            loss_sup = loss_ce + loss_dice

            cons_weight = get_current_consistency_weight(iter_num // 150)
            foreground_mask = (prediction_a == 1) & (prediction_b == 1)
            pseudo_label = foreground_mask.float()
            if args.post:
                pseudo_label = pseudo_label[0].cpu().numpy().astype(np.int64)
                pseudo_label = utils.get_largest_component_sitk(pseudo_label)
                pseudo_label = torch.from_numpy(pseudo_label).unsqueeze(0).float().cuda()
            loss_unsup_ce = F.cross_entropy(output_masks, pseudo_label.long(), reduction="mean")
            loss_unsup_mse = F.mse_loss(output_soft[:, 1], pseudo_label, reduction="mean")
            loss_cons = cons_weight * loss_unsup_mse

            gt_dis = utils.compute_sdf(label_batch[:].cpu().numpy(), output_masks[:, 0, ...].shape)
            gt_dis = torch.from_numpy(gt_dis).float().cuda()
            pred_dis = utils.compute_sdf(prediction[:].cpu().numpy(), output_masks[:, 0, ...].shape)
            pred_dis = torch.from_numpy(pred_dis).float().cuda()

            loss_topo = utils.topology_consistency_loss(pred_dis, gt_dis)
            loss_sdf = mse_loss(pred_dis, gt_dis) + args.w_topo * loss_topo

            loss = loss_sup + loss_cons + loss_sdf

            optimizer_t.zero_grad()
            loss.backward()
            optimizer_t.step()

            if iter_num % args.save_img == 0:
                prediction = torch.argmax(output_soft_a, dim=1)
                nib.save(nib.Nifti1Image(volume_batch[0, 0].cpu().data.numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label_batch[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(pseudo_label[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pseudo_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(prediction[0].cpu().detach().numpy().astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_' + str(iter_num) + '.nii.gz')

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_sup1', loss_sup, iter_num)
            writer.add_scalar('loss/loss_cons', loss_cons, iter_num)
            writer.add_scalar('loss/loss_sdf', loss_sdf, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iter %d : sup1 loss : %f, cons loss : %f, sdf loss : %f' % (iter_num, loss_sup, loss_cons, loss_sdf.item()))

            ## change lr
            lr_ = lr_sam * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_t.param_groups:
                param_group['lr'] = lr_

            if iter_num % args.save_iter == 0:
                save_mode_path_sam1 = os.path.join(snapshot_path, 'sam1_iter_' + str(iter_num) + '.pth')
                model1.save_lora_parameters(save_mode_path_sam1)
                logging.info("save model to {}".format(save_mode_path_sam1))
                save_mode_path_sam2 = os.path.join(snapshot_path, 'sam2_iter_' + str(iter_num) + '.pth')
                model2.save_lora_parameters(save_mode_path_sam2)
                logging.info("save model to {}".format(save_mode_path_sam2))
                save_mode_path_sam3 = os.path.join(snapshot_path, 'sam3_iter_' + str(iter_num) + '.pth')
                model3.save_lora_parameters(save_mode_path_sam3)
                logging.info("save model to {}".format(save_mode_path_sam3))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path_sam1 = os.path.join(snapshot_path, 'sam1_iter_' + str(iter_num) + '.pth')
    model1.save_lora_parameters(save_mode_path_sam1)
    logging.info("save model to {}".format(save_mode_path_sam1))
    save_mode_path_sam2 = os.path.join(snapshot_path, 'sam2_iter_' + str(iter_num) + '.pth')
    model2.save_lora_parameters(save_mode_path_sam2)
    logging.info("save model to {}".format(save_mode_path_sam2))
    save_mode_path_sam3 = os.path.join(snapshot_path, 'sam3_iter_' + str(iter_num) + '.pth')
    model3.save_lora_parameters(save_mode_path_sam3)
    logging.info("save model to {}".format(save_mode_path_sam3))
    writer.close()

