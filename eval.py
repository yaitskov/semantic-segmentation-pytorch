# System libs
from pathlib import Path
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import shutil
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset, massageImage
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def doSegmentation(model, imgPath):
  model.eval()

  img = Image.open(imgPath).convert('RGB')
  # preparedImgs = []
  segSize = img.size
  segSize = segSize[1], segSize[0]
  imgSizes = [300, 375, 450, 525, 600]
  with torch.no_grad():
    scores = torch.zeros(1, 150, segSize[0], segSize[1])
    for targetSize in imgSizes:
      preparedImgs = massageImage(img, imgMaxSize=1000, targetSize=targetSize, padding_constant=8)
      scores_tmp = model({'img_data': preparedImgs.contiguous()}, segSize=segSize) #(300, 300))
      scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
      # print(f"segSize {segSize} for image {img.size}")

    _, pred = torch.max(scores, dim=1)
    pred = pred.squeeze(0).numpy()

    # shutil.rmtree("layers")
    os.makedirs("layers", exist_ok = True)
    print(f"Segmented tensor info: shape = {pred.shape} type = {pred.dtype}")

    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = pred_color.astype(np.uint8) # np.concatenate((img, seg_color, pred_color),
                         #   axis=1).astype(np.uint8)

    # img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(f"layers/{Path(imgPath).stem}.png")

    # print(f"colored {pred_color.dtype} {pred_color.shape} <> {pred.dtype} {pred.shape}")
    #print(f"pred_color histogram: {np.histogram(pred_color, 20)}")
    print(f"pred histogram: {np.histogram(pred, range(150))}")

    #img = Image.frombytes(mode='L', size=segSize, data=pred.reshape(-1).numpy())
    #img.save(f"layers/{os.path.basename(imgPath)}.png")
    #img.close()

    img = Image.fromarray(
      (pred.astype(np.float32) * (255.0 / pred.max())).astype(np.uint8),
      mode='L'
      )
    img.save(f"layers/{os.path.basename(imgPath)}.gray.png")
    img.close()

def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        # torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            # scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                # feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0))

        # torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


def main(cfg, gpu):
    # try:
    #     torch.cuda.set_device(gpu)
    # except AttributeError as e:
    #     print(f"Ignore failed set_device: {e}")

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    if args.img is None:
      # Dataset and Loader
      dataset_val = ValDataset(
          cfg.DATASET.root_dataset,
          cfg.DATASET.list_val,
          cfg.DATASET)
      loader_val = torch.utils.data.DataLoader(
          dataset_val,
          batch_size=cfg.VAL.batch_size,
          shuffle=False,
          collate_fn=user_scattered_collate,
          num_workers=5,
          drop_last=True)

      # segmentation_module.cuda()

      # Main loop

      evaluate(segmentation_module, loader_val, cfg, gpu)

      print('Evaluation Done!')
    else:
      doSegmentation(segmentation_module, args.img)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        # default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        default="config/ade20k-mobilenetv2dilated-c1_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
      "--img",
      default=None,
      help="path for image file for semantic segmenation")
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder), f"checkpoint {cfg.MODEL.weights_encoder} does not exitst!"
    assert os.path.exists(cfg.MODEL.weights_decoder), f"checkpoint {cfg.MODEL.weights_decoder} does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
