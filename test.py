import os
import math
import torch
import cv2 as cv
import time
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR 
from models.common import config
import models


def save_log(recon_root, name_dataset, name_image, psnr, ssim, rate, consecutive=True):
    if not os.path.isfile(f"{recon_root}/Res_{name_dataset}_{rate}.txt"):
        log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'w')
        log.write("=" * 120 + "\n")
        log.close()
    log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'r+')
    if consecutive:
        old = log.read()
        log.seek(0)
        log.write(old)
    log.write(
        f"Res {name_image}: PSNR, {round(psnr, 2)}, SSIM, {round(ssim, 4)}\n")
    log.close()


def save_image(path, image_name, x_hat):
    recon_dataset_path = path
    recon_dataset_path_rate = f"{path}/{config.para.rate}"
    if not os.path.isdir(recon_dataset_path):
        os.mkdir(recon_dataset_path)
    if not os.path.isdir(recon_dataset_path_rate):
        os.mkdir(recon_dataset_path_rate)
    cv.imwrite(f"{recon_dataset_path_rate}/{image_name.split('.')[0]}.png", x_hat)

def imread_CS_py(Iorg):
    block_size = config.para.patch_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

def testing(network, val, save_img, manner='grey'):
    """
    The pre-processing before TCS-Net's forward propagation and the testing platform.
    """
    recon_root = "./reconstructed_images"
    if not os.path.isdir(recon_root):
        os.mkdir(recon_root)
    datasets = ["Set11"] if val else ["OST300"]# 'CIFAR10', "McM18", "General100", "Urban100", "LIVE29", "OST300","original", "BSD68"
    with torch.no_grad():
        for one_dataset in datasets:
            print(one_dataset + "reconstruction start")
            test_dataset_path = f"../dataset/{one_dataset}"
            # remove the previous log.
            if os.path.isfile(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt"):
                os.remove(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt")
            if os.path.isfile(f"{recon_root}/Res_{one_dataset}_rgb_{config.para.rate}.txt"):
                os.remove(f"{recon_root}/Res_{one_dataset}_rgb_{config.para.rate}.txt")

            sum_time, sum_psnr, sum_ssim = 0., 0., 0.
            for _, _, images in os.walk(f"{test_dataset_path}/"):
                for one_image in images:
                    
                    name_image = one_image.split('.')[0]
                    Img = cv.imread(f"{test_dataset_path}/{one_image}", flags=1)
                    Img_yuv = cv.cvtColor(Img, cv.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()
                    Iorg_y = Img_yuv[:, :, 0]
                    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                    Img_output = Ipad / 255.
                    
                    # 标准化
                    Img_output = (Img_output - 0.45) / 0.22

                    batch_x = torch.from_numpy(Img_output)
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_x = batch_x.to(config.para.device)
                    batch_x = batch_x.unsqueeze(0).unsqueeze(0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=3), dim=0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=2), dim=0)
                    start_time = time.time()
                    x_output= network(batch_x)
                    end_time = time.time()
                    x_output = torch.cat(torch.split(x_output, split_size_or_sections=1 * col_new // config.para.patch_size, dim=0), dim=2)
                    x_output = torch.cat(torch.split(x_output, split_size_or_sections=1, dim=0), dim=3)
                    x_output = x_output.squeeze(0).squeeze(0)
                    Prediction_value = x_output.cpu().data.numpy()

                    # # 标准化逆过程
                    X_rec = Prediction_value[:row, :col] * 0.22 + 0.45

                    X_rec = np.clip(X_rec, 0, 1) * 255.
                    rec_PSNR = PSNR(X_rec, Iorg.astype(np.float64), data_range=255)
                    rec_SSIM = SSIM(X_rec, Iorg.astype(np.float64), data_range=255)
                    Img_rec_yuv[:, :, 0] = X_rec
                    im_rec_rgb = cv.cvtColor(Img_rec_yuv, cv.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                    del x_output
                    execution_time = end_time - start_time
                    sum_time += execution_time
                    sum_psnr += rec_PSNR
                    sum_ssim += rec_SSIM
                    if save_img:
                        save_image(f"{recon_root}/{one_dataset}/", one_image, im_rec_rgb)
                    save_log(recon_root, one_dataset, name_image, rec_PSNR, rec_SSIM, f"gray_{config.para.rate}")
                save_log(recon_root, one_dataset, None, sum_psnr / len(images), sum_ssim / len(images),
                            f"gray_{config.para.rate}_AVG", False)
                print(
                    f"AVG RES of GRAY {one_dataset}: PSNR, {round(sum_psnr / len(images), 2)}, SSIM, {round(sum_ssim / len(images), 4)}, Time, {round(sum_time/len(images), 5)}")
                if val:
                    return round(sum_psnr / len(images), 2), round(sum_ssim / len(images), 4)




def mypsnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    my_state_dict = config.para.my_state_dict
    device = config.para.device

    net = models.SWin_CNet().eval().to(device)
    if os.path.exists(my_state_dict):
        if torch.cuda.is_available():
            trained_model = torch.load(my_state_dict, map_location=device)
        else:
            raise Exception(f"No GPU.")
        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Missing trained model of rate {config.para.rate}.")
    testing(net, val=False, save_img=False)