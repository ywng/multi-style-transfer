# Modified from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/
# The main changes are 
# 1) conditional instance normalization logic 
# 2) allow user to choose and combine multiple styles to transfer
# 3) video multiple style combine and transfer

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer import TransformerNet
from vgg import Vgg16

from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from PIL import Image


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size), # the shorter side is resize to match image_size
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255)) # convert back to [0, 255]
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # to provide a batch loader

    
    style_image = [f for f in os.listdir(args.style_image)]
    style_num = len(style_image)
    print("Total number of styles: ", style_num)
    for style_id, style_img in enumerate(style_image):
        print("id: %3s image: %30s" % (style_id, style_img))

    transformer = TransformerNet(style_num=style_num)

    if args.resume and os.path.isfile(args.resume):
        state_dict = torch.load(args.resume)
        transformer.load_state_dict(state_dict)

    transformer = transformer.to(device)

    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.Resize(args.style_size), 
        transforms.CenterCrop(args.style_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_batch = []

    for i in range(style_num):
        style = utils.load_image(args.style_image + style_image[i], size=args.style_size)
        style = style_transform(style)
        style_batch.append(style)

    style = torch.stack(style_batch).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            
            if n_batch < args.batch_size:
                break # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            optimizer.zero_grad() # initialize with zero gradients

            batch_style_control = [utils.single_style_control_list_maker(style_num, i % style_num) for i in range(count-n_batch, count)]
            batch_style_id = [i % style_num for i in range(count-n_batch, count)]
            y = transformer(x.to(device), style_control=batch_style_control)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y.to(device))
            features_x = vgg(x.to(device))
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[batch_style_id, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '') + "_" + str(int(
        args.content_weight)) + "_" + str(int(args.style_weight)) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def load_trained_model(args, device):
    style_model = TransformerNet(style_num=args.style_num)
    state_dict = torch.load(args.model)
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    return style_model

def stylize_one_img(args, content_image, style_control):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = load_trained_model(args, device)
        output = style_model(content_image, style_control=[style_control]).cpu()

    return output 

def stylize_img(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    output = stylize_one_img(args, content_image, args.style_control)
    utils.save_image('./images/output_images/'+args.output_image+'_style'+''.join(str(e) for e in args.style_control)+'.jpg', output[0])

def stylize_video(args):
    batch_size = 4
    video_clip = VideoFileClip(args.content_video, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter('./videos/output_videos/'+args.output_video+'_style'+''.join(str(e) for e in args.style_control)+'.mp4', 
                        video_clip.size, video_clip.fps, codec="libx264",
                        preset="medium", bitrate="2000k",
                        audiofile=args.content_video, threads=None,
                        ffmpeg_params=None)

    batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
    X = np.zeros(batch_shape, dtype=np.float32)

    def style_and_write(count, X):
        outputs = list()
        for i in range(count, batch_size):
            X[i] = X[count - 1]  # Use last frame to fill X
            
        for i in range(0, count):
            content_image = X[i]
            output = stylize_one_img(args, content_image, args.style_control)
            output = output[0].clone().clamp(0, 255).numpy()
            output = output.transpose(1, 2, 0).astype("uint8")
            output = Image.fromarray(output)
            video_writer.write_frame(output)

    frame_count = 0  # The frame count that written to X
    for frame in video_clip.iter_frames():
        X[frame_count] = frame
        frame_count += 1
        if frame_count % 10 == 0:
            print("Processed %d frames." % frame_count)
        if frame_count == batch_size:
            style_and_write(frame_count, X)
            frame_count = 0

    if frame_count != 0:
        style_and_write(frame_count, X)

    video_writer.close()

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=512,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=250,
                                  help="number of images after which the training loss is logged, default is 250")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--resume", type=str,
                                 help="saved checkpoint to resume training.")
    

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, 
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-video", type=str, 
                                 help="path to content video you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, 
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--output-video", type=str, 
                                 help="path for saving the output video")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, 
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--style-control", type=int, nargs='+', required = True,
                                 help="style control weights corresponding to the order in training")
    eval_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for testing, default is 4")
    eval_arg_parser.add_argument("--style-num", type=int, default=4,
                                  help="number of styles used in training, default is 4")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1) # 1 for all other types of error besides syntax
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        if args.content_video:
            stylize_video(args)
        else:
            stylize_img(args)


if __name__ == "__main__":
    main()
