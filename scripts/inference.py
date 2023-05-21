from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
from loguru import logger
sys.path.append(".")
sys.path.append("..")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
import dlib
from scripts.align_all_parallel import align_face


EXPERIMENT_TYPE = 'ffhq_aging'
MODEL_PATHS = {
    "ffhq_aging": {"id": "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC", "name": "sam_ffhq_aging.pt"}
}

# Define Inference Parameters
EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": "./pretrained_models/sam_ffhq_aging.pt",
        "image_path": "./test_images/ngoc_trinh.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}


def run_alignment(image_path, detect_face_config):
    predictor = dlib.shape_predictor(detect_face_config)
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    logger.info("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url  

def main():
    parser = argparse.ArgumentParser(description="Example config",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--age", type = int, help="Age you want to output")
    args = parser.parse_args()
    config = vars(args)

    logger.info("The config is {}".format(config))
    
    # EXPERIMENT_TYPE = 'ffhq_aging'
    # path = MODEL_PATHS[EXPERIMENT_TYPE]
    # download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
    
    # Load Pretrained Model
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
    model_path = EXPERIMENT_ARGS['model_path']
    ckpt = torch.load(model_path, map_location='cpu')

    opts = ckpt['opts']
    pprint.pprint(opts)

    # Update the training options
    opts['checkpoint_path'] = model_path

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    logger.success('Model successfully loaded!')


    # Set up input
    image_path = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]["image_path"]


    image_name = os.path.splitext(os.path.basename(image_path))[0]
    original_image = Image.open(image_path).convert("RGB")


    # Align image
    detect_face_config = "shape_predictor_68_face_landmarks.dat"
    aligned_image = run_alignment(image_path, detect_face_config)

    # Run inference 
    img_transforms = EXPERIMENT_ARGS['transform']
    input_image = img_transforms(aligned_image)

    age_transformers = AgeTransformer(target_age=int(args.age))
    
    
    # for each age transformed age, we'll concatenate the results to display them side-by-side
    results = np.array(aligned_image.resize((1024, 1024)))

    logger.info(f"Running on target age: {age_transformers.target_age}")
    with torch.no_grad():
        input_image_age = [age_transformers(input_image.cpu()).to('cuda')]
        input_image_age = torch.stack(input_image_age)
        result_tensor = run_on_batch(input_image_age, net)[0]
        result_image = tensor2im(result_tensor)
        # results = np.concatenate([results, result_image], axis=1)
        # results = Image.fromarray(result_image)
        result_image.save("./output/age_{}_{}.jpg".format(str(args.age), image_name))


if __name__ == "__main__":
    main()