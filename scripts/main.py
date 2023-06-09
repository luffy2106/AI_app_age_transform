from fastapi import FastAPI, status, UploadFile, File, Request, Form
import torch
from loguru import logger
import timeit
import pprint
from argparse import Namespace
from models.psp import pSp
import torchvision.transforms as transforms
from PIL import Image
import shutil
import dlib
from scripts.align_all_parallel import align_face
from datasets.augmentations import AgeTransformer
from fastapi.responses import HTMLResponse
from utils.common import tensor2im
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import os
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks
# Define Inference Parameters
EXPERIMENT_DATA_ARGS = {
    "model_path": "./pretrained_models/sam_ffhq_aging.pt",
    "transform": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
}

def gpu_memory_cleanup():
    torch.cuda.empty_cache()

def run_alignment(image_path, detect_face_config):
    """
    Detect face in the image
    """
    predictor = dlib.shape_predictor(detect_face_config)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    logger.info("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch


# Load model
def load_model():
    """
    Load the model and move to GPU
    """
    current_dir = os.getcwd()
    logger.info("Current Working Directory: ", current_dir)
    model_path = EXPERIMENT_DATA_ARGS["model_path"]
    ckpt = torch.load(model_path, map_location="cpu")
    opts = ckpt["opts"]
    # print object in a clear format
    pprint.pprint(opts)
    # Update the training options
    opts["checkpoint_path"] = model_path
    # Conver to name space object
    opts = Namespace(**opts)
    # Load model based on the architecture
    net = pSp(opts)
    # Set model to evaluation mode, disables these training-specific layers/operations. This ensures that the model produces consistent and deterministic outputs given the same input, which is desirable for making predictions.
    net.eval()
    # Move model from CPU to GPU
    net.cuda()
    logger.success("Model successfully loaded!")
    return net


# Set up format for log
logger.add(
    sink="log_file.log",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./static/templates")


@app.get("/", response_class=HTMLResponse)
async def upload(request: Request):
    """
    Create template to upload image
    """
    logger.info("start uploading")
    current_dir = os.getcwd()
    logger.info("Current Working Directory: ", current_dir)
    return templates.TemplateResponse("uploadfile.html", {"request": request})


# Define the API route
@app.post(
    "/submit",
    summary="Process an image",
    status_code=status.HTTP_200_OK,
    response_class=HTMLResponse,
)
async def predict(background_tasks: BackgroundTasks,request: Request, age: int = Form(...), file: UploadFile = File(...)):
    """
    Predicts a class of an image
    """
    try:
        starttime = timeit.default_timer()
        logger.info(f"Image inference start")
        starttime = timeit.default_timer()

        model = load_model()
        # Set up input
        image_path = "static/tmp.jpg"
        # Upload image and save it
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Align image
        detect_face_config = "shape_predictor_68_face_landmarks.dat"
        aligned_image = run_alignment(image_path, detect_face_config)

        # Run inference
        img_transforms = EXPERIMENT_DATA_ARGS["transform"]
        input_image = img_transforms(aligned_image)
        age_transformers = AgeTransformer(target_age=int(age))

        logger.info(f"Running on target age: {age_transformers.target_age}")
        file_path_output = "static/age_transform.jpg"
        with torch.no_grad():
            input_image_age = [age_transformers(input_image.cpu()).to("cuda")]
            input_image_age = torch.stack(input_image_age)
            result_tensor = run_on_batch(input_image_age, model)[0]
            result_image = tensor2im(result_tensor)
            # results = np.concatenate([results, result_image], axis=1)
            # results = Image.fromarray(result_image)
            result_image.save(file_path_output)

        logger.info(f"Time of inference {timeit.default_timer() - starttime}")
        logger.info("clear GPU memory")
        background_tasks.add_task(gpu_memory_cleanup)
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "file_path": file_path_output, "number": age},
        )
    except Exception as e:
        logger.exception("Error occur while run the prediction", e)
