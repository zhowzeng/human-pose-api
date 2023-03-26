import argparse
import logging
import logging.config
from typing import List

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model

from pydantic import BaseModel
from utils import b64_to_cv, read_by_b64
from version import __version__

with open('./logging.yaml', 'r') as f:
    LOGGING_CFG = yaml.safe_load(f)


app = FastAPI(
    description='Post Image and return human pose.',
    title='MMPose API',
    version=__version__
)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_headers=['*'],
    allow_methods=['*'],
    allow_origins=['*']
)


class ImageString(BaseModel):
    string: str

    class Config:
        schema_extra = {
            'example': {
                'string': read_by_b64("./demo/blue_in.jpg")
            }
        }


class ImagePath(BaseModel):
    path: str

    class Config:
        schema_extra = {
            'example': {
                'path': './demo/blue_in.jpg'
            }
        }


class Keypoints(BaseModel):
    keypoints: List[List[float]]
    score: float
    area: float


@app.on_event('startup')
async def startup_event():
    # # initialize mmpose model
    global POSE_MODEL
    # XXX: for demo
    config_file = './demo/associative_embedding_hrnet_w32_coco_512x512.py'
    checkpoint_file = './demo/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    POSE_MODEL = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'


@app.post('/v1/keypoints/2d/string', response_model=List[Keypoints])
async def post_string_and_get_2d_keypoints(req: ImageString) -> None:
    im_arr = b64_to_cv(req.string)
    pose_results, _ = inference_bottom_up_pose_model(POSE_MODEL, im_arr)
    return [Keypoints(**process_result(res)) for res in pose_results]


@app.post('/v1/keypoints/2d/path', response_model=List[Keypoints])
async def post_path_and_get_2d_keypoints(req: ImagePath) -> None:
    pose_results, _ = inference_bottom_up_pose_model(POSE_MODEL, req.path)
    return [Keypoints(**process_result(res)) for res in pose_results]


def process_result(result: dict) -> dict:
    result["keypoints"] = result["keypoints"].tolist()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action="store_true")
    args = parser.parse_args()
    logging.config.dictConfig(LOGGING_CFG)
    uvicorn.run("__main__:app", log_config=LOGGING_CFG, host=args.host, port=args.port, workers=1, reload=args.reload)
