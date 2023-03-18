# mmpose-api

[MMPose Documentation](https://mmpose.readthedocs.io/en/v0.29.0/index.html)

## Installation

1. Use virtual environment. My python version is 3.10.
    ```bash
    python3 -m venv venv  # will mkdir venv automatically
    source venv/bin/activate  # `deactivate` to leave venv
    pip install -U pip
    ```

2. Follow this [link](https://pytorch.org/get-started/locally/) to install PyTorch GPU/CPU version.

3. Install dependencies for MMPose.

    ```bash
    pip install -U openmim==0.3.6
    mim install mmcv-full==1.7.0
    mim install mmengine==0.7.0
    pip install mmpose==0.29.0
    ```

4. Install other requirements.
    ```bash
    pip install -r requirements.txt
    ```

5. Fix deprecated error of mmpose.

    Edit file `./venv/lib/python3.10/site-packages/mmpose/datasets/pipelines/bottom_up_transform.py` and replace all of `np.int` by `int`.


## Official demo

```bash
# download config and model
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
# run script
python official_demo.py
```