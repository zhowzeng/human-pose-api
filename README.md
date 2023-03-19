# MMPose-API

[MMPose Documentation](https://mmpose.readthedocs.io/en/v0.29.0/index.html)

## TODO

- [x] Run 2d demo
- [x] Implement fastapi server
- [ ] Return 2d -> 3d
- [ ] Containerize

## Installation

1. Use virtual environment. My python version is `3.8.10`.
    ```bash
    python3 -m venv venv  # will mkdir venv automatically
    source venv/bin/activate  # activate environment. to leave venv, run `deactivate`
    pip install -U pip
    ```

2. Follow this [link](https://pytorch.org/get-started/locally/) to install PyTorch GPU/CPU version. For me, `torch==2.0.0`.

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

    Edit file `./venv/lib/python3.8/site-packages/mmpose/datasets/pipelines/bottom_up_transform.py` and add one line.
    ```diff
    + 13 np.int = int

      15 def _ceil_to_multiples_of(x, base=64):
      16     """Transform x to the integral multiple of the base."""
      17     return int(np.ceil(x / base)) * base
    ```

## Launch API

```bash
python main.py
```

Visit http://127.0.0.1:8000/docs to use swagger UI

## Official demo

```bash
# download config and model
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest ./demo/
# run script
python official_demo.py
```
