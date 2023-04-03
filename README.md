# MMPose-API

[MMPose Documentation](https://mmpose.readthedocs.io/en/v0.29.0/index.html)

## TODO

- [x] Implement fastapi server
- [ ] Containerize
- [ ] Documentation for API

## Installation

1. Use virtual environment. My python version is `3.8.10`.
    ```bash
    python3 -m venv venv  # will mkdir venv automatically
    source venv/bin/activate  # activate environment. to leave venv, run `deactivate`
    pip install -U pip setuptools
    ```

1. Follow this [link](https://pytorch.org/get-started/locally/) to install PyTorch & Torchvision GPU/CPU version.

1. Install other requirements.
    ```bash
    pip install -r requirements.txt
    ```

1. Follow this [link](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md). For `PyTypeTest on non extension type` error, visit [this issue](https://github.com/MVIG-SJTU/AlphaPose/issues/1002).


## Launch API

```bash
python main.py
```

Visit http://127.0.0.1:8000/docs for swagger UI.

You can also send requests by `curl` tool or python `requests` package.

## Official demo

```bash
# download config and model
mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest ./demo/
# run script
python official_demo.py
```
