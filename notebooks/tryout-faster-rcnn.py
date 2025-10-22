import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import animaloc
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor

    return Image, ToTensor, animaloc, torch


@app.cell
def _(torch):
    model_pth_path = "data/models/tmp/faster-rcnn/best_model-by-epoch-70.pth"
    model_dict = torch.load(model_pth_path, weights_only=True)
    type(model_dict), model_dict.keys()
    return (model_dict,)


@app.cell
def _(model_dict):
    print("classes: ", model_dict["classes"])
    print("loss: ", model_dict["loss"])
    print("best_val: ", model_dict["best_val"])

    return


@app.cell
def _():
    from inspect import getsource

    from api.model_utils import make_faster_rcnn_model

    print(getsource(make_faster_rcnn_model))
    return


@app.cell
def _(animaloc, model_dict):
    # list(model_dict['model_state_dict'].keys())

    model_name = "FasterRCNNResNetFPN"  # from train cfg
    architecture = "resnet50"
    model_cls = animaloc.models.__dict__[model_name]
    model = model_cls(
        architecture=architecture, num_classes=max(model_dict["classes"]) + 1
    )  # +1 for background
    model.eval()
    return (model,)


@app.cell
def _(model_dict):
    state_dict = model_dict["model_state_dict"]
    state_dict = {k.lstrip("model."): v for k, v in state_dict.items()}
    state_dict.keys()
    return (state_dict,)


@app.cell
def _(model, state_dict):
    # model = make_faster_rcnn_model(num_classes=max(model_dict['classes']) + 1)
    model.load_state_dict(state_dict)
    model
    return


@app.cell
def _(Image, ToTensor):
    from albumentations.augmentations import Normalize

    image_path = "data/train_subframes/L_07_05_16_DSC00127_S3.JPG"

    img = Image.open(image_path)
    img1 = img.crop([1488, 300, 2000, 812])
    to_tensor = ToTensor()
    # norm = Normalize()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img1, img1_tensor.shape

    return (img1_tensor,)


@app.cell
def _(img1_tensor, model, torch):
    with torch.no_grad():
        preds = model(img1_tensor)
    preds
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
