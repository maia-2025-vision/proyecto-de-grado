import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import torch
    from dotenv import load_dotenv
    from huggingface_hub import HfApi
    import torch.nn as nn
    from transformers import ViTModel, ViTConfig
    import math

    # Replace "hf_YOUR_TOKEN_STRING" with your actual token
    load_dotenv()
    assert "HF_TOKEN" in os.environ

    api = HfApi(token=os.environ["HF_TOKEN"])
    return ViTConfig, ViTModel, nn, torch


@app.cell
def _():
    from transformers import AutoImageProcessor, AutoModel
    from transformers.image_utils import load_image

    pretrained_model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name,
        device_map="auto",
    )
    return AutoModel, load_image, model, pretrained_model_name, processor


@app.cell
def _(load_image):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(url)

    image1 = image.crop((0, 0, 64, 64))
    type(image).__module__
    return (image1,)


@app.cell
def _(processor):
    processor.size
    return


@app.cell
def _(image1, model, processor, torch):
    inputs = processor(images=[image1], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)
    print("Pooled last_hidden_state:", outputs.last_hidden_state.shape)
    return inputs, outputs


@app.cell
def _(image1):
    image1.size, image1.size[0] / 16, image1.size[1] / 16
    return


@app.cell
def _(outputs):
    outputs
    return


@app.cell(hide_code=True)
def _(nn, torch):
    class DETRHead(nn.Module):
        """DETR-style detection head with transformer decoder."""

        def __init__(
            self,
            hidden_dim=768,
            num_classes=80,  # COCO classes, adjust as needed
            num_queries=100,
            num_decoder_layers=6,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
        ):
            super().__init__()
            self.num_queries = num_queries
            self.hidden_dim = hidden_dim

            # Learnable object queries
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # Transformer decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

            # Prediction heads
            self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background/no-object
            self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)  # 3-layer MLP for bbox

        def forward(self, vit_features):
            """
            Args:
                vit_features: [batch_size, num_patches, hidden_dim]
            Returns:
                class_logits: [batch_size, num_queries, num_classes+1]
                bbox_pred: [batch_size, num_queries, 4] (normalized cx, cy, w, h)
            """
            B = vit_features.shape[0]

            # Prepare object queries
            queries = self.query_embed.weight.unsqueeze(0).repeat(
                B, 1, 1
            )  # [B, num_queries, hidden_dim]

            # Decoder: queries attend to ViT features
            decoder_out = self.decoder(queries, vit_features)  # [B, num_queries, hidden_dim]

            # Predictions
            class_logits = self.class_head(decoder_out)  # [B, num_queries, num_classes+1]
            bbox_pred = self.bbox_head(
                decoder_out
            ).sigmoid()  # [B, num_queries, 4], normalized to [0,1]

            return class_logits, bbox_pred

    class MLP(nn.Module):
        """Simple multi-layer perceptron (MLP)."""

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(
                nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
            )

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x

    return (DETRHead,)


@app.cell(hide_code=True)
def _(
    AutoModel,
    DETRHead,
    ViTConfig,
    ViTModel,
    nn,
    pretrained_model_name,
    torch,
):
    class ViTDETRDetector(nn.Module):
        """Complete object detection model: ViT backbone + DETR head."""

        def __init__(
            self,
            # vit_model_name='google/vit-base-patch16-224',
            vit_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
            # number of tokens before actual patch tokens
            n_pre_tokens=5,
            num_classes=7,
            num_queries=100,
            freeze_backbone: bool = True,
            pretrained: bool = True,
        ):
            super().__init__()

            # Load pretrained ViT from Hugging Face
            if vit_model_name.startswith("google"):
                if pretrained:
                    self.vit = ViTModel.from_pretrained(vit_model_name)
                else:
                    config = ViTConfig.from_pretrained(vit_model_name)
                    self.vit = ViTModel(config)
            else:
                self.vit = AutoModel.from_pretrained(
                    pretrained_model_name,
                    device_map="auto",
                )

            # Optionally freeze backbone
            if freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False

            # Get ViT hidden dimension
            hidden_dim = self.vit.config.hidden_size
            self.n_pre_tokens = n_pre_tokens

            # Detection head
            self.detection_head = DETRHead(
                hidden_dim=hidden_dim, num_classes=num_classes, num_queries=num_queries
            )

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize detection head weights."""
            # Initialize bbox head to predict boxes near center
            nn.init.constant_(self.detection_head.bbox_head.layers[-1].weight, 0)
            nn.init.constant_(self.detection_head.bbox_head.layers[-1].bias, 0)

        def forward(self, pixel_values):
            """
            Args:
                pixel_values: [batch_size, 3, H, W] - preprocessed images
            Returns:
                class_logits: [batch_size, num_queries, num_classes+1]
                bbox_pred: [batch_size, num_queries, 4]
            """
            # Extract features from ViT
            vit_outputs = self.vit(pixel_values=pixel_values)

            # Get patch embeddings (exclude CLS token)
            # last_hidden_state: [B, num_patches+1, hidden_dim]
            vit_features = vit_outputs.last_hidden_state[:, self.n_pre_tokens :, :]
            # Remove pre_tokens

            # Detection predictions
            class_logits, bbox_pred = self.detection_head(vit_features)

            return class_logits, bbox_pred

        def predict(self, pixel_values, confidence_threshold=0.0):
            """
            Inference with post-processing.

            Args:
                pixel_values: [batch_size, 3, H, W]
                confidence_threshold: minimum confidence for detections

            Returns:
                List of dicts with 'boxes', 'labels', 'scores' for each image
            """
            self.eval()
            with torch.no_grad():
                class_logits, bbox_pred = self.forward(pixel_values)

                # Convert logits to probabilities
                probs = torch.softmax(class_logits, dim=-1)

                # Get scores and labels (excluding background class)
                scores, labels = probs[:, :, :-1].max(dim=-1)

                # Filter by confidence
                results = []
                for i in range(len(pixel_values)):
                    mask = scores[i] > confidence_threshold
                    results.append(
                        {
                            "boxes": bbox_pred[i][mask],  # [N, 4] in format (cx, cy, w, h)
                            "labels": labels[i][mask],  # [N]
                            "scores": scores[i][mask],  # [N]
                        }
                    )

                return results

    return (ViTDETRDetector,)


@app.cell
def _(ViTDETRDetector):
    det = ViTDETRDetector().to("cuda")
    return (det,)


@app.cell
def _(det, inputs, torch):
    with torch.inference_mode():
        pred = det.predict(**inputs.to("cuda"))
    return (pred,)


@app.cell
def _(pred):
    pred[0]
    return


@app.cell
def _(pred):
    pred[0]["boxes"].shape

    return


@app.cell
def _(pred):
    pred[1].shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
