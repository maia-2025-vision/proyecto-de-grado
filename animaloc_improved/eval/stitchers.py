"""Our own version of stitches model

Added this file to fix a tiny bug found on animaloc.eval.stitchers.FasterRCNNStitcher
See "BUG FIX" below for details.
Also did a few minor style changes.
"""

from typing import TypeAlias

import torch
import torchvision
from animaloc.eval.stitchers import STITCHERS, Stitcher
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

DictOfTensors: TypeAlias = dict[str, torch.Tensor]


@STITCHERS.register()
class FasterRCNNStitcherV2(Stitcher):  # type: ignore[misc, no-any-unimported]
    """Essentially the same as FasterRCNNStitcher"""

    def __init__(
        self,
        model: torch.nn.Module,
        size: tuple[int, int],
        overlap: int = 100,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.0,
        batch_size: int = 1,
        device_name: str = "cuda",
    ) -> None:
        super().__init__(
            model, size, overlap=overlap, batch_size=batch_size, device_name=device_name
        )
        logger.info(
            f"Building FasterRCNNStitcherV2 with {batch_size=} "
            f"| {nms_threshold=} | {score_threshold=}"
        )

        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.up = False

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> list[DictOfTensors]:
        self.model.eval()
        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset)
        )

        maps = []
        for patch in dataloader:
            # print(f"stitchers: patch={len(patch)}\n", [type(it).__name__ for it in patch])
            assert len(patch) == 1, "Are we skipping data?"
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)
            # BUG FIX HERE: append cannot be called with more than one arg...
            # maps.append(*outputs)
            maps.extend(outputs)

        return maps

    def _patch_maps(self, maps: list[DictOfTensors]) -> DictOfTensors:
        boxes, labels, scores = [], [], []
        for map, limit in zip(maps, self.get_limits().values(), strict=True):
            for box in map["boxes"].tolist():
                x1, y1, x2, y2 = box
                new_box = [x1 + limit.x_min, y1 + limit.y_min, x2 + limit.x_min, y2 + limit.y_min]
                boxes.append(new_box)

            labels.extend(map["labels"].tolist())
            scores.extend(map["scores"].tolist())

        return dict(
            boxes=torch.Tensor(boxes), labels=torch.Tensor(labels), scores=torch.Tensor(scores)
        )

    def _reduce(self, map: DictOfTensors) -> DictOfTensors:
        if map["boxes"].nelement() == 0:
            return map
        else:
            indices = torchvision.ops.nms(map["boxes"], map["scores"], self.nms_threshold)
            reduced = dict(
                boxes=map["boxes"][indices],
                labels=map["labels"][indices],
                scores=map["scores"][indices],
            )
            # score thresholding
            indices = torch.nonzero((reduced["scores"] > self.score_threshold), as_tuple=True)[0]
            reduced = dict(
                boxes=reduced["boxes"][indices],
                labels=reduced["labels"][indices],
                scores=reduced["scores"][indices],
            )

            return reduced
