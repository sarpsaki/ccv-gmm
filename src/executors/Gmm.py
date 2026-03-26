import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor

from capsules.CcvGmm.src.utils.response import build_response_gmm
from capsules.CcvGmm.src.models.PackageModel import PackageModel
from capsules.CcvGmm.src.utils.gmm_core import run_gmm, encode_rle, SEGMENT_COLORS


class Gmm(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**self.request.data)

        self.images = self.request.get_param("inputImage")
        self.mode = self.request.get_param("configGmmMode")
        self.num_components = self.request.get_param("numComponents")
        self.point_x = self.request.get_param("pointX")
        self.point_y = self.request.get_param("pointY")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def process_image(self, image_array: np.ndarray, img_uid: str):
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        n_components = int(self.num_components) if self.num_components else 2
        processed_masks = []

        labels = run_gmm(image_array, n_components=n_components)

        if self.mode == "POINT":
            cx, cy = int(self.point_x), int(self.point_y)
            target_label = labels[cy, cx]
            binary_mask = (labels == target_label).astype(bool)

            processed_masks.append({
                "cluster_id": int(target_label),
                "mask": binary_mask,
            })
        else:
            for comp_id in np.unique(labels):
                comp_id = int(comp_id)
                binary_mask = (labels == comp_id).astype(bool)
                if binary_mask.sum() == 0:
                    continue
                processed_masks.append({
                    "cluster_id": comp_id,
                    "mask": binary_mask,
                })

        results = []
        for m in processed_masks:
            binary_mask = m["mask"]
            ys, xs = np.where(binary_mask)
            if len(ys) == 0:
                continue

            left, top = int(xs.min()), int(ys.min())
            right, bottom = int(xs.max()), int(ys.max())
            cid = m["cluster_id"]
            r, g, b = SEGMENT_COLORS[cid % len(SEGMENT_COLORS)]
            color_hex = f"#{r:02x}{g:02x}{b:02x}"

            results.append({
                "cluster_id": cid,
                "classLabel": f"Component_{cid}",
                "classId": cid,
                "mask_rle": encode_rle(binary_mask),
                "box_2d": [left, top, right, bottom],
                "imgUID": img_uid,
                "color": color_hex,
                "confidence": 1.0,
                "boundingBox": {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                },
            })

        return results

    def run(self):
        image_obj = Image.get_frame(img=self.images, redis_db=self.redis_db)

        self.outputData = self.process_image(np.array(image_obj.value), image_obj.uID)

        return build_response_gmm(context=self)


if __name__ == "__main__":
    Executor(sys.argv[1]).run()
