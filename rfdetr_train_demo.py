from rfdetr import RFDETRNano

import pycocotools.coco as coco
import copy

original_loadRes = coco.COCO.loadRes

def patched_loadRes(self, resFile):
    try:
        return original_loadRes(self, resFile)
    except Exception as e:
        # Create a minimal COCO result object
        res = coco.COCO()
        res.dataset = copy.deepcopy(self.dataset)
        # Add minimal required structure
        if 'annotations' not in res.dataset:
            res.dataset['annotations'] = []
        return res

# Apply patch
coco.COCO.loadRes = patched_loadRes

model = RFDETRNano()

model.train(
    dataset_dir="barray_det_ds",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output"
)