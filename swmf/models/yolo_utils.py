from typing import List
import torch


DEFAULT_YOLO_OBJ_THRESHOLD = 0.25  # The default confidence threshold
DEFAULT_YOLO_IOU_THRESHOLD = 0.45  # The default IOU threshold for NMS


YOLO_IMG_BOX_MSK = torch.Tensor([1, 0, 1, 1, 1, 1, 0]).to(bool)


def postproc_yolo_outputs(yolo_p, imgs_n: int = 0, detach: bool = True):
    """
    Postproc the YOLO results and convert.

    Args:
        yolo_p (Any) : The YOLO predictions
        imgs_n (int) : The current number of images processed
        detach (bool): Whether to detach the tensor or not 
    
    Note:
        i_format = N * [K, 6]
        o_format = [P, 7]
    """
    pp = []
    for i, r in enumerate(yolo_p):
        if detach:
            conf = r.boxes.conf.detach().cpu()
            box = r.boxes.xyxy.detach().cpu()
            cls = r.boxes.cls.detach().cpu()
        else:
            conf = r.boxes.conf
            box = r.boxes.xyxy
            cls = r.boxes.cls
        img = torch.full(cls.shape, i+imgs_n, device=cls.device)
        pp.append(torch.cat([img[:, None], cls[:, None], box, conf[:, None]], dim=1))
    pp = torch.cat(pp, dim=0)
    return pp


def postproc_yolo_targets(yolo_t, imgs_n: int = 0):
    """
    Postproc the YOLO targets labels.

    Note:
        i_format = [N, L, 5]
        o_format = [M, 6]
    """
    ll = []
    for i, t in enumerate(yolo_t):
        ll.append(torch.cat([torch.full((t.shape[0], 1), i+imgs_n), t], dim=1))
    ll = torch.cat(ll, dim=0)
    return ll


def get_model_id_from_model(model) -> str:
    """
    """
    if hasattr(model, "model_name"):
        model_id = "_".join(str(model.model_name).replace('_', '').split('.')[-2].split('data/models/')[1].split('/'))
    else:
        model_id = model.__class__.__name__

    return model_id