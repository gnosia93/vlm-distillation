"""SAM3 PCS 학습용 set-prediction 손실.

`Sam3Model.forward` 는 loss 를 반환하지 않는다 (labels 인자가 없다). SAM3 는
DETR 계열 head 를 쓰므로 Hungarian matching 기반 손실을 직접 붙여 준다.

구성:
  - matching:  focal-style class cost + L1 box cost + GIoU box cost
  - class:     sigmoid focal loss, query 200개 전체에 대해 (매칭된 것만 양성)
  - box:       매칭된 query 에 L1 + GIoU
  - mask:      매칭된 query 에 sigmoid focal + dice
  - presence:  이미지에 프롬프트 객체가 존재하는지 BCE (SAM3 의 presence head)

presence head 를 별도로 학습하는 게 핵심이다. SAM3 는 인식(이 개념이 이미지에
있나?)과 위치추정(어디에 있나?)을 분리하고, 최종 점수를
`pred_logits.sigmoid() * presence_logits.sigmoid()` 로 만든다. presence 를
학습하지 않으면 hard negative 이미지에서 점수가 내려가지 않는다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(n, hw) 단위 dice loss. 인스턴스별 loss 벡터를 반환한다."""
    probs = logits.sigmoid()
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


class Sam3SetCriterion(torch.nn.Module):
    """Hungarian matching + PCS 손실.

    Args:
        cost_class / cost_bbox / cost_giou: matching 비용 가중치.
        loss_class / loss_bbox / loss_giou / loss_mask / loss_dice / loss_presence:
            최종 손실 가중치.
        mask_points: 마스크 손실을 계산할 때 샘플링할 픽셀 수. 마스크 전체
            (288*288 = 82944) 를 쓰면 메모리를 많이 먹으므로 Mask2Former 처럼
            무작위 점 샘플링을 한다. None 이면 전체 픽셀 사용.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_class: float = 2.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        loss_mask: float = 5.0,
        loss_dice: float = 5.0,
        loss_presence: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mask_points: int | None = 12544,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.weights = {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_presence": loss_presence,
        }
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mask_points = mask_points

    @torch.no_grad()
    def _match(
        self,
        pred_logits: torch.Tensor,  # (b, q)
        pred_boxes: torch.Tensor,  # (b, q, 4) xyxy normalized
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        indices = []
        for logits, boxes, target in zip(pred_logits, pred_boxes, targets):
            tgt_boxes = target["boxes"].to(boxes.device)
            if len(tgt_boxes) == 0:
                empty = torch.zeros(0, dtype=torch.long, device=boxes.device)
                indices.append((empty, empty))
                continue

            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)

            # DETR 의 focal 기반 클래스 비용. 클래스가 1개(=프롬프트 일치)뿐이므로
            # 모든 target 에 대해 같은 값이고, query 축으로만 변한다.
            prob = logits.sigmoid()
            neg = (1 - self.focal_alpha) * (prob**self.focal_gamma) * (-(1 - prob + 1e-8).log())
            pos = self.focal_alpha * ((1 - prob) ** self.focal_gamma) * (-(prob + 1e-8).log())
            cost_class = (pos - neg).unsqueeze(1).expand(-1, len(tgt_boxes))

            cost_bbox = torch.cdist(boxes, tgt_xyxy, p=1)
            cost_giou = -generalized_box_iou(boxes, tgt_xyxy)

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            row, col = linear_sum_assignment(cost.float().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row, dtype=torch.long, device=boxes.device),
                    torch.as_tensor(col, dtype=torch.long, device=boxes.device),
                )
            )
        return indices

    def forward(self, outputs, targets: list[dict]) -> dict[str, torch.Tensor]:
        pred_logits = outputs.pred_logits  # (b, q)
        pred_boxes = outputs.pred_boxes  # (b, q, 4) xyxy, [0,1]
        pred_masks = outputs.pred_masks  # (b, q, mh, mw) logits
        presence_logits = outputs.presence_logits  # (b, 1)

        device = pred_logits.device
        indices = self._match(pred_logits, pred_boxes, targets)

        # DETR 관례대로 인스턴스 개수로 정규화한다. 단, 배치 전체가 hard negative
        # 이면 num_boxes 가 1 로 떨어져 classification 항만 수십 배로 튄다.
        # batch_size 로 하한을 두어 그 경우를 막는다 (양성 배치에서는 보통
        # num_boxes >= batch_size 라서 동작이 바뀌지 않는다).
        num_boxes = max(sum(len(t["boxes"]) for t in targets), len(targets), 1)

        # ---- classification: query 전체에 focal loss ----
        target_classes = torch.zeros_like(pred_logits)
        for i, (row, _) in enumerate(indices):
            target_classes[i, row] = 1.0
        loss_class = (
            sigmoid_focal_loss(
                pred_logits, target_classes, self.focal_alpha, self.focal_gamma, reduction="sum"
            )
            / num_boxes
        )

        # ---- presence: 프롬프트 객체가 이미지에 있는지 ----
        presence_target = torch.tensor(
            [[1.0 if len(t["boxes"]) else 0.0] for t in targets], device=device, dtype=presence_logits.dtype
        )
        loss_presence = F.binary_cross_entropy_with_logits(presence_logits, presence_target)

        # ---- box / mask: 매칭된 query 만 ----
        src_boxes, tgt_boxes = [], []
        src_masks, tgt_masks = [], []
        for i, (row, col) in enumerate(indices):
            if len(row) == 0:
                continue
            src_boxes.append(pred_boxes[i, row])
            tgt_boxes.append(box_cxcywh_to_xyxy(targets[i]["boxes"].to(device))[col])
            src_masks.append(pred_masks[i, row])
            tgt_masks.append(targets[i]["masks"].to(device)[col])

        zero = pred_logits.sum() * 0.0  # graph 를 유지하는 0
        loss_bbox = loss_giou = loss_mask = loss_dice = zero

        if src_boxes:
            sb = torch.cat(src_boxes)
            tb = torch.cat(tgt_boxes)
            loss_bbox = F.l1_loss(sb, tb, reduction="sum") / num_boxes
            loss_giou = (1 - torch.diag(generalized_box_iou(sb, tb))).sum() / num_boxes

            sm = torch.cat(src_masks)  # (n, mh, mw)
            tm = torch.cat(tgt_masks)

            # 예측 마스크와 GT 마스크 해상도가 다를 수 있으므로 GT 를 맞춰 준다.
            if tm.shape[-2:] != sm.shape[-2:]:
                tm = F.interpolate(tm.unsqueeze(1), size=sm.shape[-2:], mode="nearest").squeeze(1)

            sm = sm.flatten(1)
            tm = tm.flatten(1)

            if self.mask_points is not None and sm.shape[1] > self.mask_points:
                idx = torch.randperm(sm.shape[1], device=device)[: self.mask_points]
                sm = sm[:, idx]
                tm = tm[:, idx]

            loss_mask = (
                sigmoid_focal_loss(sm, tm, self.focal_alpha, self.focal_gamma, reduction="none")
                .mean(1)
                .sum()
                / num_boxes
            )
            loss_dice = dice_loss(sm, tm).sum() / num_boxes

        losses = {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "loss_presence": loss_presence,
        }
        losses["loss"] = sum(self.weights[k] * v for k, v in losses.items())
        return losses
