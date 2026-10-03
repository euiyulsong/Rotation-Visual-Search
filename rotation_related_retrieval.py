#!/usr/bin/env python3
"""
Related-fashion image retrieval under image rotation.

Compare:
1) baseline: rotated query -> CLIP -> cosine search
2) rot_classifier: predict 0/90/180/270 -> inverse rotate -> CLIP -> search
3) max4: query rotations [0,90,180,270] -> 4 embeddings -> per-gallery max cosine

Dataset:
    ashraq/fashion-product-images-small

Query/gallery IDs are disjoint, so this is RELATED-image retrieval, not self-retrieval.

Graded relevance for nDCG:
    3 = same articleType + gender + baseColour
    2 = same articleType + gender
    1 = same subCategory + gender
    0 = otherwise

Binary relevance for P@K / Recall@K / mAP:
    gain >= 2
"""

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    import open_clip
except ImportError as e:
    raise SystemExit(
        "Install dependencies:\n"
        "python3 -m pip install --user -U datasets open_clip_torch pandas tqdm pillow"
    ) from e


DATASET_NAME = "ashraq/fashion-product-images-small"
ROT_CLASSES = (0, 90, 180, 270)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_rgb(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.fromarray(np.asarray(x)).convert("RGB")


def rotate_pil(img: Image.Image, angle: int) -> Image.Image:
    return TF.rotate(
        img,
        float(angle),
        interpolation=transforms.InterpolationMode.BICUBIC,
        expand=False,
        fill=255,
    )


def norm_text(x) -> str:
    return "" if x is None else str(x).strip()


@dataclass
class Item:
    idx: int
    product_id: int
    gender: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: str
    image: Image.Image

    @staticmethod
    def from_row(idx: int, row: dict) -> "Item":
        return Item(
            idx=idx,
            product_id=int(row["id"]),
            gender=norm_text(row.get("gender")),
            masterCategory=norm_text(row.get("masterCategory")),
            subCategory=norm_text(row.get("subCategory")),
            articleType=norm_text(row.get("articleType")),
            baseColour=norm_text(row.get("baseColour")),
            image=pil_rgb(row["image"]),
        )


def load_items(category: str | None) -> List[Item]:
    print(f"[data] loading {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"[data] raw rows={len(ds):,}")

    items = []
    for i, row in enumerate(ds):
        if category and norm_text(row.get("masterCategory")) != category:
            continue
        if not norm_text(row.get("articleType")):
            continue
        if not norm_text(row.get("subCategory")):
            continue
        if not norm_text(row.get("gender")):
            continue
        items.append(Item.from_row(i, row))

    print(f"[data] category={category or 'ALL'}, usable rows={len(items):,}")
    return items


def sample_disjoint_splits(
    items: Sequence[Item],
    rot_train_n: int,
    rot_val_n: int,
    gallery_n: int,
    query_n: int,
    seed: int,
) -> Tuple[List[Item], List[Item], List[Item], List[Item]]:
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)

    required = rot_train_n + rot_val_n + gallery_n + query_n
    if len(pool) < required:
        raise ValueError(
            f"Need {required:,} rows but only have {len(pool):,}. "
            "Reduce split sizes."
        )

    rot_train = pool[:rot_train_n]
    rot_val = pool[rot_train_n:rot_train_n + rot_val_n]

    start = rot_train_n + rot_val_n
    gallery = pool[start:start + gallery_n]

    rel_counts: Dict[Tuple[str, str], int] = {}
    for g in gallery:
        key = (g.articleType, g.gender)
        rel_counts[key] = rel_counts.get(key, 0) + 1

    gallery_ids = {g.product_id for g in gallery}
    queries = []

    for q in pool[start + gallery_n:]:
        if q.product_id in gallery_ids:
            continue
        if rel_counts.get((q.articleType, q.gender), 0) == 0:
            continue
        queries.append(q)
        if len(queries) >= query_n:
            break

    if len(queries) < query_n:
        raise ValueError(
            f"Only found {len(queries)} valid queries with at least one "
            "same-articleType+gender item in gallery."
        )

    assert not ({q.product_id for q in queries} & gallery_ids)

    print(
        f"[split] rotation train={len(rot_train):,}, "
        f"rotation val={len(rot_val):,}, "
        f"gallery={len(gallery):,}, query={len(queries):,}"
    )

    return rot_train, rot_val, gallery, queries


ROT_TRAIN_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

ROT_EVAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class RotationDataset(Dataset):
    def __init__(self, items: Sequence[Item], train: bool):
        self.items = list(items)
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        label = random.randrange(4) if self.train else idx % 4
        angle = ROT_CLASSES[label]
        img = rotate_pil(item.image, angle)
        tfm = ROT_TRAIN_TFM if self.train else ROT_EVAL_TFM
        return tfm(img), label


def build_rotation_model() -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


@torch.no_grad()
def rotation_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.numel()
    return correct / max(total, 1)


def train_rotation_model(
    train_items,
    val_items,
    ckpt_path,
    epochs,
    batch_size,
    lr,
    device,
    workers,
):
    ckpt = Path(ckpt_path)

    if ckpt.exists():
        print(f"[rotation] loading checkpoint: {ckpt}")
        model = build_rotation_model()
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        return model.to(device).eval()

    train_loader = DataLoader(
        RotationDataset(train_items, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        RotationDataset(val_items, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    model = build_rotation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"rot train {epoch}/{epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += y.numel()

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc = rotation_accuracy(model, val_loader, device)

        print(
            f"[rotation] epoch={epoch} "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": ROT_CLASSES,
            "val_acc": best_acc,
        },
        ckpt,
    )
    print(f"[rotation] saved: {ckpt}")
    return model.eval()


@torch.no_grad()
def predict_rotation_labels(model, images, device, batch_size):
    preds = []
    t0 = time.perf_counter()

    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size]
        x = torch.stack([ROT_EVAL_TFM(im) for im in batch]).to(
            device, non_blocking=True
        )
        preds.extend(model(x).argmax(1).cpu().tolist())

    return np.asarray(preds), time.perf_counter() - t0


def load_clip(model_name, pretrained, device):
    print(f"[clip] loading {model_name} / {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_images(
    model,
    preprocess,
    images,
    device,
    batch_size,
    desc,
):
    out = []
    t0 = time.perf_counter()

    for s in tqdm(range(0, len(images), batch_size), desc=desc):
        batch = images[s:s + batch_size]
        x = torch.stack([preprocess(im) for im in batch]).to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            z = model.encode_image(x)

        z = F.normalize(z.float(), dim=-1)
        out.append(z.cpu())

    return torch.cat(out), time.perf_counter() - t0


def relevance_gain(query: Item, gallery: Item) -> int:
    if query.product_id == gallery.product_id:
        return 0

    same_gender = query.gender == gallery.gender
    same_article = query.articleType == gallery.articleType
    same_subcat = query.subCategory == gallery.subCategory
    same_color = query.baseColour == gallery.baseColour

    if same_article and same_gender and same_color:
        return 3
    if same_article and same_gender:
        return 2
    if same_subcat and same_gender:
        return 1
    return 0


def gains_for_query(query, gallery):
    return np.asarray(
        [relevance_gain(query, g) for g in gallery],
        dtype=np.int8,
    )


def dcg(gains):
    gains = gains.astype(np.float64)
    if len(gains) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum((2.0 ** gains - 1.0) * discounts))


def query_metrics(scores, gains, ks, binary_threshold=2):
    order = np.argsort(-scores, kind="stable")
    rg = gains[order]
    binary = (gains >= binary_threshold).astype(np.int32)
    rb = binary[order]
    nrel = int(binary.sum())

    out = {}

    ideal = np.sort(gains)[::-1]

    for k in ks:
        kk = min(k, len(scores))
        out[f"nDCG@{k}"] = (
            dcg(rg[:kk]) / dcg(ideal[:kk])
            if dcg(ideal[:kk]) > 0 else 0.0
        )
        hits = int(rb[:kk].sum())
        out[f"P@{k}"] = hits / max(kk, 1)
        out[f"R@{k}"] = hits / nrel if nrel > 0 else 0.0

    if nrel > 0:
        cumulative = np.cumsum(rb)
        ranks = np.arange(1, len(rb) + 1)
        precisions = cumulative / ranks
        out["AP"] = float((precisions * rb).sum() / nrel)
    else:
        out["AP"] = 0.0

    out["num_relevant"] = float(nrel)
    return out


def evaluate_scores(score_matrix, queries, gallery, ks):
    score_np = score_matrix.numpy()
    acc: Dict[str, List[float]] = {}

    for i, q in enumerate(queries):
        gains = gains_for_query(q, gallery)
        m = query_metrics(score_np[i], gains, ks)
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def cosine_scores(query_emb, gallery_emb, device, batch=256):
    g = gallery_emb.to(device)
    chunks = []

    for s in range(0, len(query_emb), batch):
        q = query_emb[s:s + batch].to(device)
        chunks.append((q @ g.T).cpu())

    return torch.cat(chunks)


def run_baseline(
    query_images,
    clip_model,
    preprocess,
    gallery_emb,
    device,
    batch_size,
):
    qemb, enc_s = encode_images(
        clip_model,
        preprocess,
        query_images,
        device,
        batch_size,
        "baseline query embeddings",
    )
    t0 = time.perf_counter()
    scores = cosine_scores(qemb, gallery_emb, device)
    return scores, enc_s + (time.perf_counter() - t0)


def run_rot_classifier(
    query_images,
    true_angle,
    rot_model,
    clip_model,
    preprocess,
    gallery_emb,
    device,
    rot_batch,
    eval_batch,
):
    labels, rot_s = predict_rotation_labels(
        rot_model,
        query_images,
        device,
        rot_batch,
    )

    predicted_angles = [ROT_CLASSES[int(i)] for i in labels]
    corrected = [
        rotate_pil(img, -pred_angle)
        for img, pred_angle in zip(query_images, predicted_angles)
    ]

    qemb, enc_s = encode_images(
        clip_model,
        preprocess,
        corrected,
        device,
        eval_batch,
        "rotation-corrected embeddings",
    )

    t0 = time.perf_counter()
    scores = cosine_scores(qemb, gallery_emb, device)
    sim_s = time.perf_counter() - t0

    true_class = ROT_CLASSES.index(true_angle % 360)
    rot_acc = float(np.mean(labels == true_class))

    return scores, rot_s + enc_s + sim_s, rot_acc


def run_max4(
    query_images,
    clip_model,
    preprocess,
    gallery_emb,
    device,
    batch_size,
):
    t0 = time.perf_counter()
    all_scores = []

    for rel_angle in ROT_CLASSES:
        imgs = [rotate_pil(im, rel_angle) for im in query_images]

        emb, _ = encode_images(
            clip_model,
            preprocess,
            imgs,
            device,
            batch_size,
            f"max4 relative {rel_angle}",
        )

        all_scores.append(
            cosine_scores(emb, gallery_emb, device)
        )

    scores = torch.stack(all_scores, dim=0).max(dim=0).values
    return scores, time.perf_counter() - t0


def format_metrics(m, k):
    return (
        f"nDCG@{k}={m[f'nDCG@{k}']:.4f} "
        f"P@{k}={m[f'P@{k}']:.4f} "
        f"R@{k}={m[f'R@{k}']:.4f} "
        f"mAP={m['AP']:.4f}"
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--category", default="Apparel")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--rot-train-n", type=int, default=8000)
    p.add_argument("--rot-val-n", type=int, default=1000)

    p.add_argument("--gallery-n", type=int, default=10000)
    p.add_argument("--query-n", type=int, default=1000)

    p.add_argument("--rot-epochs", type=int, default=3)
    p.add_argument("--rot-batch", type=int, default=128)
    p.add_argument("--rot-lr", type=float, default=1e-4)
    p.add_argument(
        "--rot-ckpt",
        default="ckpt/related_rotation_resnet18.pt",
    )

    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument(
        "--clip-pretrained",
        default="laion2b_s34b_b79k",
    )

    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument(
        "--angles",
        nargs="+",
        type=int,
        default=[0, 90, 180, 270],
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[5, 10, 20],
    )
    p.add_argument("--report-k", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out",
        default="results/related_rotation_retrieval.csv",
    )

    args = p.parse_args()

    if args.report_k not in args.ks:
        args.ks.append(args.report_k)

    for a in args.angles:
        if a % 90 != 0:
            raise ValueError(
                "This version uses a 4-class rotation model. "
                "--angles must be multiples of 90."
            )

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}")

    items = load_items(args.category)

    rot_train, rot_val, gallery, queries = sample_disjoint_splits(
        items,
        args.rot_train_n,
        args.rot_val_n,
        args.gallery_n,
        args.query_n,
        args.seed,
    )

    print("\n[relevance]")
    print("gain=3: same articleType + gender + baseColour")
    print("gain=2: same articleType + gender")
    print("gain=1: same subCategory + gender")
    print("gain=0: otherwise")
    print("P/R/mAP relevant condition: gain >= 2")
    print("query/gallery IDs are disjoint\n")

    rot_model = train_rotation_model(
        rot_train,
        rot_val,
        args.rot_ckpt,
        args.rot_epochs,
        args.rot_batch,
        args.rot_lr,
        device,
        args.workers,
    )

    clip_model, preprocess = load_clip(
        args.clip_model,
        args.clip_pretrained,
        device,
    )

    gallery_emb, gallery_s = encode_images(
        clip_model,
        preprocess,
        [x.image for x in gallery],
        device,
        args.eval_batch,
        "gallery embeddings",
    )
    print(f"[gallery] {len(gallery):,} images in {gallery_s:.2f}s")

    rows = []
    ks = sorted(set(args.ks))

    for angle in args.angles:
        angle %= 360
        print(f"\n===== QUERY ROTATION {angle}° =====")

        query_images = [
            rotate_pil(q.image, angle)
            for q in queries
        ]

        scores, elapsed = run_baseline(
            query_images,
            clip_model,
            preprocess,
            gallery_emb,
            device,
            args.eval_batch,
        )
        m = evaluate_scores(scores, queries, gallery, ks)
        row = {
            "method": "baseline",
            "angle": angle,
            **m,
            "ms_per_query": 1000 * elapsed / len(queries),
            "rot_acc": "",
        }
        rows.append(row)
        print(
            f"baseline        {format_metrics(m, args.report_k)} "
            f"ms/q={row['ms_per_query']:.2f}"
        )

        scores, elapsed, rot_acc = run_rot_classifier(
            query_images,
            angle,
            rot_model,
            clip_model,
            preprocess,
            gallery_emb,
            device,
            args.rot_batch,
            args.eval_batch,
        )
        m = evaluate_scores(scores, queries, gallery, ks)
        row = {
            "method": "rot_classifier",
            "angle": angle,
            **m,
            "ms_per_query": 1000 * elapsed / len(queries),
            "rot_acc": rot_acc,
        }
        rows.append(row)
        print(
            f"rot_classifier  {format_metrics(m, args.report_k)} "
            f"ms/q={row['ms_per_query']:.2f} "
            f"rot_acc={rot_acc:.4f}"
        )

        scores, elapsed = run_max4(
            query_images,
            clip_model,
            preprocess,
            gallery_emb,
            device,
            args.eval_batch,
        )
        m = evaluate_scores(scores, queries, gallery, ks)
        row = {
            "method": "max4",
            "angle": angle,
            **m,
            "ms_per_query": 1000 * elapsed / len(queries),
            "rot_acc": "",
        }
        rows.append(row)
        print(
            f"max4            {format_metrics(m, args.report_k)} "
            f"ms/q={row['ms_per_query']:.2f}"
        )

    print("\n=== Non-zero rotation average ===")

    nonzero = [r for r in rows if int(r["angle"]) != 0]

    for method in ("baseline", "rot_classifier", "max4"):
        rr = [r for r in nonzero if r["method"] == method]
        if not rr:
            continue

        print(
            f"{method:16s} "
            f"nDCG@{args.report_k}="
            f"{np.mean([r[f'nDCG@{args.report_k}'] for r in rr]):.4f} "
            f"P@{args.report_k}="
            f"{np.mean([r[f'P@{args.report_k}'] for r in rr]):.4f} "
            f"R@{args.report_k}="
            f"{np.mean([r[f'R@{args.report_k}'] for r in rr]):.4f} "
            f"mAP={np.mean([r['AP'] for r in rr]):.4f} "
            f"ms/query={np.mean([r['ms_per_query'] for r in rr]):.2f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out}")

    print("\nInterpretation:")
    print("- This is related-item retrieval; query IDs never appear in the gallery.")
    print("- Compare non-zero-rotation nDCG@K, P@K and mAP.")
    print("- rot_classifier ~= max4: prefer classifier because it is cheaper.")
    print("- max4 clearly > rot_classifier: max-cosine is more robust but costs ~4 CLIP encodes.")
    print("- baseline ~= corrected methods: extra rotation handling is unnecessary.")


if __name__ == "__main__":
    main()
