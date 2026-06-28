import os, random, json, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CFG = dict(
    model_name = "roberta-base",
    max_length = 128,
    batch_size = 64,
    num_epochs = 10,
    lr = 2e-5,
    weight_decay = 1e-3,
    warmup_ratio = 0.1,
    proj_dim = 256,
    temperature = 0.5,
    alpha = 0.3,    # меньше чем в one-hot — мягкий CCL более стабилен
    beta = 0.3,     # меньше чем в one-hot — мягкий LDL более стабилен
    focal_gamma = 2.0,
    seed = 42,
    data_dir = "./data",
    output_dir = "./output_03_softlabel_weighted_sampling",
    device = "cuda",
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device(CFG["device"] if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NUM_LABELS = len(EMOTIONS)

EMOTION_DESCRIPTIONS = {
    "admiration": "feeling of deep respect or warm approval towards someone or something",
    "amusement": "feeling of being entertained or finding something funny",
    "anger": "strong feeling of displeasure or hostility",
    "annoyance": "mild anger or irritation caused by something",
    "approval": "believing that something is good or acceptable",
    "caring": "feeling concern and interest in the wellbeing of others",
    "confusion": "lack of understanding or feeling uncertain about something",
    "curiosity": "strong desire to know or learn something",
    "desire": "strong wish or longing for something",
    "disappointment": "feeling sad because something was not as good as expected",
    "disapproval": "feeling that something is wrong or bad",
    "disgust": "strong feeling of dislike or revulsion",
    "embarrassment": "feeling of shame or awkwardness in a social situation",
    "excitement": "feeling of enthusiasm and eagerness",
    "fear": "feeling of being afraid or anxious about danger",
    "gratitude": "feeling of thankfulness and appreciation",
    "grief": "deep sorrow caused by loss",
    "joy": "feeling of great happiness and pleasure",
    "love": "deep affection and attachment towards someone",
    "nervousness": "feeling of anxiety or being uneasy",
    "optimism": "hopefulness and confidence about the future",
    "pride": "feeling of satisfaction from one's achievements",
    "realization": "becoming aware or understanding something clearly",
    "relief": "feeling of reassurance after anxiety or distress ends",
    "remorse": "deep regret or guilt about a past action",
    "sadness": "feeling of sorrow or unhappiness",
    "surprise": "feeling of shock or astonishment at something unexpected",
    "neutral": "no particular emotional state, calm and indifferent",
}

FILES = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

import urllib.request
os.makedirs(CFG["data_dir"], exist_ok=True)
BASE_URL = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/"
for fname in FILES:
    dst = os.path.join(CFG["data_dir"], fname)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        log.info("Downloading %s...", fname)
        urllib.request.urlretrieve(BASE_URL + fname, dst)
    else:
        log.info("%s already exists", fname)

SPLIT_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
for split in ["train", "dev", "test"]:
    dst = os.path.join(CFG["data_dir"], f"{split}.tsv")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(SPLIT_URL + f"{split}.tsv", dst)

def load_goemotions_full(data_dir):
    """Строит soft-label из full_dataset: среднее по аннотаторам."""
    dfs = []
    for fname in FILES:
        df = pd.read_csv(os.path.join(data_dir, fname))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    grouped = df.groupby("id")[EMOTIONS].mean().reset_index()
    grouped["label_vec"] = grouped[EMOTIONS].values.tolist()
    grouped["label_vec"] = grouped["label_vec"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )
    text_df = df[["id", "text"]].drop_duplicates(subset="id")
    result = grouped[["id", "label_vec"]].merge(text_df, on="id")
    return result

raw_df = load_goemotions_full(CFG["data_dir"])
soft_label_map = dict(zip(raw_df["id"], raw_df["label_vec"]))
log.info("Total unique examples: %d", len(raw_df))

sample = np.stack(raw_df["label_vec"].values)
log.info("Soft-label stats: mean=%.3f max=%.3f non-binary=%.3f",
    sample.mean(), sample.max(),
    ((sample > 0) & (sample < 1)).any(axis=1).mean()
)

def make_onehot(label_str, num_labels=NUM_LABELS):
    vec = np.zeros(num_labels, dtype=np.float32)
    if not isinstance(label_str, str):
        return vec
    for idx in label_str.split(","):
        idx = idx.strip()
        if idx.isdigit():
            vec[int(idx)] = 1.0
    return vec

def load_split(split_name):
    path = os.path.join(CFG["data_dir"], f"{split_name}.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "id"])
    soft_df = pd.DataFrame(list(soft_label_map.items()), columns=["id", "label_vec"])
    df = df.merge(soft_df, on="id", how="left")
    missing = df["label_vec"].isna()
    if missing.any():
        log.warning("%d examples in %s: fallback to one-hot", missing.sum(), split_name)
        df.loc[missing, "label_vec"] = df.loc[missing, "labels"].apply(make_onehot)
    return df

train_df = load_split("train")
val_df = load_split("dev")
test_df = load_split("test")
log.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))

train_labels = np.stack(train_df["label_vec"].values)
class_freq = train_labels.mean(axis=0)
class_weights = 1.0 / np.sqrt(class_freq + 1e-6)
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
log.info("Class weights min=%.2f mean=%.2f max=%.2f", class_weights.min(), class_weights.mean(), class_weights.max())

def compute_uncertainty_weights(label_vecs):
    eps = 1e-7
    p = np.clip(label_vecs, eps, 1 - eps)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    active = label_vecs > 0.1
    weights = np.where(
        active.sum(axis=1) > 0,
        1.0 - (H * active).sum(axis=1) / (active.sum(axis=1) + 1e-6),
        0.5,
    )
    weights = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    return weights.astype(np.float32)

train_sample_weights = compute_uncertainty_weights(train_labels)
log.info("Sample weights min=%.3f mean=%.3f max=%.3f", train_sample_weights.min(), train_sample_weights.mean(), train_sample_weights.max())

tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

label_texts = [f"{e} [EMO] {EMOTION_DESCRIPTIONS[e]}" for e in EMOTIONS]
label_encodings = tokenizer(
    label_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
)

class GoEmotionsSoftDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, sample_weights=None):
        self.texts = df["text"].tolist()
        self.labels = np.stack(df["label_vec"].values)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item

train_dataset = GoEmotionsSoftDataset(train_df, tokenizer, CFG["max_length"], train_sample_weights)
val_dataset = GoEmotionsSoftDataset(val_df, tokenizer, CFG["max_length"])
test_dataset = GoEmotionsSoftDataset(test_df, tokenizer, CFG["max_length"])

sampler = WeightedRandomSampler(
    weights=torch.tensor(train_sample_weights, dtype=torch.float),
    num_samples=len(train_dataset),
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
log.info("Train batches: %d", len(train_loader))

class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TACOSoftLabel(nn.Module):
    def __init__(self, model_name, num_labels, proj_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fcn1 = FCN(hidden, proj_dim)
        self.fcn2 = FCN(hidden, proj_dim)
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask,
                label_input_ids=None, label_attention_mask=None):
        cls_inst = self.encode_cls(input_ids, attention_mask)
        hu = F.normalize(self.fcn1(cls_inst), dim=-1)

        he = None
        scores = None
        if label_input_ids is not None:
            cls_label = self.encode_cls(label_input_ids, label_attention_mask)
            he = F.normalize(self.fcn2(cls_label), dim=-1)
            scores = self.scale * (hu @ he.t())

        return scores, hu, he

def soft_focal_bce_loss(scores, labels_soft, class_weights, sample_weights=None, gamma=2.0):
    """Focal BCE на мягких метках + взвешивание по обратной частоте класса."""
    probs = torch.sigmoid(scores)
    focal_weight = (1 - probs) ** gamma * labels_soft + probs ** gamma * (1 - labels_soft)
    bce = F.binary_cross_entropy_with_logits(scores, labels_soft, reduction="none")
    weighted = bce * focal_weight * class_weights.unsqueeze(0)
    if sample_weights is not None:
        weighted = weighted * sample_weights.unsqueeze(1)
    return weighted.mean()


def soft_ccl_loss(hu, labels_soft, temperature):
    """
    Улучшение 2: мягкий CCL.
    Вес пары (i,j) = cosine similarity их soft-label векторов.
    Кластеризация по-прежнему определяет hard pairs, но граница мягкая.
    """
    B = hu.size(0)
    X = hu.detach().cpu().numpy()
    n_clusters = min(max(2, B // 4), NUM_LABELS)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0, max_iter=50)
    cluster_ids = torch.tensor(km.fit_predict(X), device=hu.device)

    # мягкое сходство меток
    l_norm = F.normalize(labels_soft, dim=-1)
    label_sim = (l_norm @ l_norm.t()).clamp(min=0)  # (B, B) в [0, 1]

    sim = (hu @ hu.t()) / temperature
    loss_sum = torch.tensor(0.0, device=hu.device, requires_grad=False)
    count = 0

    for i in range(B):
        # hard positive: похожие метки (sim > 0.3), разные кластеры
        pos_mask = (label_sim[i] > 0.3) & (cluster_ids != cluster_ids[i])
        pos_mask[i] = False
        # hard negative: непохожие метки (sim < 0.1), один кластер
        neg_mask = (label_sim[i] < 0.1) & (cluster_ids == cluster_ids[i])

        if pos_mask.sum() == 0:
            continue

        denom_mask = pos_mask | neg_mask
        if denom_mask.sum() == 0:
            continue

        log_denom = torch.logsumexp(sim[i][denom_mask], dim=0)
        # взвешиваем позитивные пары по силе сходства меток
        for p in pos_mask.nonzero(as_tuple=True)[0]:
            weight = label_sim[i, p].detach()
            loss_sum = loss_sum + weight * (-(sim[i, p] - log_denom))
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=hu.device, requires_grad=True)
    return loss_sum / count


def soft_ldl_loss(scores, he, labels_soft):
    """
    Улучшение 3: мягкий LDL.
    confusability взвешена средними soft-label значениями —
    метки которые часто встречаются вместе получают больший штраф.
    """
    probs = torch.sigmoid(scores)  # (B, C)

    # confusability из предсказаний (как в статье)
    M_norm = F.normalize(probs, dim=0)
    cos_cols = M_norm.t() @ M_norm  # (C, C)

    # дополнительный вес: среднее совместное встречание в soft-labels
    mean_soft = labels_soft.mean(dim=0)  # (C,)
    cooccur = mean_soft.unsqueeze(1) * mean_soft.unsqueeze(0)  # (C, C) — outer product
    cooccur = cooccur / (cooccur.sum().clamp(min=1e-8))

    # итоговая confusability = смесь предсказаний и co-occurrence
    f = 0.7 * cos_cols / (cos_cols.sum().clamp(min=1e-8)) + 0.3 * cooccur

    cos_labels = he @ he.t()
    loss = (f * cos_labels).sum().clamp(min=-10, max=10)
    return loss

model = TACOSoftLabel(
    model_name=CFG["model_name"],
    num_labels=NUM_LABELS,
    proj_dim=CFG["proj_dim"],
).to(DEVICE)

label_input_ids = label_encodings["input_ids"].to(DEVICE)
label_attention_mask = label_encodings["attention_mask"].to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
total_steps = len(train_loader) * CFG["num_epochs"]
warmup_steps = int(total_steps * CFG["warmup_ratio"])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def evaluate_with_threshold(model, loader, device, thresholds=None):
    """Eval с per-class порогами (если None — используем 0.5)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = (batch["labels"].numpy() > 0.5).astype(int)
            scores, _, _ = model(ids, mask, label_input_ids, label_attention_mask)
            probs = torch.sigmoid(scores).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(lbls)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    if thresholds is None:
        thresholds = np.full(NUM_LABELS, 0.5)

    preds = (all_probs > thresholds).astype(int)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    return macro_f1, micro_f1, all_probs, all_labels


def find_best_thresholds(probs, labels):
    """Подбирает оптимальный порог per-class по F1."""
    best_thresholds = np.full(NUM_LABELS, 0.5)
    for i in range(NUM_LABELS):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(labels[:, i], (probs[:, i] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds


best_macro_f1 = 0.0
best_thresholds = np.full(NUM_LABELS, 0.5)
history = []
LOG_EVERY = 50

for epoch in range(CFG["num_epochs"]):
    model.train()
    total_loss = total_main = total_ccl = total_ldl = 0.0
    step = 0

    for batch in train_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        sw = batch.get("sample_weight")
        if sw is not None:
            sw = sw.to(DEVICE)

        scores, hu, he = model(ids, mask, label_input_ids, label_attention_mask)

        l_main = soft_focal_bce_loss(scores, lbls, class_weights_tensor, sw, CFG["focal_gamma"])
        l_ccl = soft_ccl_loss(hu, lbls, CFG["temperature"])
        l_ldl = soft_ldl_loss(scores, he, lbls)
        loss = l_main + CFG["alpha"] * l_ccl + CFG["beta"] * l_ldl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_main += l_main.item()
        total_ccl += l_ccl.item()
        total_ldl += l_ldl.item()
        step += 1

        if step % LOG_EVERY == 0:
            log.info(
                "epoch %d step %d/%d  loss=%.4f main=%.4f ccl=%.4f ldl=%.4f",
                epoch + 1, step, len(train_loader),
                loss.item(), l_main.item(), l_ccl.item(), l_ldl.item(),
            )

    # eval с порогом 0.5 и с оптимальным порогом
    macro_f1_05, micro_f1_05, val_probs, val_labels = evaluate_with_threshold(
        model, val_loader, DEVICE
    )
    thresholds = find_best_thresholds(val_probs, val_labels)
    macro_f1_opt, micro_f1_opt, _, _ = evaluate_with_threshold(
        model, val_loader, DEVICE, thresholds
    )

    n = len(train_loader)
    epoch_log = {
        "epoch": epoch + 1,
        "loss": total_loss / n,
        "l_main": total_main / n,
        "l_ccl": total_ccl / n,
        "l_ldl": total_ldl / n,
        "val_macro_f1_05": macro_f1_05,
        "val_macro_f1_opt": macro_f1_opt,
        "val_micro_f1_opt": micro_f1_opt,
    }
    history.append(epoch_log)
    log.info(
        "epoch %d  loss=%.4f  val_macro_05=%.4f  val_macro_opt=%.4f",
        epoch + 1, total_loss / n, macro_f1_05, macro_f1_opt,
    )

    if macro_f1_opt > best_macro_f1:
        best_macro_f1 = macro_f1_opt
        best_thresholds = thresholds.copy()
        torch.save(model.state_dict(), os.path.join(CFG["output_dir"], "best_model.pt"))
        log.info("best model saved (macro-F1_opt=%.4f)", best_macro_f1)

log.info("best val macro-F1 (opt threshold): %.4f", best_macro_f1)
np.save(os.path.join(CFG["output_dir"], "best_thresholds.npy"), best_thresholds)
log.info("best thresholds: min=%.2f mean=%.2f max=%.2f",
    best_thresholds.min(), best_thresholds.mean(), best_thresholds.max())

model.load_state_dict(
    torch.load(os.path.join(CFG["output_dir"], "best_model.pt"), map_location=DEVICE)
)
test_macro_f1, test_micro_f1, test_probs, test_labels = evaluate_with_threshold(
    model, test_loader, DEVICE, best_thresholds
)
log.info("Test Macro-F1=%.4f  Micro-F1=%.4f (optimal thresholds)", test_macro_f1, test_micro_f1)

test_macro_f1_05, test_micro_f1_05, _, _ = evaluate_with_threshold(
    model, test_loader, DEVICE
)
log.info("Test Macro-F1=%.4f  Micro-F1=%.4f (threshold=0.5)", test_macro_f1_05, test_micro_f1_05)

from sklearn.metrics import classification_report
preds = (test_probs > best_thresholds).astype(int)
print(classification_report(test_labels, preds, target_names=EMOTIONS, zero_division=0))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per_class_f1 = f1_score(test_labels, preds, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(EMOTIONS, per_class_f1, color="steelblue")
ax.set_xticks(range(NUM_LABELS))
ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("F1-score")
ax.set_title("TACO 28 emotions soft-label (optimal thresholds): per-emotion F1")
ax.axhline(per_class_f1.mean(), color="red", linestyle="--",
           label=f"Macro avg={per_class_f1.mean():.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "per_emotion_f1.png"), dpi=150)
plt.close()

hist_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_df["epoch"], hist_df["loss"], label="total")
axes[0].plot(hist_df["epoch"], hist_df["l_main"], label="L_main")
axes[0].plot(hist_df["epoch"], hist_df["l_ccl"], label="L_CCL")
axes[0].plot(hist_df["epoch"], hist_df["l_ldl"], label="L_LDL")
axes[0].set_title("Training losses")
axes[0].legend()
axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1_05"], label="macro-F1 (t=0.5)")
axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1_opt"], label="macro-F1 (opt)")
axes[1].set_title("Validation F1")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "training_curves.png"), dpi=150)
plt.close()

results = {
    "test_macro_f1_opt": round(test_macro_f1, 4),
    "test_macro_f1_05": round(test_macro_f1_05, 4),
    "test_micro_f1_opt": round(test_micro_f1, 4),
    "best_val_macro_f1_opt": round(best_macro_f1, 4),
    "config": CFG,
    "history": history,
}
with open(os.path.join(CFG["output_dir"], "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTest Macro-F1 (opt)={test_macro_f1:.4f}  (t=0.5)={test_macro_f1_05:.4f}")
print("Done. Results saved to", CFG["output_dir"])