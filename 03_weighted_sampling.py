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
    alpha = 0.45,
    beta = 0.5,
    eval_threshold = 0.5,
    focal_gamma = 2.0,
    seed = 42,
    data_dir = "./data",
    output_dir = "./output_03_onehot_weighted_sampling",
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
NUM_LABELS = len(EMOTIONS)  # 28
log.info("Num emotions: %d", NUM_LABELS)

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
    df["label_vec"] = df["labels"].apply(make_onehot)
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

def compute_sample_weights(label_vecs):
    rare_mask = (class_freq < 0.05).astype(np.float32)
    rare_count = (label_vecs * rare_mask).sum(axis=1)
    weights = 1.0 + rare_count
    weights = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    return weights.astype(np.float32)

train_sample_weights = compute_sample_weights(train_labels)
log.info("Sample weights min=%.3f mean=%.3f max=%.3f", train_sample_weights.min(), train_sample_weights.mean(), train_sample_weights.max())

tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

label_texts = [f"{e} [EMO] {EMOTION_DESCRIPTIONS[e]}" for e in EMOTIONS]
label_encodings = tokenizer(
    label_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
)

class GoEmotionsDataset(Dataset):
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

train_dataset = GoEmotionsDataset(train_df, tokenizer, CFG["max_length"], train_sample_weights)
val_dataset = GoEmotionsDataset(val_df, tokenizer, CFG["max_length"])
test_dataset = GoEmotionsDataset(test_df, tokenizer, CFG["max_length"])

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


class TACOMultiLabel(nn.Module):
    """
    Dual-encoder из статьи, адаптированный под multi-label.
    hu, he нормализованы — prediction score = dot product.
    Классификация: sigmoid(scores) > threshold.
    """
    def __init__(self, model_name, num_labels, proj_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.fcn1 = FCN(hidden, proj_dim)
        self.fcn2 = FCN(hidden, proj_dim)
        # масштабирующий параметр для dot-product (learnable temperature)
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
            scores = self.scale * (hu @ he.t())  # (B, C)

        return scores, hu, he

def focal_bce_loss(scores, labels, class_weights, sample_weights=None, gamma=2.0):
    probs = torch.sigmoid(scores)
    focal_weight = (1 - probs) ** gamma * labels + probs ** gamma * (1 - labels)
    bce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    weighted = bce * focal_weight * class_weights.unsqueeze(0)
    if sample_weights is not None:
        weighted = weighted * sample_weights.unsqueeze(1)
    return weighted.mean()


def ccl_loss(hu, labels_onehot, temperature):
    """
    CCL для multi-label: позитивная пара = хотя бы одна общая эмоция.
    Кластеризация на батче определяет hard pairs.
    """
    B = hu.size(0)
    X = hu.detach().cpu().numpy()
    n_clusters = min(max(2, B // 4), NUM_LABELS)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0, max_iter=50)
    cluster_ids = torch.tensor(km.fit_predict(X), device=hu.device)

    # label similarity: есть ли хоть одна общая эмоция
    label_sim = (labels_onehot @ labels_onehot.t()).clamp(max=1.0)  # (B, B)

    sim = (hu @ hu.t()) / temperature  # (B, B)
    loss_sum = 0.0
    count = 0

    for i in range(B):
        # hard positive: общие эмоции, разные кластеры
        pos_mask = (label_sim[i] > 0) & (cluster_ids != cluster_ids[i])
        pos_mask[i] = False
        # hard negative: нет общих эмоций, один кластер
        neg_mask = (label_sim[i] == 0) & (cluster_ids == cluster_ids[i])

        if pos_mask.sum() == 0:
            continue

        denom_mask = pos_mask | neg_mask
        if denom_mask.sum() == 0:
            continue

        log_denom = torch.logsumexp(sim[i][denom_mask], dim=0)
        for p in pos_mask.nonzero(as_tuple=True)[0]:
            loss_sum += -(sim[i, p] - log_denom)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=hu.device, requires_grad=True)
    return loss_sum / count


def ldl_loss(scores, he):
    """
    LDL из статьи: раздвигает эмбеддинги похожих меток.
    confusability(j, j') = cos(sigmoid(M)_:j, sigmoid(M)_:j') / normalization
    """
    probs = torch.sigmoid(scores)  # (B, C) — вероятности
    C = he.size(0)

    # cosine similarity между столбцами (метками) матрицы предсказаний
    M_norm = F.normalize(probs, dim=0)  # normalize по batch
    cos_cols = M_norm.t() @ M_norm  # (C, C)

    cos_sum = cos_cols.sum().clamp(min=1e-8)
    f = cos_cols / cos_sum  # confusability weights

    # cosine similarity между label embeddings
    cos_labels = he @ he.t()  # (C, C)

    # убираем exp — без него градиент не насыщается
    loss = (f * cos_labels).sum().clamp(min=-10, max=10)
    return loss

model = TACOMultiLabel(
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


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].numpy().astype(int)
            scores, _, _ = model(ids, mask, label_input_ids, label_attention_mask)
            preds = (torch.sigmoid(scores).cpu().numpy() > CFG["eval_threshold"]).astype(int)
            all_preds.append(preds)
            all_labels.append(lbls)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    return macro_f1, micro_f1


best_macro_f1 = 0.0
history = []
LOG_EVERY = 50

for epoch in range(CFG["num_epochs"]):
    model.train()
    total_loss = total_bce = total_ccl = total_ldl = 0.0
    step = 0

    for batch in train_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        sw = batch.get("sample_weight")
        if sw is not None:
            sw = sw.to(DEVICE)

        scores, hu, he = model(ids, mask, label_input_ids, label_attention_mask)

        l_bce = focal_bce_loss(scores, lbls, class_weights_tensor, sw, CFG["focal_gamma"])
        l_ccl = ccl_loss(hu, lbls, CFG["temperature"])
        l_ldl = ldl_loss(scores, he)
        loss = l_bce + CFG["alpha"] * l_ccl + CFG["beta"] * l_ldl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_bce += l_bce.item()
        total_ccl += l_ccl.item()
        total_ldl += l_ldl.item()
        step += 1

        if step % LOG_EVERY == 0:
            log.info(
                "epoch %d step %d/%d  loss=%.4f bce=%.4f ccl=%.4f ldl=%.4f",
                epoch + 1, step, len(train_loader),
                loss.item(), l_bce.item(), l_ccl.item(), l_ldl.item(),
            )

    macro_f1, micro_f1 = evaluate(model, val_loader, DEVICE)
    n = len(train_loader)
    epoch_log = {
        "epoch": epoch + 1,
        "loss": total_loss / n,
        "l_bce": total_bce / n,
        "l_ccl": total_ccl / n,
        "l_ldl": total_ldl / n,
        "val_macro_f1": macro_f1,
        "val_micro_f1": micro_f1,
    }
    history.append(epoch_log)
    log.info(
        "epoch %d done  loss=%.4f val_macro=%.4f val_micro=%.4f",
        epoch + 1, total_loss / n, macro_f1, micro_f1,
    )

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(CFG["output_dir"], "best_model.pt"))
        log.info("best model saved (macro-F1=%.4f)", best_macro_f1)

log.info("best val macro-F1: %.4f", best_macro_f1)

model.load_state_dict(
    torch.load(os.path.join(CFG["output_dir"], "best_model.pt"), map_location=DEVICE)
)
test_macro_f1, test_micro_f1 = evaluate(model, test_loader, DEVICE)
log.info("Test Macro-F1=%.4f  Micro-F1=%.4f", test_macro_f1, test_micro_f1)

from sklearn.metrics import classification_report
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].numpy().astype(int)
        scores, _, _ = model(ids, mask, label_input_ids, label_attention_mask)
        preds = (torch.sigmoid(scores).cpu().numpy() > CFG["eval_threshold"]).astype(int)
        all_preds.append(preds)
        all_labels.append(lbls)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
print(classification_report(all_labels, all_preds, target_names=EMOTIONS, zero_division=0))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(EMOTIONS, per_class_f1, color="steelblue")
ax.set_xticks(range(NUM_LABELS))
ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("F1-score")
ax.set_title("TACO 28 emotions one-hot: per-emotion F1")
ax.axhline(per_class_f1.mean(), color="red", linestyle="--",
           label=f"Macro avg={per_class_f1.mean():.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "per_emotion_f1.png"), dpi=150)
plt.close()

hist_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_df["epoch"], hist_df["loss"], label="total")
axes[0].plot(hist_df["epoch"], hist_df["l_bce"], label="L_BCE")
axes[0].plot(hist_df["epoch"], hist_df["l_ccl"], label="L_CCL")
axes[0].plot(hist_df["epoch"], hist_df["l_ldl"], label="L_LDL")
axes[0].set_title("Training losses")
axes[0].legend()
axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1"], label="macro-F1")
axes[1].plot(hist_df["epoch"], hist_df["val_micro_f1"], label="micro-F1")
axes[1].set_title("Validation F1")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(CFG["output_dir"], "training_curves.png"), dpi=150)
plt.close()

results = {
    "test_macro_f1": round(test_macro_f1, 4),
    "test_micro_f1": round(test_micro_f1, 4),
    "best_val_macro_f1": round(best_macro_f1, 4),
    "config": CFG,
    "history": history,
}
with open(os.path.join(CFG["output_dir"], "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTest Macro-F1={test_macro_f1:.4f}  Micro-F1={test_micro_f1:.4f}")
print("Done. Results saved to", CFG["output_dir"])