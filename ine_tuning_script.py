# ========================= IMPORTS =========================
import os, json, torch, h5py, shutil
from torch.utils.data import Dataset, DataLoader
from torch import optim
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm

# ========================= DEVICE =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ========================= PATHS =========================
DATA_DIR = "/kaggle/input/mini-coco2014-dataset-for-image-captioning"
ANNOTATIONS_PATH = os.path.join(DATA_DIR, "captions.json")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
OUTPUT_PATH = "/kaggle/working/fine_tuned_caption_model.h5"

# ========================= HYPERPARAMS =========================
BATCH_SIZE = 8
MAX_LEN = 40
EPOCHS = 8
LR = 5e-6

# ========================= MODEL =========================
MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

# Freeze encoder
for p in model.encoder.parameters():
    p.requires_grad = False
print("🧠 Encoder frozen — only GPT-2 decoder is being fine-tuned.")

# ========================= DATASET =========================
class CocoDataset(Dataset):
    def __init__(self, json_file, img_dir, fe, tok, max_len):
        with open(json_file, "r") as f:
            data = json.load(f)
        self.samples = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
        self.img_dir, self.fe, self.tok, self.max_len = img_dir, fe, tok, max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        c = self.samples[idx]
        img_id = c.get("image_id", c.get("file_name"))
        caption = c.get("caption", c.get("text"))
        img_name = f"COCO_train2014_{img_id:012d}.jpg" if isinstance(img_id, int) else img_id
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.fe(images=img, return_tensors="pt").pixel_values.squeeze()
        labels = self.tok(caption, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze()
        return {"pixel_values": pixel_values, "labels": labels}

train_ds = CocoDataset(ANNOTATIONS_PATH, IMAGES_DIR, feature_extractor, tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ========================= TRAIN =========================
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

print("\n🚀 Starting training...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    optimizer.zero_grad()

    for step, batch in enumerate(loop):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} done | Avg loss: {total_loss/len(train_loader):.4f}")

print("\n🎯 Training complete!")

# ========================= SAVE MODEL AS .H5 =========================
print("💾 Saving model to HDF5 (.h5) file...")
with h5py.File(OUTPUT_PATH, "w") as f:
    weights_bytes = torch.save(model.state_dict(), "/kaggle/working/tmp_model.pt")
    with open("/kaggle/working/tmp_model.pt", "rb") as w:
        f.create_dataset("model_weights", data=w.read())
os.remove("/kaggle/working/tmp_model.pt")

print(f"✅ Model saved as: {OUTPUT_PATH}")
