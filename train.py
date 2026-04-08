from transformers import DistilBertModel, CLIPImageProcessor, CLIPModel, DistilBertTokenizer
from torch import nn
from typing import TypedDict
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import gc
import pickle
from lightning.pytorch.loggers import WandbLogger
from flash_attn import flash_attn_func
torch.set_float32_matmul_precision('medium')

BATCH_SIZE = 64

class JudgedRecord(TypedDict):
    image: Image.Image
    question: str
    lang: str
    answer: str | list[str]
    model_name: str
    model_answer: str
    source_dataset: str
    source_id: str
    record_id: str
    correct: bool

class MMMuxConfig(TypedDict):
    vision_dim: int
    text_dim: int
    hidden_dim: int
    mixer_nhead: int
    experts: int
    top_k: int
    transformer_blocks: int

bert_model = "./distilbert-base-multilingual-cased"
clip_model = "./clip-vit-base-patch32"

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_proj_key = nn.Linear(config["vision_dim"], config["text_dim"])
        self.vision_proj_value = nn.Linear(config["vision_dim"], config["text_dim"])
        
        self.ffn = nn.Sequential(
            nn.Linear(config["text_dim"], config["hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], config["text_dim"]),
        )
        
        
        self.norm1 = nn.LayerNorm(config["text_dim"])
        self.norm2 = nn.LayerNorm(config["text_dim"])
        
        
        self.num_heads = config["mixer_nhead"]
        self.head_dim = config["text_dim"] // self.num_heads
        assert config["text_dim"] % self.num_heads == 0, "text_dim must be divisible by num_heads"

    def forward(self, text, vision):
        
        key = self.vision_proj_key(vision)
        value = self.vision_proj_value(vision)
        
        
        batch_size, seq_len_text, _ = text.shape
        q = text.view(batch_size, seq_len_text, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        
        attn_output = flash_attn_func(
            q, k, v,
            causal=False,  
            softmax_scale=None  
        )
        
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_text, -1)
        text = text + attn_output
        text = self.norm1(text)
        
        
        ffn_output = self.ffn(text)
        text = text + ffn_output
        text = self.norm2(text)
        return text

class Mixer(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config["transformer_blocks"] - idx)
        ])

    def forward(self, text, vision):
        
        for block in self.blocks:
            text = block(text, vision)
        return text

class MoEMixer(nn.Module):
    def __init__(self, config: MMMuxConfig):
        super().__init__()
        self.config = config
        self.num_experts = config["experts"]  
        self.top_k = config["top_k"]  
        
        
        self.experts = nn.ModuleList([Mixer(config, i) for i in range(self.num_experts)])
        
        
        
        self.gate = nn.Sequential(
            nn.Linear(config["text_dim"] + config["vision_dim"], config["hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config["hidden_dim"], self.num_experts)
        )

    def forward(self, text, vision):
        batch_size, _, _ = text.shape  
        
        
        combined_input = torch.cat((text[:, 0, :], vision[:, 0, :]), dim=-1)
        gate_logits = self.gate(combined_input)
        gate_logits = torch.clamp(gate_logits, -6, 6)
        gate_weights = F.softmax(gate_logits, dim=-1)  
        if self.training:
            
            
            
            
            gate_weights = torch.nn.functional.dropout(gate_weights, p=0.12)
        
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        
        
        output = torch.zeros_like(text)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  
            expert_weight = top_k_weights[:, i].unsqueeze(-1).unsqueeze(-1)  
            
            
            for b in range(batch_size):
                expert_out = self.experts[expert_idx[b]](text[b:b+1], vision[b:b+1])
                output[b:b+1] += expert_weight[b:b+1] * expert_out / self.top_k
        
        return output
    

class MMMux(L.LightningModule):
    def __init__(self, config: MMMuxConfig):
        super().__init__()
        self.config = config
        
        self.bert = DistilBertModel.from_pretrained(bert_model)
        self.clip = CLIPModel.from_pretrained(clip_model)
        
        self.bert.train()
        self.clip.train()
        
        
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model)
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model)
        
        
        self.bypass = nn.Linear(config["vision_dim"], config["text_dim"])
        self.mixer = MoEMixer(self.config)
        self.mixer_ln = nn.LayerNorm(config["text_dim"])

    def forward(self, image, text, name):
        device = next(self.parameters()).device

        
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}  
        clip_feats = self.clip.get_image_features(pixel_values=clip_inputs["pixel_values"])
        clip_feats_unsq = clip_feats.unsqueeze(1)

        
        text_inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}  
        text_outputs = self.bert(**text_inputs)
        text_embeds = text_outputs.last_hidden_state
        text_cls = text_embeds[:, 0, :]
        text_cls_unsq = text_cls.unsqueeze(1)

        
        q_a = self.mixer(text_cls_unsq, clip_feats_unsq).squeeze(1)
        q = self.mixer_ln(text_cls + q_a  + self.bypass(clip_feats))

        
        name_inputs = self.bert_tokenizer(name, return_tensors="pt", padding=True, truncation=True)
        name_inputs = {k: v.to(device) for k, v in name_inputs.items()}
        name_outputs = self.bert(**name_inputs)
        name_cls = name_outputs.last_hidden_state[:, 0, :]
        eps = 1e-6
        q_norm = q / (q.norm(dim=-1, keepdim=True) + eps)
        name_cls_norm = name_cls / (name_cls.norm(dim=-1, keepdim=True) + eps)
        cos_sim = F.cosine_similarity(q_norm, name_cls_norm, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)
        logit = torch.arctanh(cos_sim)
        return torch.clamp(logit, -6, 6)
    
    def encode(self, image, text, name):
        device = next(self.parameters()).device

        
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}  
        clip_feats = self.clip.get_image_features(pixel_values=clip_inputs["pixel_values"])
        clip_feats_unsq = clip_feats.unsqueeze(1)

        
        text_inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}  
        text_outputs = self.bert(**text_inputs)
        text_embeds = text_outputs.last_hidden_state
        text_cls = text_embeds[:, 0, :]
        text_cls_unsq = text_cls.unsqueeze(1)

        
        q_a = self.mixer(text_cls_unsq, clip_feats_unsq).squeeze(1)
        q = self.mixer_ln(text_cls + q_a  + self.bypass(clip_feats))

        
        name_inputs = self.bert_tokenizer(name, return_tensors="pt", padding=True, truncation=True)
        name_inputs = {k: v.to(device) for k, v in name_inputs.items()}
        name_outputs = self.bert(**name_inputs)
        name_cls = name_outputs.last_hidden_state[:, 0, :]
        eps = 1e-6
        q_norm = q / (q.norm(dim=-1, keepdim=True) + eps)
        name_cls_norm = name_cls / (name_cls.norm(dim=-1, keepdim=True) + eps)
        cos_sim = F.cosine_similarity(q_norm, name_cls_norm, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1 + eps, 1 - eps)
        logit = torch.arctanh(cos_sim)
        return name_cls, q

    def training_step(self, batch, batch_idx):
        images = [record["image"] for record in batch]
        questions = [record["question"] for record in batch]
        model_names = [record["model_name"] for record in batch]

        pred_logits = self.forward(images, questions, model_names)
        for im in images:
            im.close()
        target = torch.tensor(
            [1.0 if record["correct"] else 0.0 for record in batch],
            device=pred_logits.device,
        )
        loss = F.binary_cross_entropy_with_logits(pred_logits, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("train_acc", (pred_logits > 0.0).eq(target).float().mean(), on_step=True, on_epoch=True, batch_size=BATCH_SIZE)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        if loss.isnan():
            print("nan loss")
            
            return torch.tensor(0.0, requires_grad=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = [record["image"] for record in batch]
        questions = [record["question"] for record in batch]
        model_names = [record["model_name"] for record in batch]

        pred_logits = self.forward(images, questions, model_names)
        for im in images:
            im.close()
        target = torch.tensor(
            [1.0 if record["correct"] else 0.0 for record in batch],
            device=pred_logits.device,
        )
        loss = F.binary_cross_entropy_with_logits(pred_logits, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=BATCH_SIZE)
        self.log("val_acc", (pred_logits > 0.0).eq(target).float().mean(), prog_bar=True, on_epoch=True, batch_size=BATCH_SIZE)
        return loss

    def test_step(self, batch, batch_idx):
        images = [record["image"] for record in batch]
        questions = [record["question"] for record in batch]
        model_names = [record["model_name"] for record in batch]
        for im in images:
            im.close()
        pred_logits = self.forward(images, questions, model_names)
        target = torch.tensor(
            [1.0 if record["correct"] else 0.0 for record in batch],
            device=pred_logits.device,
        )
        loss = F.binary_cross_entropy_with_logits(pred_logits, target)
        self.log("test_loss", loss, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("test_acc", (pred_logits > 0.0).eq(target).float().mean(), prog_bar=True, batch_size=BATCH_SIZE)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=8e-5, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.96, patience=50, min_lr=5e-6)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_acc_step",
                "interval": "step",
            }
        }


import random
from collections import deque
class JudgedRecordIterableDataset(IterableDataset):
    def __init__(self, file_names: list[str], skip_n=2001, total=20000, buffer_size=BATCH_SIZE // 16, modulus=10):
        self.file_names = file_names
        self.skip_n = skip_n
        self.total = total
        self.buffer_size = buffer_size  
        self.modulus = 10

    def __iter__(self):
        buffer = deque(maxlen=self.buffer_size)
        iterator = iterator_from_pickle(self.file_names, self.skip_n, self.total)
        cnt = 0
        valid_moduls = range(20)
        for record in iterator:
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)  
                v = buffer.popleft()
                cnt += 1
                cnt %= self.modulus
                if cnt % self.modulus in valid_moduls:
                    gc.collect()
                    yield v  
                del v
            buffer.append(record)

        
        random.shuffle(buffer)
        while buffer:
            yield buffer.popleft()

from tqdm import trange
from torch.utils.data import get_worker_info
from torch.utils.data import get_worker_info
import pickle
import io
import base64
from tqdm import trange

import itertools
from torch.utils.data import get_worker_info
import pickle
import io
import base64
from tqdm import trange
import os
import pickle
import itertools
import io
import base64
from PIL import Image
from tqdm import trange
from torch.utils.data import get_worker_info
from flash_attn import flash_attn_qkvpacked_func

CACHE_DIR = "./skip_cache"  
import gc
def iterator_from_pickle(file_names: list[str], skip_n=2001, total=20000):
    from torch.utils.data import get_worker_info
    import os, pickle, itertools, base64, io
    from PIL import Image
    from tqdm import trange

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    num_workers = worker_info.num_workers if worker_info else 1

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"worker_3_skip{skip_n}.pkl")

    seek_positions = {}
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            seek_positions = pickle.load(f)

    files = []
    file_iterators = []
    for file_name in file_names:
        f = open(file_name, "rb")
        if (file_name, skip_n) in seek_positions:
            f.seek(seek_positions[(file_name, skip_n)])
        else:
            for _ in trange(skip_n, desc=f"Worker-{worker_id} Skipping {file_name}"):
                try:
                    lded = pickle.load(f)
                    if not isinstance(lded['image'], str):
                        lded['image'].close()
                    del lded
                    gc.collect()
                except EOFError:
                    break
            seek_positions[(file_name, skip_n)] = f.tell()
        files.append(f)
        file_iterators.append((file_name, f))

    if not os.path.exists(cache_file): 
        with open(cache_file, "wb") as f:
            pickle.dump(seek_positions, f)

    import json
    blacklist = frozenset(json.load(open("./blacklist.json", "r")))

    count = 0
    active_iters = list(file_iterators)

    while active_iters and count < total:
        for idx in range(len(active_iters) - 1, -1, -1):  
            file_name, f = active_iters[idx]
            try:
                res = pickle.load(f)
                if res['record_id'] in blacklist:
                    if not isinstance(res['image'], str):
                        res['image'].close()
                    continue
                if isinstance(res['image'], str):
                    res['image'] = Image.open(io.BytesIO(base64.b64decode(res['image'])))
                if res['image'].size[0] > 1024 or res['image'].size[1] > 1024:
                    max_dim = max(res['image'].size)
                    scale = 1024 / max_dim
                    res['image'] = res['image'].resize(
                        (int(res['image'].size[0] * scale), int(res['image'].size[1] * scale))
                    )
                if count % num_workers == worker_id:
                    yield res
                count += 1
                if count >= total:
                    break
            except EOFError:
                f.close()
                active_iters.pop(idx)  
            except Exception as e:
                print(f"Error reading from {file_name}: {e}")
                continue

    for f in files:
        f.close()
    gc.collect()


if __name__ == "__main__":
    
    config: MMMuxConfig = {
        "vision_dim": 512,    
        "text_dim": 768,      
        "hidden_dim": 1024,   
        "mixer_nhead": 8,
        "experts": 4,
        "transformer_blocks": 4,
        "top_k": 1,
    }

    
    model = MMMux(config)
    
    
    import os
    pkl_data_files = [
        "./close_alm_train.pkl", "./close_no_alm_train.pkl",
    ]
    
    train_dataset = JudgedRecordIterableDataset(pkl_data_files, 0, 9070400)
    
    

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x, num_workers=24, prefetch_factor=1)
    
    
    prefix = "close_all"
    
    logger = WandbLogger(project="mmm_final_final", name=prefix)
    logger.log_hyperparams(config)
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="auto",
        logger=logger,
        strategy='ddp_find_unused_parameters_true',
        
        precision="bf16-true",
        log_every_n_steps=5,
        devices=3,
        gradient_clip_val=1,  
        gradient_clip_algorithm="norm",
    )
    
    trainer.fit(model, train_loader)

    
    

    trainer.save_checkpoint(f"{prefix}/mmm.ckpt")
    torch.save(model.state_dict(), f"{prefix}_mmm.pth")
    import json
    json.dump(model.config, open(f"{prefix}_mmm.json", "w"))
