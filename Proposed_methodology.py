import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from PIL import Image
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import math
import json
from torchvision import transforms
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.hidden_dim = 256
        self.num_heads = 4
        self.ff_dim = 1024
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.dropout = 0.1
        self.num_classes = 1
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.mask_prob = 0.3
        self.beta = 0.1
        self.num_diffusion_steps = 100
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.lambda_v2t = 0.1
        self.lambda_t2t = 0.1
        self.data_path = "data/blip2_fine_tuned.csv"
        self.image_dir = 'data/combined_adr_images'
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        
    def _validate_config(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert 0 <= self.mask_prob <= 1, "mask_prob must be between 0 and 1"
        assert self.num_encoder_layers > 0, "num_encoder_layers must be positive"
        assert self.num_decoder_layers > 0, "num_decoder_layers must be positive"

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.out_proj(out)
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x)
        x = x + self.feed_forward(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, self_mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory, cross_mask)
        x = residual + self.dropout(x)
        
        x = x + self.feed_forward(x)
        return x

class ScoreNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.hidden_dim * 2
        self.hidden_dim = config.hidden_dim
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim + 1, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(3)
        ])
        
        self.cross_attention = MultiHeadAttention(config)
        self.output_proj = nn.Linear(self.hidden_dim, config.hidden_dim)

    def forward(self, x, t, condition):
        t = t.unsqueeze(-1)
        h = torch.cat([x, condition, t], dim=-1)
        
        for layer in self.layers:
            h_prev = h
            h = layer(h)
            h = h + h_prev
            
            h = self.cross_attention(
                q=h.unsqueeze(1),
                k=condition.unsqueeze(1),
                v=condition.unsqueeze(1)
            ).squeeze(1)
        
        return self.output_proj(h)

class StochasticDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_steps = config.num_diffusion_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x, t):
        alpha_bar = self.alpha_bars[t]
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar)[:, None] * x + torch.sqrt(1 - alpha_bar)[:, None] * noise
        return noisy_x, noise

    def reverse_diffusion(self, x, score_fn, condition=None):
        x_t = torch.randn_like(x)
        for t in reversed(range(self.num_steps)):
            time_tensor = torch.ones(x.shape[0], device=x.device) * t
            score = score_fn(x_t, time_tensor, condition)
            x_t = self.reverse_diffusion_step(x_t, score, t)
        return x_t

    def reverse_diffusion_step(self, x, score, t):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / (torch.sqrt(1 - alpha_bar_t))) * score)
        var = beta_t * noise
        
        return mean + var

class FeatureAlignmentModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.v2t_decoder = TransformerDecoderLayer(config)
        self.t2t_decoder = TransformerDecoderLayer(config)
        self.config = config

    def forward(self, text_features, visual_features, blip_features):
        v2t_aligned = self.v2t_decoder(text_features, visual_features)
        t2t_aligned = self.t2t_decoder(text_features, blip_features)
        return v2t_aligned, t2t_aligned

    def compute_loss(self, v2t_aligned, t2t_aligned, text_features):
        v2t_loss = F.mse_loss(v2t_aligned, text_features)
        t2t_loss = F.mse_loss(t2t_aligned, text_features)
        return self.config.lambda_v2t * v2t_loss, self.config.lambda_t2t * t2t_loss

class FeatureRecoveryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta_min = config.beta_start
        self.beta_max = config.beta_end
        self.num_steps = config.num_diffusion_steps
        self.hidden_dim = config.hidden_dim
        self.score_network = ScoreNetwork(config)

    def time_weight(self, t):
        """Calculate time-dependent weighting factor pλ(t)"""
        
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return 1.0 / beta_t

    def drift_coefficient(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return -0.5 * beta_t * x

    def diffusion_coefficient(self, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return torch.sqrt(beta_t)

    def score_matching_loss(self, x_k, x_obs, t, noise_scale=1.0):
        """
        Compute conditional score matching loss:
        Lscore = Eξ[|pλ(t)sk(xk(t), xIobs(t), t; θk) + z|²]
        
        Args:
            x_k: Features being recovered
            x_obs: Observed/conditioning features
            t: Time step
            noise_scale: Scale of noise distribution
        """
        batch_size = x_k.shape[0]
        
        
        z = torch.randn_like(x_k) * noise_scale
        
        
        p_lambda = self.time_weight(t)
        
        
        score = self.score_network(x_k, t, x_obs)
        
        
        weighted_score = p_lambda.unsqueeze(-1) * score
        
        
        loss = torch.pow(weighted_score + z, 2).mean()
        
        return loss

    def forward_sde(self, x, dt=1e-3):
        batch_size = x.shape[0]
        t = torch.zeros(batch_size, device=x.device)
        x_t = x.clone()
        
        for step in range(self.num_steps):
            t = t + dt
            dw = torch.randn_like(x) * math.sqrt(dt)
            dx = self.drift_coefficient(x_t, t) * dt + \
                 self.diffusion_coefficient(t).unsqueeze(-1) * dw
            x_t = x_t + dx
            
        return x_t

    def reverse_sde(self, x, condition, dt=1e-3):
        batch_size = x.shape[0]
        t = torch.ones(batch_size, device=x.device)
        x_t = x.clone()
        
        total_score_loss = 0
        
        for step in range(self.num_steps):
            
            score = self.score_network(x_t, t, condition)
            score_loss = self.score_matching_loss(x_t, condition, t)
            total_score_loss += score_loss
            
            drift = self.drift_coefficient(x_t, t)
            diffusion = self.diffusion_coefficient(t)
            
            dw = torch.randn_like(x) * math.sqrt(dt) if step < self.num_steps - 1 else 0
            
            dx = (drift - diffusion.unsqueeze(-1)**2 * score) * dt + \
                 diffusion.unsqueeze(-1) * dw
            
            x_t = x_t + dx
            t = t - dt
            
        
        return x_t, total_score_loss / self.num_steps

class ModAlignADE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.text_encoder.gradient_checkpointing_enable()
        self.text_proj = nn.Linear(768, config.hidden_dim)
        
        
        self.visual_encoder = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.visual_proj = nn.Linear(768, config.hidden_dim)
        
        
        self.text_transformer = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        self.visual_transformer = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        
        
        self.alignment_module = FeatureAlignmentModule(config)
        self.recovery_module = FeatureRecoveryModule(config)
        self.diffusion = StochasticDiffusion(config)
        
        
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        self.config = config

    def encode_text(self, text_inputs):
        outputs = self.text_encoder(**text_inputs)
        hidden_states = outputs.last_hidden_state
        x = self.text_proj(hidden_states)
        
        for layer in self.text_transformer:
            x = layer(x)
        return x

    def encode_visual(self, image_inputs):
        x = self.visual_encoder(image_inputs)
        x = x.unsqueeze(1).expand(-1, 197, -1)
        x = self.visual_proj(x)
        
        for layer in self.visual_transformer:
            x = layer(x)
        return x

    def recover_features(self, x, condition, is_text=True):
        x_cls = x[:, 0, :]
        condition_cls = condition[:, 0, :]
        
        recovered_cls, score_loss = self.recovery_module.reverse_sde(
            x=x_cls,
            condition=condition_cls
        )
        
        recovered = recovered_cls.unsqueeze(1).expand(-1, x.size(1), -1)
        return recovered, score_loss

    def forward(self, text_inputs, image_inputs, blip_inputs=None, modality_masks=None):
        
        text_features = self.encode_text(text_inputs)
        visual_features = self.encode_visual(image_inputs)
        
        if blip_inputs is not None:
            blip_features = self.encode_text(blip_inputs)
        else:
            blip_features = text_features.clone()
        
        # Feature alignment
        v2t_aligned, t2t_aligned = self.alignment_module(text_features, visual_features, blip_features)
        alignment_losses = self.alignment_module.compute_loss(v2t_aligned, t2t_aligned, text_features)
        
        # Initialize score matching loss
        total_score_loss = 0
        
        # Handle missing modalities
        if modality_masks is not None:
            if modality_masks['text'].any():
                text_features, text_score_loss = self.recover_features(text_features, visual_features, is_text=True)
                total_score_loss += text_score_loss
            if modality_masks['vision'].any():
                visual_features, visual_score_loss = self.recover_features(visual_features, text_features, is_text=False)
                total_score_loss += visual_score_loss
        
        # Feature fusion
        text_cls = text_features[:, 0, :]
        visual_cls = visual_features[:, 0, :]
        blip_cls = blip_features[:, 0, :]
        
        fused_features = torch.cat([text_cls, visual_cls, blip_cls], dim=-1)
        output = self.fusion(fused_features)
        
        return {
            'output': output,
            'v2t_loss': alignment_losses[0],
            't2t_loss': alignment_losses[1],
            'score_loss': total_score_loss,
            'text_features': text_features,
            'visual_features': visual_features,
            'blip_features': blip_features
        }

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_retries=3):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        for retry in range(self.max_retries):
            try:
                text = str(row['Text'])
                text_encoding = self.tokenizer(
                    text,
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                )
                
                image_path = os.path.join(self.image_dir, row['Image'])
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image)
                
                label = torch.tensor(row['label'], dtype=torch.float)
                
                return {
                    'text': {k: v.squeeze(0) for k, v in text_encoding.items()},
                    'image': image,
                    'label': label
                }
            except Exception as e:
                if retry == self.max_retries - 1:
                    logger.error(f"Failed to load data at index {idx} after {self.max_retries} attempts: {str(e)}")
                    raise
                continue

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.mask_prob = config.mask_prob
        self.best_val_loss = float('inf')

    def _generate_random_masks(self, batch_size):
        masks = {
            'text': torch.zeros(batch_size, dtype=torch.bool, device=self.config.device),
            'vision': torch.zeros(batch_size, dtype=torch.bool, device=self.config.device)
        }
        for i in range(batch_size):
            if torch.rand(1).item() < self.mask_prob:
                if torch.rand(1).item() < 0.5:
                    masks['text'][i] = True
                else:
                    masks['vision'][i] = True
        return masks

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            text_inputs = {k: v.to(self.config.device) for k, v in batch['text'].items()}
            images = batch['image'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            
            modality_masks = self._generate_random_masks(labels.size(0))
            
            outputs = self.model(text_inputs, images, modality_masks=modality_masks)
            
            task_loss = self.criterion(outputs['output'].squeeze(), labels)
            alignment_loss = outputs['v2t_loss'] + outputs['t2t_loss']
            score_loss = outputs['score_loss']
            
            # Combined loss with score matching
            loss = task_loss + self.config.beta * alignment_loss + self.lambda_score * score_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs['output']).detach().cpu().numpy()
            all_preds.extend(preds.ravel())
            all_labels.extend(labels.cpu().numpy().ravel())
        
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                text_inputs = {k: v.to(self.config.device) for k, v in batch['text'].items()}
                images = batch['image'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                modality_masks = self._generate_random_masks(labels.size(0))
                outputs = self.model(text_inputs, images, modality_masks=modality_masks)
                
                task_loss = self.criterion(outputs['output'].squeeze(), labels)
                alignment_loss = outputs['v2t_loss'] + outputs['t2t_loss']
                loss = task_loss + self.config.beta * alignment_loss
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs['output']).detach().cpu().numpy()
                all_preds.extend(preds.ravel())
                all_labels.extend(labels.cpu().numpy().ravel())
        
        metrics = self.compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics

    @staticmethod
    def compute_metrics(preds, labels):
        preds = (np.array(preds) > 0.5).astype(np.int64)
        labels = np.array(labels).astype(np.int64)
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds),
            'f1': f1_score(labels, preds)
        }

    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
        if metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with validation loss: {metrics['loss']:.4f}")

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = Config()
    config._validate_config()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Using device: {config.device}")
    logger.info(f"Using masking probability: {config.mask_prob}")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    try:
        dataset = MultimodalDataset(
            csv_path=config.data_path,
            image_dir=config.image_dir,
            tokenizer=tokenizer
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

        model = ModAlignADE(config)
        trainer = Trainer(model, train_loader, val_loader, config)

        best_val_loss = float('inf')
        train_start_time = datetime.now()

        for epoch in range(config.num_epochs):
            epoch_start = datetime.now()
            
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()
            
            trainer.save_checkpoint(epoch, val_metrics)
            
            epoch_time = datetime.now() - epoch_start

            logger.info(f"\nEpoch {epoch+1}/{config.num_epochs}")
            logger.info(f"Time: {epoch_time}")
            logger.info("Train Metrics:")
            for k, v in train_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            logger.info("Validation Metrics:")
            for k, v in val_metrics.items():
                logger.info(f"  {k}: {v:.4f}")

            logger.info("-" * 50)

        training_time = datetime.now() - train_start_time
        results = {
            'final_val_metrics': val_metrics,
            'training_time': str(training_time),
            'total_epochs': config.num_epochs,
            'best_val_loss': best_val_loss
        }

        with open(os.path.join(config.log_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"\nTraining completed! Total time: {training_time}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise e

    finally:
        logger.info("Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
