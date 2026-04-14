import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json


class TabularDataset(Dataset):
    def __init__(self, data: pd.DataFrame, targets: Union[pd.Series, pd.DataFrame] = None, 
                 categorical_cols: List[str] = None, numerical_cols: List[str] = None,
                 text_cols: List[str] = None, array_cols: List[str] = None):
        self.data = data
        self.targets = targets
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.text_cols = text_cols or []
        self.array_cols = array_cols or []
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.vocab = {}
        self.max_text_len = 100
        self.max_array_len = 50
        
        self._preprocess()
    
    def _preprocess(self):
        self.processed_data = {}
        
        if self.categorical_cols:
            cat_data = []
            for col in self.categorical_cols:
                le = LabelEncoder()
                encoded = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                cat_data.append(encoded)
            self.processed_data['categorical'] = np.stack(cat_data, axis=1)
        
        if self.numerical_cols:
            num_data = self.data[self.numerical_cols].values.astype(np.float32)
            self.processed_data['numerical'] = self.scaler.fit_transform(num_data)
        
        if self.text_cols:
            text_data = []
            for col in self.text_cols:
                tokenized = self._tokenize_text(self.data[col])
                text_data.append(tokenized)
            self.processed_data['text'] = np.stack(text_data, axis=1)
        
        if self.array_cols:
            array_data = []
            for col in self.array_cols:
                processed = self._process_arrays(self.data[col])
                array_data.append(processed)
            self.processed_data['arrays'] = np.stack(array_data, axis=1)
    
    def _tokenize_text(self, texts):
        if not hasattr(self, 'char_to_idx'):
            chars = set(''.join(texts.astype(str)))
            self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        
        tokenized = np.zeros((len(texts), self.max_text_len), dtype=np.int64)
        for i, text in enumerate(texts.astype(str)):
            for j, char in enumerate(text[:self.max_text_len]):
                tokenized[i, j] = self.char_to_idx.get(char, 0)
        return tokenized
    
    def _process_arrays(self, arrays):
        processed = np.zeros((len(arrays), self.max_array_len), dtype=np.float32)
        for i, arr in enumerate(arrays):
            if isinstance(arr, str):
                arr = json.loads(arr)
            arr = np.array(arr, dtype=np.float32)
            length = min(len(arr), self.max_array_len)
            processed[i, :length] = arr[:length]
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = {}
        for key, value in self.processed_data.items():
            item[key] = torch.tensor(value[idx])
        
        if self.targets is not None:
            target = torch.tensor(self.targets.iloc[idx] if isinstance(self.targets, pd.Series) 
                                 else self.targets.iloc[idx].values, dtype=torch.float32)
            return item, target
        return item


class TabularEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.cat_embeddings = nn.ModuleList()
        if 'categorical' in config:
            for vocab_size in config['categorical']['vocab_sizes']:
                self.cat_embeddings.append(nn.Embedding(vocab_size, config['categorical']['embed_dim']))
        
        if 'text' in config:
            self.text_embedding = nn.Embedding(config['text']['vocab_size'], config['text']['embed_dim'])
            self.text_encoder = nn.LSTM(config['text']['embed_dim'], config['text']['hidden_dim'], 
                                       batch_first=True, bidirectional=True)
        
        if 'arrays' in config:
            self.array_encoder = nn.LSTM(1, config['arrays']['hidden_dim'], batch_first=True)
        
        total_dim = 0
        if 'categorical' in config:
            total_dim += len(config['categorical']['vocab_sizes']) * config['categorical']['embed_dim']
        if 'numerical' in config:
            total_dim += config['numerical']['input_dim']
        if 'text' in config:
            total_dim += config['text']['hidden_dim'] * 2 * config['text']['num_cols']
        if 'arrays' in config:
            total_dim += config['arrays']['hidden_dim'] * config['arrays']['num_cols']
        
        self.total_dim = total_dim
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        
        if 'categorical' in x and self.cat_embeddings:
            for i, emb in enumerate(self.cat_embeddings):
                encoded.append(emb(x['categorical'][:, i]))
        
        if 'numerical' in x:
            encoded.append(x['numerical'])
        
        if 'text' in x and hasattr(self, 'text_embedding'):
            batch_size, num_cols, seq_len = x['text'].shape
            text_flat = x['text'].view(-1, seq_len)
            embedded = self.text_embedding(text_flat)
            _, (h, _) = self.text_encoder(embedded)
            h = h.transpose(0, 1).contiguous().view(batch_size, num_cols, -1)
            encoded.append(h.view(batch_size, -1))
        
        if 'arrays' in x and hasattr(self, 'array_encoder'):
            batch_size, num_cols, arr_len = x['arrays'].shape
            arr_flat = x['arrays'].view(-1, arr_len, 1)
            _, (h, _) = self.array_encoder(arr_flat)
            h = h.squeeze(0).view(batch_size, num_cols, -1)
            encoded.append(h.view(batch_size, -1))
        
        return torch.cat(encoded, dim=1)


class SupervisedTabularNet(nn.Module):
    def __init__(self, config: Dict[str, Any], output_dim: int, multi_output: bool = False):
        super().__init__()
        self.encoder = TabularEncoder(config)
        self.multi_output = multi_output
        
        hidden_dims = config.get('hidden_dims', [256, 128, 64])
        layers = []
        input_dim = self.encoder.total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encoder(x)
        features = self.backbone(encoded)
        output = self.output(features)
        return output


class ReinforcementTabularNet(nn.Module):
    def __init__(self, config: Dict[str, Any], action_dim: int, state_value: bool = True):
        super().__init__()
        self.encoder = TabularEncoder(config)
        self.state_value = state_value
        
        hidden_dims = config.get('hidden_dims', [256, 128])
        layers = []
        input_dim = self.encoder.total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.policy = nn.Linear(input_dim, action_dim)
        
        if state_value:
            self.value = nn.Linear(input_dim, 1)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        encoded = self.encoder(x)
        features = self.backbone(encoded)
        logits = self.policy(features)
        
        if self.state_value:
            value = self.value(features)
            return logits, value
        return logits
    
    def get_action(self, x: Dict[str, torch.Tensor], deterministic: bool = False):
        if self.state_value:
            logits, value = self.forward(x)
        else:
            logits = self.forward(x)
            value = None
        
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, log_prob, value


class SelfSupervisedTabularNet(nn.Module):
    def __init__(self, config: Dict[str, Any], latent_dim: int = 128):
        super().__init__()
        self.encoder = TabularEncoder(config)
        self.latent_dim = latent_dim
        
        encoder_dims = config.get('encoder_dims', [256, 128])
        layers = []
        input_dim = self.encoder.total_dim
        
        for hidden_dim in encoder_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*layers)
        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)
        
        decoder_dims = config.get('decoder_dims', [128, 256])
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.decoder_net = nn.Sequential(*layers)
        self.decoder_output = nn.Linear(input_dim, self.encoder.total_dim)
        
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def encode(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        h = self.encoder_net(encoded)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_net(z)
        return self.decoder_output(h)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_embedding(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu
    
    def contrastive_loss(self, x1: Dict[str, torch.Tensor], x2: Dict[str, torch.Tensor], 
                        temperature: float = 0.5) -> torch.Tensor:
        z1 = self.projection_head(self.get_embedding(x1))
        z2 = self.projection_head(self.get_embedding(x2))
        
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / temperature
        
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        pos_sim = torch.cat([sim_matrix[i, i + batch_size].unsqueeze(0) 
                            for i in range(batch_size)] + 
                           [sim_matrix[i + batch_size, i].unsqueeze(0) 
                            for i in range(batch_size)])
        
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        return loss.mean()


class SupervisedTrainer:
    def __init__(self, model: SupervisedTabularNet, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def train_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], 
                   is_classification: bool = True) -> float:
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        
        if is_classification:
            if y.dim() == 1:
                loss = F.cross_entropy(output, y.long())
            else:
                loss = F.binary_cross_entropy_with_logits(output, y)
        else:
            loss = F.mse_loss(output, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, is_classification: bool = True) -> float:
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch, is_classification)
            total_loss += loss
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader, is_classification: bool = True) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = {k: v.to(self.device) for k, v in x.items()}
                y = y.to(self.device)
                output = self.model(x)
                
                if is_classification:
                    if y.dim() == 1:
                        loss = F.cross_entropy(output, y.long())
                    else:
                        loss = F.binary_cross_entropy_with_logits(output, y)
                else:
                    loss = F.mse_loss(output, y)
                
                total_loss += loss.item()
        return total_loss / len(dataloader)


class ReinforcementTrainer:
    def __init__(self, model: ReinforcementTabularNet, device: str = 'cuda', 
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], 
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, states: List[Dict[str, torch.Tensor]], actions: List[int], 
                   rewards: List[float], dones: List[bool], 
                   clip_ratio: float = 0.2, entropy_coef: float = 0.01) -> Dict[str, float]:
        
        all_logits = []
        all_values = []
        
        for state in states:
            state = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            logits, value = self.model(state)
            all_logits.append(logits.squeeze(0))
            all_values.append(value.squeeze())
        
        advantages, returns = self.compute_gae(rewards, all_values, dones)
        
        old_log_probs = []
        for logits, action in zip(all_logits, actions):
            log_prob = F.log_softmax(logits, dim=-1)[action]
            old_log_probs.append(log_prob.detach())
        old_log_probs = torch.stack(old_log_probs)
        
        self.optimizer.zero_grad()
        
        new_logits = []
        new_values = []
        for state in states:
            state = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            logits, value = self.model(state)
            new_logits.append(logits.squeeze(0))
            new_values.append(value.squeeze())
        
        new_log_probs = []
        entropy = []
        for logits, action in zip(new_logits, actions):
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)[action]
            new_log_probs.append(log_prob)
            entropy.append(-(probs * log_prob).sum())
        
        new_log_probs = torch.stack(new_log_probs)
        entropy = torch.stack(entropy).mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        new_values = torch.stack(new_values)
        value_loss = F.mse_loss(new_values, returns)
        
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


class SelfSupervisedTrainer:
    def __init__(self, model: SelfSupervisedTabularNet, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
    
    def vae_loss(self, recon: torch.Tensor, target: torch.Tensor, 
                 mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, target, reduction='sum') / target.size(0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
        return recon_loss + beta * kld
    
    def train_step_vae(self, batch: Dict[str, torch.Tensor], beta: float = 1.0) -> float:
        x = {k: v.to(self.device) for k, v in batch.items()}
        
        target = self.model.encoder(x).detach()
        
        self.optimizer.zero_grad()
        recon, mu, logvar = self.model(x)
        loss = self.vae_loss(recon, target, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_contrastive(self, batch1: Dict[str, torch.Tensor], 
                              batch2: Dict[str, torch.Tensor], temperature: float = 0.5) -> float:
        x1 = {k: v.to(self.device) for k, v in batch1.items()}
        x2 = {k: v.to(self.device) for k, v in batch2.items()}
        
        self.optimizer.zero_grad()
        loss = self.model.contrastive_loss(x1, x2, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader, mode: str = 'vae', 
                   beta: float = 1.0, temperature: float = 0.5) -> float:
        self.model.train()
        total_loss = 0
        
        if mode == 'vae':
            for batch in dataloader:
                if isinstance(batch, tuple):
                    batch = batch[0]
                loss = self.train_step_vae(batch, beta)
                total_loss += loss
        elif mode == 'contrastive':
            for batch in dataloader:
                if isinstance(batch, tuple):
                    batch = batch[0]
                loss = self.train_step_contrastive(batch, batch, temperature)
                total_loss += loss
        
        self.scheduler.step()
        return total_loss / len(dataloader)