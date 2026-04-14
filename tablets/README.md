# Табличная нейросеть для анализа данных

Универсальная библиотека для работы с табличными данными с использованием глубокого обучения. Поддерживает различные типы данных и три режима обучения.

## Возможности

- Обработка категориальных, численных, текстовых данных и массивов
- Три режима обучения: supervised, reinforcement, self-supervised
- Поддержка одно- и многовыходных моделей
- Встроенная предобработка данных
- Гибкая архитектура с настраиваемыми слоями

## Типы данных

### Категориальные (categorical)
Строковые или целочисленные категории (пол, тип продукта, статус)

### Численные (numerical)
Числовые значения (возраст, цена, температура)

### Текстовые (text)
Строки произвольной длины (описания, комментарии, названия)

### Массивы (arrays)
Списки чисел (временные ряды, координаты, векторы признаков)

## Установка

Требуется Python 3.7+ и следующие библиотеки:
```
torch
pandas
numpy
scikit-learn
```
## Быстрый старт

### 1. Supervised Learning (Классификация)
```
import pandas as pd
from torch.utils.data import DataLoader

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F'],
    'age': [25, 30, 35, 28],
    'description': ['text1', 'text2', 'text3', 'text4'],
    'features': [[1,2,3], [4,5,6], [7,8,9], [1,1,1]]
})
labels = pd.Series([0, 1, 0, 1])

dataset = TabularDataset(
    data=df,
    targets=labels,
    categorical_cols=['gender'],
    numerical_cols=['age'],
    text_cols=['description'],
    array_cols=['features']
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

config = {
    'categorical': {'vocab_sizes': [3], 'embed_dim': 8},
    'numerical': {'input_dim': 1},
    'text': {'vocab_size': 100, 'embed_dim': 16, 'hidden_dim': 32, 'num_cols': 1},
    'arrays': {'hidden_dim': 16, 'num_cols': 1},
    'hidden_dims': [64, 32]
}

model = SupervisedTabularNet(config, output_dim=2)
trainer = SupervisedTrainer(model, device='cpu')

for epoch in range(10):
    loss = trainer.train_epoch(dataloader, is_classification=True)
    print(f'Epoch {epoch}: {loss}')
```
### 2. Регрессия (одиночный выход)
```
targets = pd.Series([1.5, 2.3, 3.1, 2.8])
dataset = TabularDataset(df, targets, numerical_cols=['age'])
dataloader = DataLoader(dataset, batch_size=2)

model = SupervisedTabularNet(config, output_dim=1)
trainer = SupervisedTrainer(model)

for epoch in range(10):
    loss = trainer.train_epoch(dataloader, is_classification=False)
```
### 3. Множественные выходы
```
targets = pd.DataFrame({
    'price': [100, 200, 150, 180],
    'rating': [4.5, 3.8, 4.2, 4.0]
})

dataset = TabularDataset(df, targets, numerical_cols=['age'])
model = SupervisedTabularNet(config, output_dim=2, multi_output=True)
trainer = SupervisedTrainer(model)

for epoch in range(10):
    loss = trainer.train_epoch(dataloader, is_classification=False)
```
### 4. Reinforcement Learning

Для задач принятия решений на основе состояния окружения
```
state = {
    'category': torch.tensor([[0]]),
    'numerical': torch.tensor([[1.5]]),
    'text': torch.tensor([[[1,2,3,0,0]]]),
    'arrays': torch.tensor([[[1,2,3,0,0]]])
}

rl_model = ReinforcementTabularNet(config, action_dim=4, state_value=True)
rl_trainer = ReinforcementTrainer(rl_model, gamma=0.99)

action, log_prob, value = rl_model.get_action(state, deterministic=False)

Сбор траектории и обучение:

states = []
actions = []
rewards = []
dones = []

for step in range(100):
    action, log_prob, value = rl_model.get_action(current_state)
    next_state, reward, done = env.step(action)
    
    states.append(current_state)
    actions.append(action.item())
    rewards.append(reward)
    dones.append(done)
    
    current_state = next_state
    
    if done:
        break

metrics = rl_trainer.train_step(states, actions, rewards, dones)
```
### 5. Self-Supervised Learning

Для обучения без меток (feature extraction, предобучение)
```
dataset = TabularDataset(df, categorical_cols=['gender'], numerical_cols=['age'])
dataloader = DataLoader(dataset, batch_size=4)

ssl_model = SelfSupervisedTabularNet(config, latent_dim=32)
ssl_trainer = SelfSupervisedTrainer(ssl_model)
```
Обучение VAE:
```
for epoch in range(50):
    loss = ssl_trainer.train_epoch(dataloader, mode='vae', beta=1.0)
```
Обучение с контрастными потерями:
```
for epoch in range(50):
    loss = ssl_trainer.train_epoch(dataloader, mode='contrastive', temperature=0.5)
```
Извлечение эмбеддингов:
```
embeddings = ssl_model.get_embedding(batch_data)
```
## Конфигурация модели
```
config = {
    'categorical': {
        'vocab_sizes': [10, 5, 3],  # размеры словарей для каждой категориальной колонки
        'embed_dim': 8              # размерность эмбеддингов
    },
    'numerical': {
        'input_dim': 5              # количество численных признаков
    },
    'text': {
        'vocab_size': 1000,         # размер словаря символов/токенов
        'embed_dim': 64,            # размерность эмбеддингов
        'hidden_dim': 128,          # размер скрытого слоя LSTM
        'num_cols': 2               # количество текстовых колонок
    },
    'arrays': {
        'hidden_dim': 64,           # размер скрытого слоя LSTM
        'num_cols': 3               # количество колонок с массивами
    },
    'hidden_dims': [256, 128, 64],  # размеры скрытых слоев
    'encoder_dims': [128, 64],      # для self-supervised
    'decoder_dims': [64, 128]       # для self-supervised
}
```
## Инференс
```
model.eval()
with torch.no_grad():
    prediction = model(sample_data)
    
    # Для классификации
    predicted_class = prediction.argmax(dim=1)
    probabilities = torch.softmax(prediction, dim=1)
    
    # Для регрессии
    predicted_value = prediction
```
## Параметры обучения

### SupervisedTrainer
```
trainer = SupervisedTrainer(
    model=model,
    device='cuda'  # или 'cpu'
)
```
### ReinforcementTrainer
```
trainer = ReinforcementTrainer(
    model=model,
    device='cuda',
    gamma=0.99,           # коэффициент дисконтирования
    gae_lambda=0.95       # параметр GAE
)
```
train_step параметры:
- clip_ratio: клиппинг для PPO (по умолчанию 0.2)
- entropy_coef: вес энтропийного бонуса (по умолчанию 0.01)

### SelfSupervisedTrainer
```
trainer = SelfSupervisedTrainer(
    model=model,
    device='cuda'
)
```
train_epoch параметры:
- mode: 'vae' или 'contrastive'
- beta: вес KL-дивергенции для VAE (по умолчанию 1.0)
- temperature: температура для contrastive loss (по умолчанию 0.5)

## Примеры использования

### Предсказание оттока клиентов
```
df = pd.read_csv('customers.csv')
labels = df['churned']
df = df.drop('churned', axis=1)

dataset = TabularDataset(
    data=df,
    targets=labels,
    categorical_cols=['subscription_type', 'country'],
    numerical_cols=['age', 'balance', 'tenure'],
    text_cols=['last_complaint']
)

model = SupervisedTabularNet(config, output_dim=2)
trainer = SupervisedTrainer(model)
```
### Рекомендательная система с RL
```
state = get_user_state(user_id)
action, _, _ = rl_model.get_action(state, deterministic=True)
recommended_item = action_to_item(action)
```
### Кластеризация пользователей
```
ssl_model = SelfSupervisedTabularNet(config, latent_dim=16)
ssl_trainer = SelfSupervisedTrainer(ssl_model)
ssl_trainer.train_epoch(dataloader, mode='vae')

embeddings = []
for batch in dataloader:
    emb = ssl_model.get_embedding(batch)
    embeddings.append(emb)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(torch.cat(embeddings).numpy())
```
## Расширение функциональности

### Добавление нового типа данных

Модифицируйте TabularEncoder:
```
class CustomTabularEncoder(TabularEncoder):
    def __init__(self, config):
        super().__init__(config)
        if 'custom' in config:
            self.custom_encoder = nn.Linear(config['custom']['dim'], 64)
    
    def forward(self, x):
        encoded = super().forward(x)
        if 'custom' in x:
            custom_encoded = self.custom_encoder(x['custom'])
            encoded = torch.cat([encoded, custom_encoded], dim=1)
        return encoded
```
### Кастомная функция потерь
```
class CustomTrainer(SupervisedTrainer):
    def train_step(self, batch, is_classification=True):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        
        # Ваша кастомная функция потерь
        loss = custom_loss_function(output, y)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
```
## Оптимизация производительности

- Используйте batch_size кратный степени 2 (16, 32, 64)
- Для больших датасетов используйте num_workers в DataLoader
- Включите gradient accumulation для экономии памяти
- Используйте mixed precision training с torch.cuda.amp

dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

## Лицензия

MIT

## Поддержка

Для вопросов и предложений создавайте issue в репозитории