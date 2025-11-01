# Panduan Penggunaan Model GNN

## 📖 Daftar Isi
1. [Quick Start](#quick-start)
2. [Step-by-Step Tutorial](#step-by-step-tutorial)
3. [Customization](#customization)
4. [Tips & Tricks](#tips--tricks)
5. [FAQ](#faq)

## Quick Start

### 1. Install Dependencies
```bash
# Via pip
pip install -r requirements.txt

# Atau via PowerShell script
.\install.ps1
```

### 2. Jalankan Notebook
```bash
jupyter notebook
# Buka: notebook/notebook.ipynb
# Jalankan semua cell (Kernel -> Restart & Run All)
```

### 3. Hasil
Model akan:
- Train selama max 200 epoch (atau sampai early stopping)
- Menyimpan checkpoint terbaik ke `model/best_gcn_model.pth`
- Generate 4 visualisasi di folder `model/`

## Step-by-Step Tutorial

### Langkah 1: Import Libraries
Notebook akan mengimport semua library yang diperlukan:
- PyTorch & PyTorch Geometric untuk GNN
- Matplotlib untuk visualisasi
- Scikit-learn untuk evaluasi

### Langkah 2: Setup Device
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Otomatis menggunakan GPU jika tersedia, kalau tidak akan menggunakan CPU.

### Langkah 3: Load Dataset
Dataset Cora akan di-download otomatis ke folder `dataset/Cora/`

**Info Dataset:**
- 2,708 nodes (papers)
- 10,556 edges (citations)
- 1,433 features per node
- 7 classes

### Langkah 4: Bangun Model
Model GCN dengan 2 layer:
```
Input (1433) -> GCNConv -> ReLU -> Dropout 
            -> GCNConv -> LogSoftmax -> Output (7)
```

### Langkah 5: Training
Training dengan fitur:
- **Early Stopping**: Stop jika val_acc tidak meningkat selama 10 epoch
- **Model Checkpoint**: Simpan model terbaik
- **LR Scheduler**: Turunkan learning rate setiap 50 epoch

### Langkah 6: Evaluasi
Model terbaik dievaluasi pada test set dengan:
- Accuracy
- Confusion Matrix
- Classification Report (precision, recall, f1-score)

### Langkah 7: Visualisasi
Gunakan t-SNE untuk visualisasi node embeddings dalam 2D.

## Customization

### Mengubah Hyperparameters

```python
# Dalam notebook, ubah nilai berikut:

# Hidden layer size
hidden_channels = 32  # Default: 16

# Dropout rate
dropout = 0.6  # Default: 0.5

# Learning rate
lr = 0.005  # Default: 0.01

# Weight decay (L2 regularization)
weight_decay = 1e-3  # Default: 5e-4

# Early stopping patience
patience = 15  # Default: 10

# Max epochs
max_epochs = 300  # Default: 200
```

### Menggunakan Arsitektur Berbeda

#### GAT (Graph Attention Network)
```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, 
                            heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, 
                            heads=1, concat=False, dropout=dropout)
        self.dropout = Dropout(p=dropout)
    
    def forward(self, x, edge_index, return_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        embedding = x
        x = self.dropout(x)
        
        if return_embedding:
            return embedding
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Inisialisasi
model = GAT(dataset.num_features, 8, dataset.num_classes).to(device)
```

#### GraphSAGE
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = Dropout(p=dropout)
    
    def forward(self, x, edge_index, return_embedding=False):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = x
        x = self.dropout(x)
        
        if return_embedding:
            return embedding
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Inisialisasi
model = GraphSAGE(dataset.num_features, 16, dataset.num_classes).to(device)
```

### Menggunakan Dataset Berbeda

PyTorch Geometric menyediakan banyak dataset:

```python
from torch_geometric.datasets import Planetoid

# CiteSeer
dataset = Planetoid(root='../dataset/CiteSeer', name='CiteSeer')

# PubMed
dataset = Planetoid(root='../dataset/PubMed', name='PubMed')

# Amazon Computers
from torch_geometric.datasets import Amazon
dataset = Amazon(root='../dataset/Amazon', name='Computers')

# Reddit (large scale)
from torch_geometric.datasets import Reddit
dataset = Reddit(root='../dataset/Reddit')
```

## Tips & Tricks

### 1. Meningkatkan Performa

**Tambah Hidden Layers:**
```python
class GCN3Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN3Layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
```

**Batch Normalization:**
```python
from torch.nn import BatchNorm1d

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
```

**Residual Connections:**
```python
class GCNResidual(torch.nn.Module):
    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 2. Debugging Tips

**Check untuk overfitting:**
```python
# Jika train_acc >> val_acc, model overfit
# Solusi:
# 1. Tingkatkan dropout
# 2. Tambah weight_decay
# 3. Early stopping dengan patience lebih kecil
```

**Check untuk underfitting:**
```python
# Jika train_acc rendah dan val_acc rendah
# Solusi:
# 1. Tambah hidden_channels
# 2. Tambah layer
# 3. Turunkan dropout
# 4. Train lebih lama
```

**Memory Management:**
```python
# Jika CUDA out of memory:
# 1. Gunakan CPU
device = torch.device('cpu')

# 2. Atau kurangi hidden_channels
hidden_channels = 8  # dari 16
```

### 3. Monitoring Training

**TensorBoard (Opsional):**
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gcn_experiment')

# Dalam training loop:
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
```

### 4. Ensemble Models

```python
# Train multiple models dengan random seed berbeda
models = []
for seed in [42, 123, 456, 789, 999]:
    torch.manual_seed(seed)
    model = GCN(...).to(device)
    # ... train model ...
    models.append(model)

# Prediksi dengan voting
def ensemble_predict(models, data):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            predictions.append(pred)
    
    # Majority voting
    predictions = torch.stack(predictions)
    final_pred = predictions.mode(dim=0).values
    return final_pred
```

## FAQ

### Q: Apakah bisa dijalankan tanpa GPU?
**A:** Ya! Model akan otomatis menggunakan CPU jika GPU tidak tersedia. Training akan lebih lambat (~2-3 menit di CPU vs ~30 detik di GPU).

### Q: Bagaimana cara menyimpan dan load model?
**A:** Model otomatis disimpan di `model/best_gcn_model.pth`. Untuk load:
```python
checkpoint = torch.load('../model/best_gcn_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Q: Hasil akurasi berbeda setiap kali run?
**A:** Normal karena inisialisasi random. Untuk hasil konsisten:
```python
torch.manual_seed(42)
np.random.seed(42)
```

### Q: Bagaimana cara menambah layer?
**A:** Lihat bagian [Customization](#customization) untuk contoh GCN 3-layer.

### Q: Error: "torch_geometric not found"?
**A:** Install PyTorch Geometric:
```bash
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster
```

### Q: Akurasi tidak meningkat?
**A:** Coba:
1. Turunkan learning rate: `lr=0.005`
2. Tambah hidden_channels: `hidden_channels=32`
3. Kurangi dropout: `dropout=0.3`
4. Train lebih lama: `max_epochs=300`

### Q: Bagaimana cara visualisasi graph?
**A:** Gunakan NetworkX:
```python
import networkx as nx

# Convert to NetworkX
G = torch_geometric.utils.to_networkx(data)

# Plot
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=data.y.cpu(), node_size=20, cmap='tab10')
plt.show()
```

### Q: Bisa digunakan untuk dataset sendiri?
**A:** Ya! Format data harus:
```python
from torch_geometric.data import Data

# Buat data object
data = Data(
    x=node_features,      # [num_nodes, num_features]
    edge_index=edges,     # [2, num_edges]
    y=labels,             # [num_nodes]
    train_mask=...,       # [num_nodes] boolean
    val_mask=...,         # [num_nodes] boolean
    test_mask=...         # [num_nodes] boolean
)
```

## 📞 Support

Jika ada pertanyaan atau issue:
1. Check dokumentasi PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
2. Review notebook dan comments dengan teliti
3. Check error message untuk troubleshooting hints

## 🎓 Learning Resources

- **PyTorch Geometric Tutorial**: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
- **GCN Paper**: https://arxiv.org/abs/1609.02907
- **Graph Neural Networks Explained**: https://distill.pub/2021/gnn-intro/
- **CS224W Stanford Course**: http://web.stanford.edu/class/cs224w/

---

Happy coding! 🚀
