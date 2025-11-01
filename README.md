# Graph Neural Network (GNN) - Node Classification

Proyek ini mengimplementasikan **Graph Convolutional Network (GCN)** untuk tugas **Node Classification** menggunakan dataset **Cora**.

## 📋 Fitur Utama

✅ **Model GCN 2-layer** dengan PyTorch Geometric  
✅ **Early Stopping** dengan patience 10 epoch  
✅ **Model Checkpoint** - menyimpan model terbaik  
✅ **Learning Rate Scheduler** (StepLR)  
✅ **Visualisasi t-SNE** untuk node embeddings  
✅ **Confusion Matrix & Classification Report**  
✅ **Modular & Extensible** - mudah diganti ke GAT/GraphSAGE  

## 🗂️ Struktur Folder

```
Graph Neural Network/
├── dataset/               # Dataset Cora (auto-download)
│   └── Cora/
├── model/                 # Model checkpoint dan visualisasi
│   ├── best_gcn_model.pth
│   ├── training_progress.png
│   ├── confusion_matrix.png
│   ├── tsne_visualization.png
│   └── tsne_per_class.png
├── notebook/
│   └── notebook.ipynb     # Main notebook
└── README.md
```

## 📦 Instalasi Dependencies

### Requirement:
```bash
pip install torch torchvision
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install matplotlib scikit-learn numpy
```

**Catatan**: Sesuaikan versi PyTorch dan CUDA dengan sistem Anda.

## 🚀 Cara Menggunakan

1. **Buka notebook**: `notebook/notebook.ipynb`
2. **Jalankan semua cell** secara berurutan
3. Model akan otomatis:
   - Download dataset Cora
   - Train model dengan early stopping
   - Simpan model terbaik
   - Generate visualisasi

## 📊 Dataset Cora

- **Nodes**: 2,708 scientific publications
- **Edges**: 10,556 citation links
- **Features**: 1,433 word features per node
- **Classes**: 7 classes (topik penelitian)
- **Split**: 140 train / 500 validation / 1000 test

### Class Labels:
1. Case_Based
2. Genetic_Algorithms
3. Neural_Networks
4. Probabilistic_Methods
5. Reinforcement_Learning
6. Rule_Learning
7. Theory

## 🏗️ Arsitektur Model

```
GCN(
  (conv1): GCNConv(1433, 16)
  (conv2): GCNConv(16, 7)
  (dropout): Dropout(p=0.5)
)
```

**Total Parameters**: ~23,000

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden Channels | 16 |
| Dropout | 0.5 |
| Learning Rate | 0.01 |
| Weight Decay | 5e-4 |
| Optimizer | Adam |
| Scheduler | StepLR (step=50, gamma=0.5) |
| Max Epochs | 200 |
| Early Stopping Patience | 10 |

## 📈 Expected Performance

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~80%

*Hasil dapat bervariasi karena inisialisasi random*

## 🎯 Output Files

Setelah training selesai, file berikut akan disimpan di folder `model/`:

1. **best_gcn_model.pth** - Model checkpoint terbaik
2. **training_progress.png** - Loss, accuracy, dan learning rate
3. **confusion_matrix.png** - Confusion matrix pada test set
4. **tsne_visualization.png** - t-SNE embedding visualization
5. **tsne_per_class.png** - t-SNE per class

## 🔧 Cara Extend ke Arsitektur Lain

Model dirancang modular. Untuk menggunakan arsitektur berbeda:

### Graph Attention Network (GAT)
```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
    # ... rest of implementation
```

### GraphSAGE
```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    # ... rest of implementation
```

Kemudian ganti inisialisasi model di notebook.

## 📝 Code Highlights

### Training dengan Early Stopping
```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    save_checkpoint(model, optimizer, epoch, val_acc)
else:
    patience_counter += 1

if patience_counter >= patience:
    print("Early stopping triggered")
    break
```

### t-SNE Visualization
```python
embeddings = model(data.x, data.edge_index, return_embedding=True)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
```

## 🐛 Troubleshooting

### Error: CUDA out of memory
- Gunakan CPU: `device = torch.device('cpu')`
- Atau kurangi `hidden_channels`

### Error: PyTorch Geometric tidak terinstall
- Install dengan: `pip install torch-geometric`
- Install dependencies: `torch-scatter`, `torch-sparse`, `torch-cluster`

### Error: Dataset tidak bisa didownload
- Download manual dari [Cora Dataset](https://linqs.soe.ucsc.edu/data)
- Letakkan di folder `dataset/Cora/`

## 📚 Referensi

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **GCN Paper**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **Cora Dataset**: https://linqs.soe.ucsc.edu/data

## 👨‍💻 Author

Created for Graph Neural Network learning project.

## 📄 License

MIT License - Free to use and modify.

---

**Happy Learning! 🎓**
