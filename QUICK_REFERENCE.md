# Quick Reference Guide

## 🚀 Quick Commands

### Installation
```powershell
# Option 1: Using PowerShell script
.\install.ps1

# Option 2: Manual installation
pip install torch torchvision
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster
pip install matplotlib scikit-learn numpy jupyter
```

### Running the Project
```powershell
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab

# Navigate to: notebook/notebook.ipynb
# Run all cells: Kernel -> Restart & Run All
```

### Check Installation
```python
# In Python/Jupyter:
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

## 📁 File Locations

| File | Location | Purpose |
|------|----------|---------|
| Main Notebook | `notebook/notebook.ipynb` | Training & evaluation |
| Best Model | `model/best_gcn_model.pth` | Saved checkpoint |
| Config | `config.py` | Hyperparameters |
| Dataset | `dataset/Cora/` | Cora dataset |
| Visualizations | `model/*.png` | All plots |

## ⚙️ Key Parameters

### Model
```python
hidden_channels = 16        # Hidden layer size
dropout = 0.5               # Dropout rate
num_layers = 2              # Number of GCN layers
```

### Training
```python
max_epochs = 200            # Maximum epochs
learning_rate = 0.01        # Initial LR
weight_decay = 5e-4         # L2 regularization
patience = 10               # Early stopping
```

### Scheduler
```python
step_size = 50              # LR decay every N epochs
gamma = 0.5                 # LR decay factor
```

## 🎯 Common Tasks

### 1. Train Model
```python
# In notebook, run cells 1-16
# Model will train and save to: model/best_gcn_model.pth
```

### 2. Load Saved Model
```python
checkpoint = torch.load('../model/best_gcn_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 3. Predict Single Node
```python
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[node_idx].argmax()
    confidence = torch.exp(out[node_idx][pred])
```

### 4. Get Embeddings
```python
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index, return_embedding=True)
```

### 5. Evaluate on Test Set
```python
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[data.test_mask].argmax(dim=1)
    acc = (pred == data.y[data.test_mask]).sum() / data.test_mask.sum()
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Force CPU usage
device = torch.device('cpu')

# Or reduce model size
hidden_channels = 8
```

### Module Not Found
```powershell
# Reinstall PyTorch Geometric
pip install --upgrade torch-geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster
```

### Notebook Not Found
```powershell
# Make sure you're in the right directory
cd "c:\project\project-ai\Graph Neural Network"
jupyter notebook
```

### Import Errors
```python
# Check installations
import sys
print(sys.executable)  # Python path

pip list | grep torch  # Windows: pip list | findstr torch
```

## 📊 Expected Results

### Performance Metrics
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~80%
- **Training Time**: 30s (GPU) / 2-3min (CPU)

### Output Files
After running all cells, you should have:
```
model/
├── best_gcn_model.pth          # Model checkpoint
├── training_progress.png        # Loss & accuracy plots
├── confusion_matrix.png         # Confusion matrix
├── tsne_visualization.png       # t-SNE all nodes
└── tsne_per_class.png          # t-SNE per class
```

## 🎨 Visualization Preview

### Training Progress
- Plot 1: Training loss over epochs
- Plot 2: Validation accuracy over epochs
- Plot 3: Learning rate schedule

### t-SNE Visualization
- Shows node embeddings in 2D
- Color-coded by class
- Green edges: correct predictions
- Red edges: incorrect predictions

### Confusion Matrix
- 7x7 matrix for Cora classes
- Shows prediction distribution
- Darker = more predictions

## 💡 Pro Tips

### Speed Up Training
```python
# Use GPU
device = torch.device('cuda')

# Reduce logging
log_interval = 20
```

### Better Performance
```python
# More hidden units
hidden_channels = 32

# More layers
# Modify GCN class to add conv3

# Less dropout (if underfitting)
dropout = 0.3
```

### Debug Mode
```python
# Verbose output
print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

# Check gradients
for name, param in model.named_parameters():
    print(f"{name}: {param.grad.norm()}")
```

## 📚 Resources

### Documentation
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [GCN Paper](https://arxiv.org/abs/1609.02907)
- [Cora Dataset](https://linqs.soe.ucsc.edu/data)

### Tutorials
- [PyG Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/)
- [CS224W Stanford](http://web.stanford.edu/class/cs224w/)
- [Distill GNN](https://distill.pub/2021/gnn-intro/)

### Support
- GitHub Issues (if applicable)
- Stack Overflow: [pytorch-geometric]
- PyG Discussions: [GitHub Discussions](https://github.com/pyg-team/pytorch_geometric/discussions)

## 🔑 Keyboard Shortcuts (Jupyter)

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell |
| `Ctrl+Enter` | Run cell (stay) |
| `Alt+Enter` | Run cell (insert below) |
| `A` | Insert cell above |
| `B` | Insert cell below |
| `DD` | Delete cell |
| `M` | Change to markdown |
| `Y` | Change to code |
| `Ctrl+S` | Save notebook |

## 📞 Getting Help

1. **Read the error message** - usually contains the solution
2. **Check USAGE_GUIDE.md** - detailed explanations
3. **Review notebook comments** - inline documentation
4. **Check config.py** - parameter documentation
5. **Search PyG docs** - official documentation

---

**Version**: 1.0.0  
**Last Updated**: November 1, 2025  
**Quick Start**: `.\install.ps1` → `jupyter notebook` → Run All Cells
