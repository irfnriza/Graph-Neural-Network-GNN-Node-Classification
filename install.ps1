# Installation Script for Graph Neural Network Project
# PowerShell script untuk Windows

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "GNN Project - Installation Script" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing PyTorch..." -ForegroundColor Yellow
Write-Host "Note: This will install CPU version. For GPU, please install manually." -ForegroundColor Gray
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric

Write-Host ""
Write-Host "Installing PyG dependencies..." -ForegroundColor Yellow
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

Write-Host ""
Write-Host "Installing additional packages..." -ForegroundColor Yellow
pip install matplotlib scikit-learn numpy jupyter ipykernel

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Installation completed!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Open Jupyter: jupyter notebook" -ForegroundColor White
Write-Host "2. Navigate to: notebook/notebook.ipynb" -ForegroundColor White
Write-Host "3. Run all cells to train the model" -ForegroundColor White
Write-Host ""
Write-Host "For GPU support, please reinstall PyTorch with CUDA:" -ForegroundColor Gray
Write-Host "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Gray
