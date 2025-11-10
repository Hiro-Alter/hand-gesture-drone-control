# Código para utilizar PyTorch con DirectML en Windows
import torch
import torch_directml
from torchvision import models

# Seleccionar dispositivo DirectML
device = torch_directml.device()

# Cargar un modelo preentrenado
model = models.resnet18(weights="IMAGENET1K_V1")

# Pasar el modelo a DirectML
model.to(device)

# Ejemplo de inferencia
dummy_input = torch.randn(1, 3, 224, 224, device=device)
output = model(dummy_input)
print("Inferencia OK:", output.shape)
