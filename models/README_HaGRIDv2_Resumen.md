# 🖐️ HaGRIDv2 – Dataset de Reconocimiento de Gestos Manuales

## 📘 Resumen del Dataset

**HaGRIDv2** (*HAnd Gesture Recognition Image Dataset*) es un conjunto de imágenes diseñado para el entrenamiento y validación de sistemas de **reconocimiento de gestos manuales (HGR)**.  
Contiene imágenes RGB en **FullHD**, con alta diversidad de personas, iluminación y contextos.

- **Tamaño total:** 1.5 TB  
- **Total de imágenes:** 1,086,158  
- **Clases de gestos:** 33 + 1 clase adicional “no_gesture”  
- **División:**  
  - Entrenamiento: 76 % (821,458 imágenes)  
  - Validación: 9 % (99,200 imágenes)  
  - Prueba: 15 % (165,500 imágenes)  
- **Participantes:** 65,977 personas únicas  
- **Formato de anotaciones:** COCO JSON con bounding boxes, etiquetas y puntos de referencia de MediaPipe.  

El dataset puede utilizarse para tareas de **clasificación** o **detección** de gestos, y es especialmente útil en aplicaciones de:
- Control gestual sin contacto  
- Videoconferencias interactivas  
- Robótica y automatización  
- Sistemas vehiculares inteligentes  

📄 Referencia:  
Nuzhdin, A. et al. (2024). *HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition*. arXiv: [2412.01508](https://arxiv.org/abs/2412.01508)

---

## ⚙️ Instalación

```bash
git clone https://github.com/hukenovs/hagrid.git
cd hagrid
conda create -n gestures python=3.11 -y
conda activate gestures
pip install -r requirements.txt
```

---

## 📥 Descarga del Dataset

El conjunto de datos puede descargarse desde los enlaces oficiales o mediante el script `download.py`:

```bash
python download.py --save_path <RUTA> --annotations --dataset
```

---

## 🧠 Modelos Preentrenados Disponibles

| Tipo de modelo | Arquitectura | mAP / F1 |
|----------------|--------------|-----------|
| Detección de gestos | YOLOv10x | 89.4 |
| Detección de gestos | YOLOv10n | 88.2 |
| Clasificación (Full Frame) | ResNet152 | 98.6 |
| Clasificación (Full Frame) | MobileNetV3_large | 93.4 |

---

## 🚁 Aplicación en el Proyecto: Control Gestual de Dron Virtual

Este proyecto utiliza una **selección optimizada de gestos** del conjunto HaGRIDv2 para controlar un dron virtual.  
Los gestos fueron elegidos por su claridad visual y ergonomía.

| Acción del dron | Gesto | Descripción |
|------------------|--------|--------------|
| **Despegar / Encender motores** | ✊ `fist` | Inicio de vuelo o activación. |
| **Aterrizar / Apagar motores** | ✋ `palm` | Señal universal de detención. |
| **Detenerse / Hover** | ✋ `stop` | Pausar el movimiento. |
| **Avanzar** | 👉 `point` | Indica dirección hacia adelante. |
| **Retroceder** | 👎 `dislike` | Movimiento inverso o atrás. |
| **Moverse a la izquierda** | ✌️ `peace` | Dirección lateral izquierda. |
| **Moverse a la derecha** | ✌️ `peace inverted` | Dirección lateral derecha. |
| **Subir** | ✌️ `two up` | Ascenso vertical. |
| **Bajar** | ✌️ `two up inverted` | Descenso vertical. |
| **Girar (rotar sobre eje Z)** | 🤘 `rock` | Giro controlado del dron. |

---

## 📄 Licencia

Este dataset está disponible bajo la licencia **Creative Commons Atribución-CompartirIgual 4.0 Internacional (CC BY-SA 4.0)**.  
Consulta la licencia completa en:  
[https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

---

## 👥 Créditos

Proyecto HaGRIDv2 desarrollado por:
- Alexander Kapitanov  
- Andrey Makhlyarchuk  
- Karina Kvanchiani  
- Aleksandr Nagaev  
- Roman Kraynov  
- Anton Nuzhdin  

Repositorio oficial: [https://github.com/hukenovs/hagrid](https://github.com/hukenovs/hagrid)
