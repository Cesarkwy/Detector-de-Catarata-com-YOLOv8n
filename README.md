# PDI - Detecção de Catarata em Vídeos

Projeto de Processamento Digital de Imagens (PDI) para detecção automática de cataratas em vídeos de olhos utilizando YOLOv8.

## Descrição do Projeto

Este projeto implementa uma pipeline completa de detecção de estruturas oftalmológicas em vídeos:

- **Detecção de Córnea**: Identifica a região da córnea no olho
- **Detecção de Pupila**: Detecta a pupila com fallback para transformada de Hough
- **Detecção de Catarata**: Identifica a presença e localização de catarata
- **Análise de Incisão**: Calcula o ângulo dominante da incisão cirúrgica dentro da catarata
- **Exportação de Métricas**: Salva dados de análise em arquivo CSV por frame

### Saída Visual

O vídeo anotado contém overlays com as seguintes cores:
- 🔴 **Vermelho**: Córnea (detecção do modelo)
- 🔵 **Azul**: Pupila (detecção do modelo)
- 🟡 **Amarelo**: Pupila (fallback Hough, quando modelo não detecta)
- 🟢 **Verde**: Catarata (detecção do modelo com linha de incisão)

## Requisitos

### Dependências do Sistema
- Python 3.8+
- pip (gerenciador de pacotes Python)

### Bibliotecas Python Necessárias

```
opencv-python>=4.5.0
numpy>=1.19.0
ultralytics>=8.0.0  # YOLOv8
torch>=1.9.0        # Backend do YOLOv8
torchvision>=0.10.0
```

## Instalação

### 1. Clonar/Preparar o Repositório

```bash
cd c:\Users\cesar\Desktop\PDI2
```

### 2. Criar Ambiente Virtual (Recomendado)

```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Se der erro de permissão, execute:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Instalar Dependências

```powershell
# Instalar dependências do arquivo requirements.txt
pip install -r requirements.txt

# Ou instalar manualmente:
pip install opencv-python numpy ultralytics torch torchvision
```

### 4. Verificar Instalação

```powershell
python -c "import cv2, numpy, torch, ultralytics; print(' Todas as dependências estão OK!')"
```

## Como Usar

### Estrutura de Diretórios Esperada

```
PDI2/
├── infer_video_yolov8_bbox.py    # Script principal de inferência
├── extract_frames.py              # Script para extrair frames
├── requirements.txt               # Dependências
├── yolov8n.pt                     # Modelo YOLOv8 (nano)
├── data/
│   ├── raw/                       # Vídeos de entrada
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── frames/                    # Frames extraídos (gerado)
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt        # Modelo treinado (seu modelo)
└── debug_videoX/                  # Frames de debug (gerado)
```

### Exemplo 1: Inferência Básica

Executar detecção em um vídeo com configuração padrão:

```powershell
python infer_video_yolov8_bbox.py `
  --model runs\detect\train\weights\best.pt `
  --source data\raw\video2.mp4 `
  --output video2_annot_debug.mp4 `
  --csv video2_metrics_debug.csv
```

### Exemplo 2: Inferência Completa (Recomendado)

Usar todos os parâmetros para máximo controle:

```powershell
python infer_video_yolov8_bbox.py `
  --model runs\detect\train\weights\best.pt `
  --source data\raw\video2.mp4 `
  --output video2_annot_debug.mp4 `
  --csv video2_metrics_debug.csv `
  --device cpu `
  --imgsz 960 `
  --conf 0.1 `
  --iou 0.7 `
  --cornea_alias "cornea,iris" `
  --pupil_alias "pupil,pupila" `
  --catarata_alias "catarata" `
  --show_all `
  --debug_dir debug_video2
```

### Exemplo 3: Com GPU (CUDA)

Se você tiver NVIDIA GPU instalada:

```powershell
python infer_video_yolov8_bbox.py `
  --model runs\detect\train\weights\best.pt `
  --source data\raw\video1.mp4 `
  --output video1_annot.mp4 `
  --csv video1_metrics.csv `
  --device cuda:0 `
  --imgsz 960 `
  --conf 0.15
```

### Exemplo 4: Extrair Frames de um Vídeo

Preparar dataset extraindo frames em intervalos regulares:

```powershell
python extract_frames.py `
  --video data\raw\video1.mp4 `
  --out data\frames\video1 `
  --step 30
```

Isso vai salvar 1 frame a cada 30 quadros do vídeo em `data/frames/video1/`.

## Parâmetros de Configuração

### Parâmetros de Modelo

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--model` | *obrigatório* | Caminho para o modelo YOLOv8 (.pt) |
| `--source` | *obrigatório* | Caminho do vídeo de entrada |
| `--device` | None | Device: `cpu`, `cuda:0`, `dml` |
| `--imgsz` | 640 | Tamanho de entrada da rede (640, 960, 1280) |
| `--conf` | 0.05 | Limiar de confiança para detecção (0.0-1.0) |
| `--iou` | 0.7 | Limiar IoU para NMS (0.0-1.0) |

### Parâmetros de Saída

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--output` | `bbox_annot.mp4` | Arquivo de vídeo anotado de saída |
| `--csv` | `bbox_metrics.csv` | Arquivo CSV com métricas por frame |
| `--debug_dir` | None | Diretório para salvar frames de debug |

### Parâmetros de Processamento

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--ema` | 0.0 | Fator de suavização EMA (0.0 = sem suavização) |
| `--show_all` | False | Flag para desenhar todas as detecções (debug) |

### Parâmetros de Mapeamento de Classes

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--cornea_alias` | `cornea` | Aliases para córnea (separados por vírgula) |
| `--pupil_alias` | `pupil,pupila` | Aliases para pupila |
| `--catarata_alias` | `catarata,cataract` | Aliases para catarata |

## Saída CSV

O arquivo CSV gerado contém as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| `frame` | Número do frame |
| `time_s` | Tempo em segundos |
| `cornea_area` | Área da bounding box da córnea (pixels²) |
| `cornea_conf` | Confiança da detecção da córnea (0.0-1.0) |
| `pupil_area` | Área da bounding box da pupila (pixels²) |
| `pupil_conf` | Confiança da detecção da pupila (0.0-1.0) |
| `catarata_area` | Área da bounding box da catarata (pixels²) |
| `catarata_conf` | Confiança da detecção da catarata (0.0-1.0) |
| `catarata_presence` | 1 se catarata detectada, 0 caso contrário |
| `center_distance_px` | Distância euclidiana entre centros de córnea e pupila (pixels) |
| `incision_angle_deg` | Ângulo dominante da incisão (0-180°, onde 0° = horizontal) |

## Dicas de Uso

### Ajustar Sensibilidade

- **Aumentar detecções**: reduzir `--conf` (ex: 0.05)
- **Menos falsos positivos**: aumentar `--conf` (ex: 0.25)

### Melhorar Performance

- **Reduzir tempo**: usar `--imgsz 640` (mais rápido) ou `--device cuda:0` (se tem GPU)
- **Melhorar qualidade**: usar `--imgsz 1280` (mais lento, mais preciso)

### Debug

- Use `--show_all` para visualizar todas as detecções (com confiança baixa)
- Use `--debug_dir debug_folder` para salvar primeiros 50 frames anotados
- Verifique o CSV para análises de métricas

## Estrutura do Código

### `infer_video_yolov8_bbox.py`

Script principal que implementa:

- **`dist()`**: Calcula distância euclidiana entre pontos
- **`ema_update()`**: Filtro de média móvel exponencial
- **`match_class_id()`**: Mapeia aliases de classe para IDs do modelo
- **`clip_int()`**: Limita valor inteiro a um intervalo
- **`find_pupil_hough_in_roi()`**: Detecta pupila com transformada de Hough (fallback)
- **`compute_dominant_orientation_deg()`**: Calcula ângulo de incisão via Hough Lines
- **`draw_dominant_line_on_overlay()`**: Desenha linha de orientação no vídeo
- **`parse_args()`**: Parser de argumentos de linha de comando
- **`main()`**: Função principal de processamento

### `extract_frames.py`

Script auxiliar para extrair frames de vídeos:

- **`main()`**: Extrai frames em intervalos regulares

## Troubleshooting

### Erro: "Não foi possível abrir o vídeo"
- Verifique o caminho do vídeo
- Certifique-se que o formato é suportado (MP4, AVI, MOV)

### Erro: "model not found"
- Verifique o caminho do modelo `.pt`
- Download do modelo: `yolov8n.pt` (nano), `yolov8m.pt` (médio), `yolov8l.pt` (grande)
- Foi testado utilizando o modelo: `yolov8n.pt`

### Erro de GPU: "CUDA out of memory"
- Reduza `--imgsz` (ex: 640)
- Use `--device cpu` para processar com CPU

### Nenhuma detecção no CSV
- Aumentar `--conf` para valores muito altos pode resultar em 0 detecções
- Verificar se o modelo foi treinado com os dados corretos
- Usar `--show_all` para debug
