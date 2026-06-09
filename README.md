# Pipeline OCR de Placas Veiculares Brasileiras

## Descrição

Pipeline completo de **detecção, segmentação e reconhecimento** de placas veiculares brasileiras (padrão Mercosul) a partir de imagens fotográficas, com comparação entre dois motores de OCR:

- **Tesseract OCR** – motor clássico baseado em heurísticas e LSTM
- **EasyOCR** – motor moderno baseado em redes neurais (CRAFT + CRNN)

## Dataset

20 imagens de veículos com placas no padrão Mercosul (BRASIL).  
**Observação importante:** as imagens possuem uma **marca d'água digital** (`BRASIL MERCOSUL` repetida em tom médio-cinza ~80–150) sobreposta à região branca dos caracteres, o que dificulta a binarização e reduz a acurácia do OCR convencional.

## Estrutura do Projeto

```
plate_ocr/
├── images/                      # Dataset: 20 imagens JPG
├── results_tesseract/
│   ├── resultados_tesseract.txt # Resultados em texto (tabela)
│   └── resultados_tesseract.json
├── results_easyocr/
│   ├── resultados_easyocr.txt   # Resultados em texto (tabela)
│   └── resultados_easyocr.json
├── run_tesseract.py             # Pipeline Tesseract (completo, executável)
├── run_easyocr.py               # Pipeline EasyOCR  (completo, executável)
├── pipeline_tesseract.py        # Versão modular com utils/
├── pipeline_easyocr.py          # Versão modular com utils/
├── utils/
│   ├── __init__.py
│   └── preprocessing.py         # Funções compartilhadas de pré-processamento
└── README.md
```

## Pipeline — Etapas

```
Imagem
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ETAPA 1 – DETECÇÃO DA PLACA (3 estratégias encadeadas)          │
│                                                                   │
│  1. Moldura escura: dilata pixels <15 → busca retângulo 2×–8×   │
│  2. Faixa azul Mercosul: HSV hue 90–130, componente horizontal   │
│  3. Alto contraste: varredura de linhas com std > 40             │
└─────────────────────────────────────────────────────────────────┘
  │ ROI da placa
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ETAPA 2 – PRÉ-PROCESSAMENTO (múltiplas variantes)               │
│                                                                   │
│  · Remove faixa azul (top 20%) e margens (3% laterais)          │
│  · Threshold adaptativo  bs=31/41/51/61, C=10/12/14/16          │
│    → melhor para watermark (janela local reduz fundo uniforme)   │
│  · Threshold fixo        th=60/75/90/105                         │
│  · CLAHE + Otsu          (equalização adaptativa de histograma)  │
└─────────────────────────────────────────────────────────────────┘
  │ Imagens binarizadas
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ETAPA 3 – OCR                                                    │
│                                                                   │
│  Tesseract: PSM 7 / 6 / 8  +  whitelist A–Z 0–9                 │
│  EasyOCR:  CRAFT (detector) + CRNN (reconhecedor)               │
│            → imagem colorida + variantes binarizadas             │
└─────────────────────────────────────────────────────────────────┘
  │ Lista de candidatos
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ETAPA 4 – SELEÇÃO DO MELHOR RESULTADO                           │
│                                                                   │
│  Tesseract: votação por maioria entre resultados de 7 chars      │
│  EasyOCR:  seleção por maior confiança; fusão de fragmentos      │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Texto da placa + metadados
```

## Como Executar

### Tesseract OCR
```bash
# Instalar dependências
pip install opencv-python pytesseract pillow numpy
# Ubuntu/Debian: sudo apt install tesseract-ocr

# Executar
python run_tesseract.py --images ./images --output ./results_tesseract

# Com imagens de debug (ROIs salvas)
python run_tesseract.py --images ./images --output ./results_tesseract --debug
```

### EasyOCR
```bash
# Instalar dependências (requer ~2 GB, inclui PyTorch)
pip install easyocr torch torchvision opencv-python numpy

# Executar
python run_easyocr.py --images ./images --output ./results_easyocr
```

## Comparação Tesseract vs EasyOCR

| Característica         | Tesseract 5                       | EasyOCR                          |
|------------------------|-----------------------------------|----------------------------------|
| Abordagem              | LSTM + heurísticas                | CRAFT + CRNN (deep learning)     |
| Velocidade (CPU)       | Rápido (~0.5 s/imagem)            | Mais lento (~2–5 s/imagem)       |
| Velocidade (GPU)       | Não usa GPU                       | Muito rápido com CUDA            |
| Robustez a distorções  | Média                             | Alta                             |
| Robustez a watermark   | Baixa–média                       | Média–alta (processa cor)        |
| Saída de confiança     | Não nativo                        | Sim (0.0–1.0 por leitura)        |
| Instalação             | Leve (binário + modelo)           | Pesada (PyTorch + modelos DL)    |
| Whitelist de chars     | Sim (`tessedit_char_whitelist`)   | Sim (`allowlist`)                |

## Desafio do Dataset

A marca d'água (`BRASIL MERCOSUL`) repetida em tom médio-cinza (valor 80–150)
sobre o fundo branco da placa (>200) cria um padrão de ruído que:

- Interfere com limiares globais (Otsu confunde watermark com caracteres)
- Reduz o contraste relativo dos caracteres reais
- Cria artefatos em formas de letras que o OCR pode misturar com os chars reais

**Solução adotada:** threshold adaptativo com janela grande (block_size=51–61)
que normaliza o fundo localmente, reduzindo o efeito do watermark uniforme.

## Resultados

Os resultados detalhados estão em:
- `results_tesseract/resultados_tesseract.txt`
- `results_easyocr/resultados_easyocr.txt`

O EasyOCR tende a ter melhor desempenho neste dataset por:
1. Processar a imagem colorida (chars escuros em fundo cinza-watermark)
2. Usar detector neural (CRAFT) menos sensível a ruído uniforme
3. Retornar confiança que permite selecionar a leitura mais confiável
