# Deliverable 2 AI-HPC Lab: DISTRIBUTED

## Descripción general

El objetivo de esta segunda parte es **acelerar el entrenamiento del modelo BERT-Base** (`bert-base-uncased`) sobre el conjunto de datos **SQuAD v1.1**, utilizando **entrenamiento distribuido en PyTorch Lightning**.  

En esta versión, se han probado **dos estrategias de paralelización**:  
- **DDP (Distributed Data Parallel)**, la opción estándar y más eficiente para entrenamientos multi-GPU.  
- **FSDP (Fully Sharded Data Parallel)**, que optimiza el uso de memoria dividiendo los pesos del modelo entre GPUs.  

El objetivo es comparar el rendimiento en distintos escenarios y analizar el *speedup* obtenido respecto al baseline de una sola GPU A100.

---

## Estructura de archivos

| Archivo | Descripción |
|----------|-------------|
| `train_distributed.py` | Script principal en Python. Implementa el entrenamiento distribuido con PyTorch Lightning. La estrategia (`ddp` o `fsdp`) se selecciona en el `Trainer`. |
| `distributed.slurm` | Script SLURM para ejecutar el entrenamiento distribuido en uno o varios nodos del clúster. Utiliza `srun` para el lanzamiento correcto de procesos DDP/FSDP. |
| `README.md` | Este documento. Explica el código, configuración, resultados y comparativa con el baseline. |

---

## Entorno de ejecución

- **Sistema operativo:** Rocky Linux 8.8  
- **Python:** 3.10.8 (entorno virtual `venv`)  
- **Librerías principales:**
  - `torch`
  - `pytorch-lightning`
  - `transformers`
  - `datasets`
  - `tensorboard`
- **Nodos del clúster:** `a100-xx`
- **GPU:** NVIDIA A100-PCIE-40GB
- **Driver:** 570.86.15  
- **CUDA:** 12.8  
- **Kernel:** 4.18.0  

---

## Implementación distribuida

### 1. Cambios respecto al baseline
- Sustitución del `Trainer` de Hugging Face por **PyTorch Lightning Trainer**. 
- Definición de una clase `BertQA(pl.LightningModule)` que implementa:
  - `training_step` y `validation_step` para el cálculo de pérdidas.
  - `configure_optimizers` con el optimizador **AdamW** (mismo que en baseline). 
- Uso de `DataLoader` propios para entrenamiento y validación.  
- Activación del modo distribuido mediante:

```python
  trainer = pl.Trainer(
      accelerator="gpu",
      devices="auto",
      num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
      strategy="ddp",  # o "fsdp"
      max_epochs=4,
      logger=logger,
      callbacks=[checkpoint_callback]
  )
```
El número de nodos y GPUs se detecta automáticamente desde las variables de entorno de SLURM.

## Configuración del entrenamiento

| Parámetro | Valor |
|------------|--------|
| Modelo | `bert-base-uncased` |
| Dataset | SQuAD v1.1 |
| Ejemplos usados | 10,000 (entrenamiento) / 500 (validación) |
| Épocas | 4 |
| Tamaño de batch | 4 |
| Learning rate | 3e-5 |
| Optimizador | AdamW |
| Estrategia de entrenamiento | `ddp` o `fsdp` (según el experimento) |
| Registro | TensorBoard (`./logs`) |
| Checkpoints | `ModelCheckpoint` guardando el mejor modelo (menor `val_loss`) |
| Dispositivo | GPU (NVIDIA A100) |
| Framework | PyTorch Lightning |

## Ejecución de SLURM

El trabajo se lanza con:
```bash
sbatch distributed.slurm
```

Este script activa el entorno virtual, ejecuta el entrenamiento mediante srun y registra la información del nodo de computo con la ejecución de nvidia-smi.
Se probaron configuraciones con 1 o 2 nodos, y 1 o 2 GPUs por nodo, dependiendo de la estrategia.


## Resultados

| Estrategia | Nodos | GPUs totales | Tiempo (min) | Speedup vs Baseline |
|-------------|--------|---------------|---------------|----------------------|
| **Baseline (HF Trainer)** | 1 | 1 | 12.98 | 1.00× |
| **DDP** | 1 | 1 | 13.77 | 0.94× |
| **DDP** | 1 | 2 | 7.78 | 1.67× |
| **DDP** | 2 | 4 | 4.77 | 2.72× |
| **FSDP** | 1 | 2 | 9.99 | 1.30× |
| **FSDP** | 2 | 4 | 4.44 | 2.92× |



## Análisis de rendimiento

- DDP mostró una escalabilidad casi lineal, alcanzando un speedup ×2.7 al pasar de una GPU a 4 GPUs distribuidas en dos nodos.

- FSDP obtiene resultados similares pero con un leve overhead inicial (la configuración 1 nodo, 2 GPUs es levemente mas lenta), compensado al aumentar el número de GPUs.

- En configuraciones pequeñas (1 GPU), los tiempos son algo mayores probablemente por el coste de inicialización del backend de Nvidia para comunicaciones entre varias GPUs (nccl).

- En todos los casos, se verificó el uso correcto de las GPUs mediante nvidia-smi y los logs de PyTorch Lightning (LOCAL_RANK, CUDA_VISIBLE_DEVICES).

- No se detectaron errores de sincronización ni pérdidas de rendimiento por I/O.


## Observaciones

- El rendimiento mejora significativamente a partir de 2 GPUs o más, con reducciones de tiempo superiores al 60%.

- FSDP ofrece un uso más eficiente de memoria, lo que podría permitir batch sizes mayores en futuros experimentos.

- Se mantuvo el mismo preprocesamiento, dataset y parámetros de entrenamiento que el baseline para una comparación justa.

- Los tiempos reportados son la mediana de 5 ejecuciones por configuración para reducir la variabilidad. Se han adjuntado al repositorio los logs de la ejecución elegida como mediana para cada uno de los casos

## Conclusión

El entrenamiento distribuido con PyTorch Lightning permite acelerar de forma notable el entrenamiento de modelos grandes como BERT.

En particular:

- DDP ofrece la mejor relación simplicidad / rendimiento.

- FSDP muestra potencial en entornos de gran escala, con menor consumo de memoria.

- Se obtiene un speedup de casi 3× al usar 2 nodos con 2 GPUs A100 cada uno.