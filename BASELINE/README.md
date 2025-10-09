# Deliverable 1 AI-HPC Lab: BASELINE

## Descripción general

El objetivo de esta práctica es medir el tiempo de entrenamiento de un modelo de Deep Learning en PyTorch y utilizarlo como referencia (baseline) para comparar posteriormente con una versión distribuida (Deliverable 2).

En esta primera parte, se ha utilizado el modelo BERT-Base (`bert-base-uncased`) sobre el conjunto de datos SQuAD v1.1, realizando el entrenamiento en una única GPU NVIDIA A100 del clúster HPC proporcionado en la asignatura.

---

## Estructura de archivos

| Archivo | Descripción |
|----------|-------------|
| `train_baseline.py` | Script principal en Python. Implementa el entrenamiento de BERT para *Question Answering* usando la librería **Hugging Face Transformers** y PyTorch. Incluye medición del tiempo total de entrenamiento. |
| `run_baseline.sh` | Script Bash que activa el entorno virtual, lanza el entrenamiento y guarda los logs. |
| `baseline.slurm` | Script SLURM para enviar el trabajo al planificador del clúster, solicitando **una GPU A100**, 32 CPUs y 96 GB de RAM. Incluye comprobación del nodo y del modelo de GPU mediante `nvidia-smi`. |
| `README.md` | Este documento. Explica los contenidos, los resultados y observaciones del entrenamiento. |

---

## Entorno de ejecución

- **Sistema operativo:** Rocky Linux 8.8
- **Python:** 3.10.8 (entorno virtual `venv`)
- **Librerías principales:**
  - `torch`
  - `transformers` 
  - `datasets`
  - `accelerate`
  - `tensorboard`
- **Nodo del clúster:** `c210-12
- **GPU:** `NVIDIA A100-PCIE-40GB`
- **Versión del driver:** 570.86.15
- **Versión de CUDA:** 12.8
- **Versión del kernel:** 4.18.0 (inferior a la recomendada 5.5.0, aunque sin problemas durante la ejecución)

---

## Configuración del entrenamiento

| Parámetro | Valor |
|------------|--------|
| Modelo | `bert-base-uncased` |
| Dataset | SQuAD v1.1 |
| Ejemplos usados | 10000 de entrenamiento / 500 de validación |
| Épocas | 4 |
| Tamaño de batch | 4 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Registro | TensorBoard (`./logs`) |
| Optimizador | AdamW (por defecto en Transformers) |

---

## Ejecución

El entrenamiento se realizó enviando el trabajo con:

```bash
sbatch baseline.slurm
```

Este script solicita una GPU A100 y ejecuta run_baseline.sh, que a su vez lanza train_baseline.py dentro del entorno virtual venv.

Durante la ejecución, SLURM asignó el nodo c210-12, como se confirma en el log mediante la salida de nvidia-smi:

- Running on host: c210-12
- GPU: NVIDIA A100-PCIE-40GB
- Driver Version: 570.86.15, CUDA Version: 12.8


---

## Resultados

| Métrica | Valor |
|----------|--------|
| Tiempo total de entrenamiento | **779.3411 s** (~**12.98 min**) |
| Samples por segundo | 51.325 |
| Steps por segundo | 12.831 |
| Pérdida final (`train_loss`) | 0.00648 |

---

## Perfilado y rendimiento

- El entrenamiento fue **estable y sin errores**. 
- Se observó una utilización correcta de GPU (modo `cuda` activado automáticamente). 
- Los resultados de `nvidia-smi` confirman el uso de la GPU A100 durante todo el proceso. 
- Aunque no se realizó un perfilado avanzado con herramientas adicionales, los tiempos registrados por el Trainer son representativos del rendimiento esperado en una A100.

## Observaciones

- El entrenamiento se completó correctamente en menos de 6 minutos, lo que confirma que la infraestructura HPC está bien configurada para PyTorch con GPU.

- Se utilizaron etiquetas dummy para las posiciones de respuesta en el dataset, dado que el objetivo principal es la medición de rendimiento, no la calidad del modelo.

- Se recomienda no ejecutar este entrenamiento en nodos con GPU T4, ya que los tiempos aumentan significativamente (~5 × más lentos).

- Este resultado servirá como baseline para comparar el speedup obtenido en el Deliverable 2 (entrenamiento distribuido con dos nodos y cuatro A100s).

## Conclusión

El baseline se ha completado con éxito cumpliendo todos los requisitos del enunciado:

- Uso de PyTorch y Transformers.

- Ejecución en una sola GPU A100.

- Medición y reporte del tiempo de entrenamiento.

- Resultados reproducibles y documentados.

Estos datos servirán como punto de partida para implementar y comparar el entrenamiento distribuido en el siguiente entregable.