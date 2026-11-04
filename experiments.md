# Эксперименты

Обновить: `python scripts/expeiment_to_md.py`. Конфигурация и числа заполняются автоматически.
Новые запуски из outputs добавляются при обновлении. Поля в блоках «Описание и выводы»
редактируются вручную и сохраняются; «Изменения» и первая строка «Вывод» попадают в общую таблицу.

Loss и train IoU приведены на эпохе лучшего valid IoU. Эпохи считаются с 1.
`—` означает, что значения нет в логах. `*` — среднее записанных batch loss за эпоху,
если epoch loss отсутствует; без взвешивания по размеру батча и дополнительной синхронизации GPU.

| Эксперимент | Что изменили | Valid IoU ↑ | Valid loss ↓ | Эпоха | Вывод |
| --- | --- | --- | --- | --- | --- |
| [001_baseline · slurm_8518584](#slurm_8518584) | — | 0.85647 | 0.24351* | 30 | Главная проблема в переобучении. |

<!-- experiment:{"id": "slurm_8518584", "name": "001_baseline", "tensorboard": "slurm_8518584/tensorboard", "epochs": "30/30", "best_epoch": "30", "train_loss": "0.06550*", "valid_loss": "0.24351*", "train_iou": "0.93655", "valid_iou": "0.85647"} -->
<a id="slurm_8518584"></a>

## 001_baseline · slurm_8518584

### Конфигурация

| Параметр | Значение |
| --- | --- |
| Модель | BasicUnet(in_channels=3, num_classes=1, base_channels=32) |
| Размер изображения | [256, 256] |
| Batch size на GPU | 4 |
| Узлы / GPU на узел | 2 / 1 |
| Накопление градиентов | 1 |
| Optimizer | Adam(lr=0.0001) |
| Loss | BCEWithLogitsLoss() |
| Scheduler | нет |
| Precision | 32-true |
| Seed | 42 |

<details>
<summary>Полная конфигурация</summary>

| Параметр | Значение |
| --- | --- |
| `name` | 001_baseline |
| `seed` | 42 |
| `checkpoint_dir` | ${hydra:runtime.output_dir}/checkpoints |
| `data.root` | data/train/data |
| `data.image_size` | [256, 256] |
| `data.batch_size` | 4 |
| `data.train_workers` | 3 |
| `data.valid_workers` | 3 |
| `data.pin_memory` | True |
| `model._target_` | src.models.BasicUnet |
| `model.in_channels` | 3 |
| `model.num_classes` | 1 |
| `model.base_channels` | 32 |
| `loss._target_` | torch.nn.BCEWithLogitsLoss |
| `optimizer._target_` | torch.optim.Adam |
| `optimizer._partial_` | True |
| `optimizer.lr` | 0.0001 |
| `scheduler` | None |
| `logger._target_` | lightning.pytorch.loggers.TensorBoardLogger |
| `logger.save_dir` | ${hydra:runtime.output_dir} |
| `logger.name` | tensorboard |
| `logger.version` |  |
| `trainer.max_epochs` | 30 |
| `trainer.accelerator` | cuda |
| `trainer.devices` | 1 |
| `trainer.precision` | 32-true |
| `trainer.log_every_n_steps` | 1 |
| `trainer.accumulate_grad_batches` | 1 |
| `trainer.default_root_dir` | ${hydra:runtime.output_dir} |
| `trainer.num_nodes` | 2 |
| `trainer.strategy` | ddp |

</details>

### Результаты

| Эпоха | Train loss | Valid loss | Train IoU | Valid IoU |
| --- | --- | --- | --- | --- |
| 30 | 0.06550* | 0.24351* | 0.93655 | 0.85647 |

### Описание и выводы

<!-- notes:start -->
Цель: —

Изменения: —

Вывод:
Главная проблема в переобучении.
После 10 эпохи модель запоминает train, но почти не улучшает обобщение.

К этому может приводить:
- Отсутствие аугментаций
- Submission работает в масштабе 512x512, когда обучение 256x256
- Сейчас IoU включает фон, то есть MeanIoU усредняет классы фон и животное
- BatchNorm получает всего 4 изображения на GPU, а у меня DDP и статистики скорее всего не шерятся

На изображениях модель выделяет животное, но местами захватывает фон и ошибается в конутер.

<!-- notes:end -->

TensorBoard: `slurm_8518584/tensorboard` (при `--logdir outputs`).

Конфиг: [outputs/slurm_8518584/.hydra/config.yaml](outputs/slurm_8518584/.hydra/config.yaml).

Чекпойнты: `outputs/slurm_8518584/checkpoints`. Записано эпох: 30/30.
<!-- /experiment -->
