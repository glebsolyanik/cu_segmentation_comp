"""Generate experiments.md from Hydra configs and TensorBoard; preserve manual notes."""

import argparse
import json
import re
from bisect import bisect_left
from pathlib import Path
from statistics import fmean

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def config_rows(config: dict, prefix: str = "") -> list[str]:
    rows = []
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(config_rows(value, name))
        else:
            rows.append(f"| `{name}` | {cell(value)} |")
    return rows


def read_scalars(directory: Path) -> dict:
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    return {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}


def value_at(scalars: dict, tag: str, step: int) -> str:
    points = [point for point in scalars.get(tag, []) if point.step == step]
    return f"{points[-1].value:.5f}" if points else "—"


def loss_at(scalars: dict, stage: str, step: int) -> str:
    value = value_at(scalars, f"{stage}_loss_epoch", step)
    if value != "—":
        return value

    epochs = sorted(scalars.get("epoch", []), key=lambda point: point.wall_time)
    target_epochs = [point.value for point in epochs if point.step == step]
    if not target_epochs:
        return "—"
    target_epoch = target_epochs[-1]
    times = [point.wall_time for point in epochs]
    values = []
    # Lightning writes the epoch tag after the loss scalars. Validation batch
    # steps have their own counter, so match by event time rather than step.
    for point in scalars.get(f"{stage}_loss", []):
        index = bisect_left(times, point.wall_time)
        if index < len(epochs) and epochs[index].value == target_epoch:
            values.append(point.value)
    return f"{fmean(values):.5f}*" if values else "—"


def manual_field(notes: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:[ \t]*(.*)(?:\n([^\n]+))?", notes, re.MULTILINE)
    if not match:
        return "—"
    return match.group(1).strip() or (match.group(2) or "—").strip()


def prepare_notes(notes: str | None) -> str:
    if notes is None:
        return "Цель: —\n\nИзменения: —\n\nНаблюдения: —\n\nВывод: —\n\nРезультат соревнования: —"
    # Keep existing prose verbatim, including multiline conclusions.
    missing = [f"{field}: —" for field in ("Цель", "Изменения")
               if not re.search(rf"^{field}:", notes, re.MULTILINE)]
    return "\n\n".join([*missing, notes])


def short_config(cfg: dict) -> list[str]:
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    optimizer = cfg.get("optimizer", {})
    trainer = cfg.get("trainer", {})
    rows = {
        "Модель": model,
        "Размер изображения": data.get("image_size", "—"),
        "Batch size на GPU": data.get("batch_size", "—"),
        "Узлы / GPU на узел": f"{trainer.get('num_nodes', 1)} / {trainer.get('devices', '—')}",
        "Накопление градиентов": trainer.get("accumulate_grad_batches", 1),
        "Optimizer": optimizer,
        "Loss": cfg.get("loss", "—"),
        "Scheduler": cfg.get("scheduler"),
        "Precision": trainer.get("precision", "—"),
        "Seed": cfg.get("seed", "—"),
    }
    def display(value):
        if isinstance(value, dict):
            target = value.get("_target_", "").rsplit(".", 1)[-1]
            params = ", ".join(f"{key}={item}" for key, item in value.items() if not key.startswith("_"))
            return f"{target}({params})"
        return "нет" if value is None else value
    return [f"| {key} | {cell(display(value))} |" for key, value in rows.items()]


def make_entry(run: Path, output: Path, old: dict | None) -> tuple[dict, str]:
    cfg = OmegaConf.to_container(OmegaConf.load(run / ".hydra/config.yaml"), resolve=False)
    event_dirs = sorted({p.parent for p in run.rglob("*tfevents*")})
    if len(event_dirs) > 1:
        raise ValueError(f"Multiple TensorBoard directories in {run}; use separate Hydra run directories")
    logdir = event_dirs[0] if event_dirs else None
    scalars = read_scalars(logdir) if logdir else {}
    valid = scalars.get("valid_iou", [])
    best = max(valid, key=lambda point: point.value) if valid else None
    epoch_points = {point.step: point.value for point in scalars.get("epoch", [])}
    best_epoch = str(int(epoch_points[best.step]) + 1) if best and best.step in epoch_points else "—"
    last_epoch = str(int(max(epoch_points.values())) + 1) if epoch_points else "—"
    tb_name = str(logdir.relative_to(output)) if logdir else "—"
    row = {
        "id": run.relative_to(output).as_posix(),
        "name": cfg.get("name", run.name),
        "tensorboard": tb_name,
        "epochs": f"{last_epoch}/{cfg.get('trainer', {}).get('max_epochs', '—')}",
        "best_epoch": best_epoch,
        "train_loss": loss_at(scalars, "train", best.step) if best else "—",
        "valid_loss": loss_at(scalars, "valid", best.step) if best else "—",
        "train_iou": value_at(scalars, "train_iou", best.step) if best else "—",
        "valid_iou": f"{best.value:.5f}" if best else "—",
    }
    notes = prepare_notes(old["notes"] if old else None)
    section = (
        f"<a id=\"{row['id']}\"></a>\n\n"
        f"## {row['name']} · {row['id']}\n\n"
        "### Конфигурация\n\n| Параметр | Значение |\n| --- | --- |\n"
        + "\n".join(short_config(cfg))
        + "\n\n<details>\n<summary>Полная конфигурация</summary>\n\n"
        + "| Параметр | Значение |\n| --- | --- |\n"
        + "\n".join(config_rows(cfg))
        + "\n\n</details>\n\n### Результаты\n\n"
        + "| Эпоха | Train loss | Valid loss | Train IoU | Valid IoU |\n"
        + "| --- | --- | --- | --- | --- |\n"
        + "| " + " | ".join(row[key] for key in ("best_epoch", "train_loss", "valid_loss", "train_iou", "valid_iou")) + " |\n\n"
        + "### Описание и выводы\n\n"
        + "<!-- notes:start -->\n" + notes + "\n<!-- notes:end -->\n"
        + f"\nTensorBoard: `{tb_name}` (при `--logdir {output}`).\n\n"
        + f"Конфиг: [{run / '.hydra/config.yaml'}]({run / '.hydra/config.yaml'}).\n\n"
        + f"Чекпойнты: `{run / 'checkpoints'}`. Записано эпох: {row['epochs']}.\n"
    )
    return row, section


def update_journal(output: Path, journal: Path) -> None:
    existing = journal.read_text() if journal.exists() else ""
    entries = {}
    pattern = r"<!-- experiment:(.*?) -->\n(.*?)<!-- /experiment -->"
    for match in re.finditer(pattern, existing, re.DOTALL):
        row = json.loads(match.group(1))
        section = match.group(2).rstrip() + "\n"
        notes = re.search(r"<!-- notes:start -->\n(.*?)\n<!-- notes:end -->", section, re.DOTALL)
        if not notes:
            raise ValueError(f"Missing manual notes markers for {row['id']}")
        entries[row["id"]] = {"row": row, "section": section, "notes": notes.group(1)}
    if existing and not entries:
        raise ValueError(f"{journal} has no generated entries; move the old journal before generating")

    for config in sorted(output.rglob(".hydra/config.yaml")):
        run = config.parent.parent
        key = run.relative_to(output).as_posix()
        row, section = make_entry(run, output, entries.get(key))
        notes = re.search(r"<!-- notes:start -->\n(.*?)\n<!-- notes:end -->", section, re.DOTALL).group(1)
        entries[key] = {"row": row, "section": section, "notes": notes}

    lines = [
        "# Эксперименты", "",
        "Обновить: `python scripts/expeiment_to_md.py`. Конфигурация и числа заполняются автоматически.",
        "Новые запуски из outputs добавляются при обновлении. Поля в блоках «Описание и выводы»",
        "редактируются вручную и сохраняются; «Изменения» и первая строка «Вывод» попадают в общую таблицу.", "",
        "Loss и train IoU приведены на эпохе лучшего valid IoU. Эпохи считаются с 1.",
        "`—` означает, что значения нет в логах. `*` — среднее записанных batch loss за эпоху,",
        "если epoch loss отсутствует; без взвешивания по размеру батча и дополнительной синхронизации GPU.", "",
        "| Эксперимент | Что изменили | Valid IoU ↑ | Valid loss ↓ | Эпоха | Вывод |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries.values():
        row = entry["row"]
        values = [f"[{row['name']} · {row['id']}](#{row['id']})",
                  manual_field(entry["notes"], "Изменения"), row["valid_iou"], row["valid_loss"],
                  row["best_epoch"], manual_field(entry["notes"], "Вывод")]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    for entry in entries.values():
        lines.extend(["", f"<!-- experiment:{json.dumps(entry['row'], ensure_ascii=False)} -->", entry["section"].rstrip(), "<!-- /experiment -->"])
    journal.write_text("\n".join(lines) + "\n")
    print(f"Updated {journal}: {len(entries)} runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--journal", type=Path, default=Path("experiments.md"))
    args = parser.parse_args()
    if not args.outputs.is_dir():
        parser.error(f"Outputs directory does not exist: {args.outputs}")
    update_journal(args.outputs, args.journal)


if __name__ == "__main__":
    main()
