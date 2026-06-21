from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unilab.logging.common import BaseTrainingLogger, _fmt_number, _load_wandb


class OnPolicyLogger(BaseTrainingLogger):
    """Rich logger for on-policy RL (PPO, A2C, etc)."""

    def __init__(
        self,
        algo_name: str = "PPO",
        max_iterations: int = 1500,
        num_envs: int = 4096,
        num_steps: int = 24,
        env_name: str = "",
        log_dir: str = "",
        log_backend: str = "tensorboard",
        wandb_project: str = "unilab",
        wandb_entity: str | None = None,
        wandb_name: str = "",
        wandb_group: str | None = None,
        wandb_job_type: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_notes: str | None = None,
        backend_schema: str = "unilab",
    ):
        if backend_schema not in {"unilab", "rsl_rl"}:
            raise ValueError("backend_schema must be 'unilab' or 'rsl_rl'")
        super().__init__(
            algo_name=algo_name,
            max_iterations=max_iterations,
            num_envs=num_envs,
            env_name=env_name,
            log_dir=log_dir,
            log_backend=log_backend,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_name=wandb_name,
            wandb_group=wandb_group,
            wandb_job_type=wandb_job_type,
            wandb_tags=wandb_tags,
            wandb_notes=wandb_notes,
            tensorboard_subdir="tb",
        )
        self.num_steps = num_steps
        self.backend_schema = backend_schema

    def start(self, *, status: str = ""):
        super().start(status=status)

    def finish(self, *, title: str = "Training Summary", extra_summary: str = ""):
        super().finish(title=title, extra_summary=extra_summary)

    def log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None = None,
        reward: float | None = None,
        reward_components: dict[str, float] | None = None,
        collect_time: float = 0.0,
        train_time: float = 0.0,
        iteration_time: float | None = None,
        collect_label: str = "Collect",
    ):
        self._iteration = iteration
        self._collect_time = collect_time
        self._train_time = train_time
        self._iteration_time = iteration_time
        self._collect_label = collect_label

        if metrics:
            self._latest_metrics.update(metrics)
        if reward is not None:
            self._reward_history.append(reward)
        if reward_components:
            self._latest_reward_components = reward_components

        self._refresh()
        self._backend_log_step(iteration, metrics, reward, reward_components)

    def _backend_log_step(
        self,
        iteration: int,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_components: dict[str, float] | None,
    ):
        if self._tb_writer:
            w = self._tb_writer
            for key, value in self._scalar_payload(metrics, reward, reward_components).items():
                w.add_scalar(key, value, iteration)

        if self._wandb_run:
            wandb = _load_wandb()
            if wandb is None:
                return

            log_dict: dict[str, Any] = {"iteration": iteration}
            log_dict.update(self._scalar_payload(metrics, reward, reward_components))
            wandb.log(log_dict, step=iteration)

    def _metric_key(self, key: str) -> str:
        if self.backend_schema == "rsl_rl":
            if key.startswith("async/"):
                suffix = key.removeprefix("async/")
                return f"Async/{suffix}"
            if key == "learning_rate":
                return "Loss/learning_rate"
            return f"Loss/{key}"
        return f"train/{key}"

    def _scalar_payload(
        self,
        metrics: dict[str, float] | None,
        reward: float | None,
        reward_components: dict[str, float] | None,
    ) -> dict[str, float]:
        payload: dict[str, float] = {}
        if metrics:
            for key, value in metrics.items():
                payload[self._metric_key(key)] = value

        if self.backend_schema == "rsl_rl":
            if reward is not None:
                payload["Train/mean_reward"] = reward
            if self._mean_ep_length > 0:
                payload["Train/mean_episode_length"] = self._mean_ep_length
            payload["Perf/collection_time"] = self._collect_time
            payload["Perf/learning_time"] = self._train_time
            steps_per_sec = self._steps_per_second()
            if steps_per_sec is not None:
                payload["Perf/total_fps"] = float(int(steps_per_sec))
            if self._iteration_time is not None:
                payload["Perf/iteration_time"] = self._iteration_time
        else:
            if reward is not None:
                payload["reward/mean"] = reward
            if self._mean_ep_length > 0:
                payload["episode/length"] = self._mean_ep_length
            payload["perf/collect_time_ms"] = self._collect_time * 1000
            payload["perf/train_time_ms"] = self._train_time * 1000
            if self._iteration_time is not None:
                payload["perf/iteration_time_ms"] = self._iteration_time * 1000
            steps_per_sec = self._steps_per_second()
            if steps_per_sec is not None:
                payload["perf/steps_per_sec"] = steps_per_sec

        if reward_components:
            for key, value in reward_components.items():
                payload[key if key.startswith("reward/") else f"reward/{key}"] = value
        return payload

    def _build_display(self) -> Panel:
        header = self._build_compact_header(include_status=True)
        left = self._build_metrics_table()
        right = self._build_reward_table()
        bottom = self._build_timing_table()
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(width=2)
        grid.add_column(ratio=1)
        grid.add_row(left, "", right)
        return Panel(
            Group(header, Text(""), grid, Text(""), bottom),
            title=(
                "[bold] 🚀 UniLab On-Policy Training [/]"
                if self._unicode_console
                else "[bold] UniLab On-Policy Training [/]"
            ),
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_metrics_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold cyan",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Losses & Metrics", style="white", ratio=2)
        table.add_column("Value", style="yellow", justify="right", ratio=1)

        if not self._latest_metrics:
            table.add_row("[dim]Waiting for data...[/]", "")
        else:
            loss_keys = sorted([key for key in self._latest_metrics if "loss" in key.lower()])
            other_keys = sorted([key for key in self._latest_metrics if "loss" not in key.lower()])
            for key in loss_keys:
                value = self._latest_metrics[key]
                style = "red" if value > 10 else "yellow"
                table.add_row(key.replace("_", " ").title(), f"[{style}]{_fmt_number(value)}[/]")
            for key in other_keys:
                value = self._latest_metrics[key]
                table.add_row(f"  {key.replace('_', ' ').title()}", _fmt_number(value))

        return table

    def _build_reward_table(self) -> Table:
        return self._build_reward_table_common(
            wait_message="[dim]Waiting for data...[/]",
            include_ep_length=False,
        )

    def _iteration_duration(self) -> float:
        return self._iteration_time or (self._collect_time + self._train_time)

    def _steps_per_second(self) -> float | None:
        iter_time = self._iteration_duration()
        if iter_time <= 0:
            return None
        return self.num_envs * self.num_steps / iter_time

    def _build_timing_table(self) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            show_edge=False,
            header_style="bold blue",
            expand=True,
            pad_edge=False,
        )
        table.add_column("Learner", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)
        table.add_column("Collector", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)
        table.add_column("System", style="white", ratio=2, no_wrap=True)
        table.add_column("Value", style="yellow", justify="right", ratio=1, no_wrap=True)

        iter_time = self._iteration_duration()
        steps_per_sec = self._steps_per_second()
        fps = int(steps_per_sec) if steps_per_sec is not None else 0

        learner_items = [
            ("Train", f"{self._train_time * 1000:.1f}ms"),
            ("Iter Time", f"{iter_time * 1000:.1f}ms"),
        ]
        hidden_collect = self._latest_metrics.get("async/hidden_collect_time")
        if hidden_collect is not None:
            learner_items.append(("Hidden Collect", f"{hidden_collect * 1000:.1f}ms"))
        collector_items = [
            (self._collect_label, f"{self._collect_time * 1000:.1f}ms"),
        ]
        rollout_collect = self._latest_metrics.get("async/rollout_collect_time")
        if rollout_collect is not None:
            collector_items.append(("Rollout Collect", f"{rollout_collect * 1000:.1f}ms"))
        system_items = [
            ("Envs", f"{self.num_envs:,}"),
            ("Steps/s", f"{fps:,}"),
        ]

        row_count = max(len(learner_items), len(collector_items), len(system_items))
        for index in range(row_count):
            row: list[str] = []
            for items in (learner_items, collector_items, system_items):
                if index < len(items):
                    row.extend(items[index])
                else:
                    row.extend(["", ""])
            table.add_row(*row)

        return table

    def _build_compact_header(
        self,
        *,
        include_status: bool,
        extra_fields: list[tuple[str, str]] | None = None,
    ) -> Text:
        header_extra_fields: list[tuple[str, str]] = []
        steps_per_second = self._steps_per_second()
        if steps_per_second is not None:
            header_extra_fields.append((f"Steps/s {steps_per_second:,.0f}", "bold green"))
        if extra_fields:
            header_extra_fields.extend(extra_fields)
        return super()._build_compact_header(
            include_status=include_status,
            extra_fields=header_extra_fields,
        )
