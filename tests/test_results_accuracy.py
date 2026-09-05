import csv
import io
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from results.gerar_resultados import (
    CSIM_SPECS,
    Metric,
    RunResult,
    STAGE_ORDER,
    STATE_LOG,
    discover_latest_runs,
    paired_csim_vivado_rows,
    render_csv,
    render_report,
)


def write_summary(
    path: Path,
    run_id: str,
    accuracy: str | None = None,
    scope_to: str | None = None,
    successful_stages: list[str] | None = None,
) -> None:
    accuracy_section = ""
    if accuracy is not None:
        accuracy_section = (
            "\n## Acurácia CSim\n\n"
            f"- Acurácia final: `{accuracy}`\n"
            f"- Log: `{path.parent.parent / 'logs' / 'csim.log'}`\n"
        )
    scope_section = ""
    if scope_to is not None:
        scope_section = (
            "\n## Escopo da execução\n\n"
            "- Etapa inicial: `prepare`\n"
            f"- Etapa final: `{scope_to}`\n"
        )
    stage_lines = "".join(
        f"- `{stage}`: success\n"
        for stage in (successful_stages or ["prepare"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Resumo da execução NeuroHLS\n\n"
        f"- Run: `{run_id}`\n"
        "- Projeto: `/tmp/event_driven_zero_pwl`\n"
        "- Top: `snn_to_hls`\n"
        "- FPGA: `xcu250-figd2104-2L-e`\n"
        "- Clock: `150 MHz` (`6.66667 ns`)\n"
        + scope_section
        + "\n## Etapas\n\n"
        + stage_lines
        + accuracy_section,
        encoding="utf-8",
    )


class ResultsAccuracyTests(unittest.TestCase):
    @staticmethod
    def scoped_result(
        network: str,
        metrics: dict[str, Decimal | int],
    ) -> RunResult:
        result = RunResult(
            network=network,
            backend=(
                "time-driven" if network.startswith("time_driven") else "event-driven"
            ),
            run_id=f"20260803T210000Z-{network}",
            run_dir=Path("/tmp") / network,
            summary_path=Path("/tmp") / network / "reports" / "summary.md",
            summary_exists=True,
            scope_to_stage="vivado-synth",
            stages={
                stage: "success"
                for stage in STAGE_ORDER[: STAGE_ORDER.index("vivado-synth") + 1]
            },
        )
        result.metrics = {
            key: Metric(value=value, raw=str(value))
            for key, value in metrics.items()
        }
        return result

    def test_reads_accuracy_from_selected_summary_and_renders_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20260803T190000Z-current"
            run = root / "sim" / "runs" / "event_driven_zero_pwl" / run_id
            summary = run / "reports" / "summary.md"
            write_summary(summary, run_id, "88,57%")

            results = discover_latest_runs(root / "sim" / "runs")
            metric = results[0].metrics[CSIM_SPECS[0].key]
            self.assertEqual(Decimal("88.57"), metric.value)

            report = render_report(
                results,
                root / "results" / "comparativo.md",
                root / "results" / "metricas.csv",
                root,
            )
            self.assertIn("## Acurácia CSim", report)
            self.assertIn("88,57%", report)
            csv_text = render_csv(results, root)
            rows = list(csv.DictReader(io.StringIO(csv_text)))
            accuracy_row = next(row for row in rows if row["grupo"] == "CSim")
            self.assertEqual("Acurácia final", accuracy_row["metrica"])
            self.assertEqual("88.57", accuracy_row["valor_normalizado"])
            self.assertEqual("88,57%", accuracy_row["valor_original"])
            self.assertEqual("%", accuracy_row["unidade"])
            self.assertEqual("reportado", accuracy_row["estado"])

    def test_uses_only_latest_run_log_and_keeps_missing_as_nd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            network = root / "sim" / "runs" / "event_driven_zero_pwl"
            old_id = "20260803T180000Z-old"
            old_run = network / old_id
            write_summary(old_run / "reports" / "summary.md", old_id, "99,00%")

            latest_id = "20260803T190000Z-latest"
            latest_run = network / latest_id
            write_summary(latest_run / "reports" / "summary.md", latest_id)
            (latest_run / "logs").mkdir(parents=True)
            (latest_run / "logs" / "csim.log").write_text(
                " *** Final Acc: 92.86%\n",
                encoding="utf-8",
            )

            result = discover_latest_runs(root / "sim" / "runs")[0]
            metric = result.metrics["csim.accuracy_pct"]
            self.assertEqual(latest_id, result.run_id)
            self.assertEqual(Decimal("92.86"), metric.value)
            self.assertEqual(STATE_LOG, metric.state)
            self.assertEqual(latest_run / "logs" / "csim.log", metric.source)

            (latest_run / "logs" / "csim.log").unlink()
            result = discover_latest_runs(root / "sim" / "runs")[0]
            self.assertNotIn("csim.accuracy_pct", result.metrics)

    def test_network_filter_does_not_change_latest_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "sim" / "runs"
            for network, run_id, accuracy in (
                ("event_driven_zero_pwl", "20260803T190000Z-event", "88,57%"),
                ("time_driven_zero", "20260803T190001Z-time", "91,43%"),
            ):
                write_summary(
                    runs / network / run_id / "reports" / "summary.md",
                    run_id,
                    accuracy,
                )

            selected = discover_latest_runs(
                runs,
                {"event_driven_zero_pwl"},
            )
            self.assertEqual(["event_driven_zero_pwl"], [item.network for item in selected])
            self.assertEqual("20260803T190000Z-event", selected[0].run_id)

    def test_vivado_scope_is_complete_and_omits_post_synth_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "sim" / "runs"
            run_id = "20260803T200000Z-vivado"
            stages = STAGE_ORDER[: STAGE_ORDER.index("vivado-synth") + 1]
            write_summary(
                runs
                / "event_driven_zero_pwl"
                / run_id
                / "reports"
                / "summary.md",
                run_id,
                "85,71%",
                scope_to="vivado-synth",
                successful_stages=stages,
            )

            results = discover_latest_runs(runs)
            self.assertFalse(results[0].incomplete)
            self.assertEqual("vivado-synth", results[0].required_stages[-1])

            report = render_report(
                results,
                root / "results" / "comparativo.md",
                root / "results" / "metricas.csv",
                root,
            )
            self.assertIn("escopo até `vivado-synth` concluído", report)
            self.assertIn("## Acurácia CSim", report)
            self.assertIn("## Estimativa de recursos HLS", report)
            self.assertIn("## Uso de recursos pós-síntese", report)
            self.assertIn("## Encerramento intencional do fluxo", report)
            self.assertNotIn("## Atividade SAIF", report)
            self.assertNotIn("## Potência", report)
            self.assertNotIn("†", report.split("## Runs selecionados", 1)[1].split("## Plataforma", 1)[0])

            rows = list(csv.DictReader(io.StringIO(render_csv(results, root))))
            groups = {row["grupo"] for row in rows}
            self.assertIn("CSim", groups)
            self.assertIn("HLS", groups)
            self.assertIn("Vivado OOC", groups)
            self.assertNotIn("SAIF", groups)
            self.assertNotIn("Potência", groups)

    def test_hls_scope_footer_does_not_claim_vivado_synthesis(self):
        result = self.scoped_result("event_driven_zero_active_list", {})
        result.scope_to_stage = "hls-synth"
        result.stages = {
            stage: "success"
            for stage in STAGE_ORDER[: STAGE_ORDER.index("hls-synth") + 1]
        }

        report = render_report(
            [result],
            Path("/tmp/results/comparativo.md"),
            Path("/tmp/results/metricas.csv"),
            Path("/tmp"),
        )
        self.assertIn("avaliados somente até `hls-synth`", report)
        self.assertNotIn("avaliados somente até `vivado-synth`", report)

    def test_to_stage_override_supports_legacy_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "sim" / "runs"
            run_id = "20260803T200001Z-legacy"
            stages = STAGE_ORDER[: STAGE_ORDER.index("vivado-synth") + 1]
            write_summary(
                runs
                / "event_driven_zero_pwl"
                / run_id
                / "reports"
                / "summary.md",
                run_id,
                "85,71%",
                successful_stages=stages,
            )

            legacy = discover_latest_runs(runs)[0]
            self.assertTrue(legacy.incomplete)
            scoped = discover_latest_runs(
                runs,
                to_stage_override="vivado-synth",
            )[0]
            self.assertFalse(scoped.incomplete)

    def test_vivado_scope_renders_paired_accuracy_and_resource_overhead(self):
        time_result = self.scoped_result(
            "time_driven_zero",
            {
                "csim.accuracy_pct": Decimal("80"),
                "vivado.LUT.used": 100,
                "vivado.FF.used": 200,
                "vivado.BRAM.used": Decimal("2"),
                "vivado.DSP.used": 10,
            },
        )
        event_result = self.scoped_result(
            "event_driven_zero_pwl",
            {
                "csim.accuracy_pct": Decimal("85.5"),
                "vivado.LUT.used": 150,
                "vivado.FF.used": 180,
                "vivado.BRAM.used": Decimal("3"),
                "vivado.DSP.used": 20,
            },
        )

        rows = paired_csim_vivado_rows([time_result, event_result])
        self.assertEqual(5, len(rows))
        self.assertEqual("5,50 p.p.", rows[0][-1])
        self.assertEqual("50,00%", rows[1][-1])
        self.assertEqual("-10,00%", rows[2][-1])
        self.assertEqual("50,00%", rows[3][-1])
        self.assertEqual("100,00%", rows[4][-1])

        report = render_report(
            [time_result, event_result],
            Path("/tmp/results/comparativo.md"),
            Path("/tmp/results/metricas.csv"),
            Path("/tmp"),
        )
        paired_heading = "## Comparação pareada time-driven × event-driven"
        self.assertIn(paired_heading, report)
        self.assertLess(
            report.index(paired_heading),
            report.index("## Encerramento intencional do fluxo"),
        )
        self.assertIn("Acurácia CSim", report)
        self.assertIn("Vivado OOC — LUT", report)

    def test_vivado_pair_omits_missing_values_and_zero_denominators(self):
        time_result = self.scoped_result(
            "time_driven_subtract",
            {
                "vivado.LUT.used": 0,
                "vivado.BRAM.used": Decimal("2"),
            },
        )
        event_result = self.scoped_result(
            "event_driven_subtract_ts_efa",
            {
                "vivado.LUT.used": 100,
                "vivado.BRAM.used": Decimal("0"),
                "vivado.DSP.used": 20,
            },
        )

        rows = paired_csim_vivado_rows([time_result, event_result])
        self.assertEqual(1, len(rows))
        self.assertEqual("Vivado OOC — BRAM", rows[0][2])
        self.assertEqual("-100,00%", rows[0][-1])
        self.assertNotIn("N/D", str(rows))


if __name__ == "__main__":
    unittest.main()
