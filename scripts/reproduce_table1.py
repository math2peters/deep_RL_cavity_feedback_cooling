from pathlib import Path

import pandas as pd
import yaml

from combined_performance_metric_calculator import combine_metrics, load_and_process_data, save_combined_results
from detuning_performance_metric_calculator import calculate_performance_metric as calculate_detuning_metrics, save_detailed_results as save_detuning_results
from photon_number_performance_metric_calculator import calculate_performance_metric as calculate_photon_metrics, save_detailed_results as save_photon_results


def load_config():
    script_dir = Path(__file__).resolve().parent
    package_root = script_dir.parent
    with open(script_dir / "params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return package_root, params["fig4"], params["table1"], params["output_dir"]


def build_dataframe(values):
    rows = []
    labels = ["Survival", "Cooling", "Energy", "Overall"]
    groups = [
        ("Detuning", values["detuning"]),
        ("Photon Count", values["photon_count"]),
        ("Combined", values["combined"]),
    ]
    for group, group_values in groups:
        for model, metrics in group_values.items():
            row = {"Metric Group": group, "Model": model}
            for label, value in zip(labels, metrics):
                row[label] = value
            rows.append(row)
    return pd.DataFrame(rows)


def format_latex(df):
    order = ["MLP (Sim.)", "MLP (Expt.)", "Differentiator"]
    groups = ["Detuning", "Photon Count", "Combined"]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Metric Group} & \textbf{Model} & \textbf{Survival} & \textbf{Cooling} & \textbf{Energy} & \textbf{Overall} \\",
        r"\midrule",
    ]
    for group_index, group in enumerate(groups):
        group_df = df[df["Metric Group"] == group].set_index("Model")
        for model_index, model in enumerate(order):
            prefix = rf"\multirow{{3}}{{*}}{{\textbf{{{group}}}}} & " if model_index == 0 else " & "
            row = group_df.loc[model]
            lines.append(
                prefix
                + f"{model} & {row['Survival']:.3f} & {row['Cooling']:.3f} & {row['Energy']:.3f} & {row['Overall']:.3f} \\\\",
            )
        if group_index < len(groups) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main():
    package_root, fig4_cfg, table_cfg, output_dir_name = load_config()
    output_dir = package_root / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = package_root / fig4_cfg["data_root"]
    model_folders = fig4_cfg["model_folders"]

    detuning_results = load_and_process_data(data_root, model_folders, "detuning")
    photon_results = load_and_process_data(data_root, model_folders, "photon_number")

    detuning_metrics = calculate_detuning_metrics(detuning_results, param_name="detuning", performance_threshold=0.5)
    photon_metrics = calculate_photon_metrics(photon_results, param_name="photon_number", performance_threshold=0.5)
    combined_metrics = combine_metrics(detuning_metrics, photon_metrics)

    save_detuning_results(detuning_metrics, output_dir / "detuning_performance_metrics.csv")
    save_photon_results(photon_metrics, output_dir / "photon_number_performance_metrics.csv")
    save_combined_results(combined_metrics, output_dir / "combined_performance_metrics.csv")

    actual_df = pd.DataFrame(
        [
            {
                "Metric Group": "Detuning",
                "Model": data["label"],
                "Survival": data["metric_scores"]["fraction_trapped"],
                "Cooling": data["metric_scores"]["cooling_timescale"],
                "Energy": data["metric_scores"]["20_step_energy"],
                "Overall": data["overall_metric"],
            }
            for _, data in detuning_metrics.items()
        ]
        + [
            {
                "Metric Group": "Photon Count",
                "Model": data["label"],
                "Survival": data["metric_scores"]["fraction_trapped"],
                "Cooling": data["metric_scores"]["cooling_timescale"],
                "Energy": data["metric_scores"]["20_step_energy"],
                "Overall": data["overall_metric"],
            }
            for _, data in photon_metrics.items()
        ]
        + [
            {
                "Metric Group": "Combined",
                "Model": data["label"],
                "Survival": data["metric_scores"]["fraction_trapped"],
                "Cooling": data["metric_scores"]["cooling_timescale"],
                "Energy": data["metric_scores"]["20_step_energy"],
                "Overall": data["overall_metric"],
            }
            for _, data in combined_metrics.items()
        ]
    )
    actual_df = actual_df.sort_values(["Metric Group", "Model"]).reset_index(drop=True)

    numeric_cols = ["Survival", "Cooling", "Energy", "Overall"]


    actual_df.to_csv(output_dir / table_cfg["output_csv"], index=False)
    with open(output_dir / table_cfg["output_tex"], "w", encoding="utf-8") as handle:
        handle.write(format_latex(actual_df))


if __name__ == "__main__":
    main()
