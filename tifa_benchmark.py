import argparse
from tifascore.question_gen import get_question_and_answers
from tifascore.question_filter import filter_question_and_answers
from tifascore.unifiedqa import UnifiedQAModel
from tifascore.tifa_score import tifa_score_benchmark, tifa_score_single
from tifascore.vqa_models import VQAModel
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

JSON_FILE_PATH = os.path.join(os.getcwd(), "tifa_v1.0", "tifa_v1.0_text_inputs.json")
QA_FILE_PATH = os.path.join(os.getcwd(), "tifa_v1.", "tifa_v1.0_question_answers.json")


# 입력받는 경로
def parse_args():
    parser = argparse.ArgumentParser(description="TIFA Benchmark Script")

    parser.add_argument(
        "--image-output-path",
        type=str,
        required=True,
        help="Path to where images are stored (model folders).",
    )
    parser.add_argument(
        "--json-output-path",
        type=str,
        required=True,
        help="Path to save generated JSON files.",
    )
    parser.add_argument(
        "--benchmark-result-path",
        type=str,
        required=True,
        help="Path to save benchmark results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_output_base_path = args.image_output_path
    json_output_base_path = args.json_output_path
    benchmark_result_base_path = args.benchmark_result_path

    os.makedirs(json_output_base_path, exist_ok=True)
    os.makedirs(benchmark_result_base_path, exist_ok=True)

    model_names = [
        f
        for f in os.listdir(image_output_base_path)
        if os.path.isdir(os.path.join(image_output_base_path, f))
    ]

    with open(JSON_FILE_PATH, "r") as f:
        data = json.load(f)

    for model_name in model_names:
        image_output_path = os.path.join(image_output_base_path, model_name)
        json_output_file = os.path.join(json_output_base_path, f"{model_name}.json")

        if not os.path.exists(json_output_file):
            new_json_data = {}

            for item in data:
                image_id = item["id"]
                new_json_data[image_id] = os.path.join(
                    image_output_path, f"{image_id}.png"
                )

            with open(json_output_file, "w") as f:
                json.dump(new_json_data, f, indent=4)

            print(f"New JSON file created: {json_output_file}")
        else:
            print(
                f"JSON file already exists for model {model_name}, skipping JSON generation."
            )

        benchmark_result_file = os.path.join(
            benchmark_result_base_path, f"tifa_benchmark_{model_name}_results.json"
        )

        if not os.path.exists(benchmark_result_file):
            results = tifa_score_benchmark(
                "mplug-large", QA_FILE_PATH, json_output_file
            )

            with open(benchmark_result_file, "w") as f:
                json.dump(results, f, indent=4)

            print(
                f"TIFA benchmark completed for {model_name}. Results saved to {benchmark_result_file}"
            )
        else:
            print(
                f"TIFA benchmark results already exist for model {model_name}, skipping benchmark."
            )

    print("Done with all processes: JSON creation and TIFA benchmark.")

    # Visualization
    json_dir = benchmark_result_base_path

    tifa_average_data = {}
    accuracy_by_type_data = {}

    for filename in os.listdir(json_dir):
        if not ("sdxl" in filename or "step" in filename):
            continue
        if filename.endswith("results.json"):
            with open(os.path.join(json_dir, filename)) as f:
                data = json.load(f)
                model_name = filename.replace("tifa_benchmark_", "").replace(
                    "_results.json", ""
                )
                if len(model_name) > 20:
                    model_name = model_name.replace(
                        "SDXL_jector_core_Dtype3_Ptype3_lr4e-7_bs48_Tddp_Fv-param_", ""
                    )
                print(f"Processing file: {filename} with model name: {model_name}")

                tifa_average_data[model_name] = round(data["tifa_average"], 4)
                accuracy_by_type_data[model_name] = {
                    k: round(v, 4) for k, v in data["accuracy_by_type"].items()
                }
        print(f"{filename} done added")

    df_accuracy = pd.DataFrame(accuracy_by_type_data)
    df_accuracy.loc["tifa_average"] = pd.Series(tifa_average_data)
    df_accuracy = df_accuracy.transpose()
    columns_order = ["tifa_average"] + [
        col for col in df_accuracy.columns if col != "tifa_average"
    ]

    df_accuracy = df_accuracy[columns_order]
    df_accuracy = df_accuracy.sort_index()

    # Table Visualization
    fig, ax = plt.subplots(figsize=(25, len(df_accuracy) * 0.5))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df_accuracy.values,
        colLabels=df_accuracy.columns,
        rowLabels=df_accuracy.index,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Accuracy Metrics by Model")
    plt.savefig("chart.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Line Plot Visualization
    fig, ax = plt.subplots(figsize=(20, 8))
    colors = plt.cm.get_cmap("tab20", len(df_accuracy.columns))

    # Plot each category (column) with a different color
    for i, category in enumerate(df_accuracy.columns):
        ax.plot(
            df_accuracy.index,
            df_accuracy[category],
            marker="o",
            linestyle="-",
            label=category,
            color=colors(i),
        )

    ax.set_xlabel("Step (Model Name)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    plt.title("Scores by Category and Step (Model Name)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Add legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.savefig("graph.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
