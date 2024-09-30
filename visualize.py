import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TIFA Benchmark Results")
    parser.add_argument(
        "--tifa-benchmark-result-directory",
        type=str,
        required=True,
        help="Path to the directory containing tifa benchmark result JSON files.",
    )
    parser.add_argument(
        "--image-save-directory",
        type=str,
        required=True,
        help="Path to the directory where result images will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_dir = args.result_directory
    image_save_dir = args.image_save_directory

    # Ensure the save directory exists, if not create it
    os.makedirs(image_save_dir, exist_ok=True)

    # Initialize empty dictionaries to store data from JSON files
    tifa_average_data = {}
    accuracy_by_type_data = {}

    # Iterate over each JSON file in the directory
    for filename in os.listdir(json_dir):
        file_path = os.path.join(json_dir, filename)
        if os.path.isfile(file_path):  # Check if it is a file
            with open(file_path) as f:
                data = json.load(f)
                model_name = filename.replace(
                    ".json", ""
                )  # Use filename without .json as model name
                tifa_average_data[model_name] = round(data["tifa_average"], 4)
                accuracy_by_type_data[model_name] = {
                    k: round(v, 4) for k, v in data["accuracy_by_type"].items()
                }

    # Create a DataFrame for accuracy_by_type
    df_accuracy = pd.DataFrame(accuracy_by_type_data)

    # Add tifa_average as the last row in the DataFrame
    df_accuracy.loc["tifa_average"] = pd.Series(tifa_average_data)

    # Transpose the DataFrame so that models are columns and metrics are rows
    df_accuracy = df_accuracy.transpose()

    # Plotting the DataFrame as a table
    fig, ax = plt.subplots(
        figsize=(25, len(df_accuracy) * 0.5)
    )  # Adjust the height based on number of models
    ax.axis("tight")
    ax.axis("off")

    # Plotting the table
    table = ax.table(
        cellText=df_accuracy.values,
        colLabels=df_accuracy.columns,
        rowLabels=df_accuracy.index,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # Scale to make rows taller

    plt.title("Accuracy Metrics by Model")

    # Save the table as an image file
    chart_path = os.path.join(image_save_dir, "result_chart.png")
    plt.savefig(chart_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Plotting the accuracy metrics as a graph
    fig, ax = plt.subplots(figsize=(20, 6))

    # Generate colors for each model
    colors = plt.cm.get_cmap("tab10", len(df_accuracy))

    # Plot each model's accuracy_by_type
    for i, (model, row) in enumerate(df_accuracy.iterrows()):
        ax.plot(
            row.index,
            row.values,
            marker="o",
            linestyle="-",
            label=model,
            color=colors(i),
        )

    # Adding legend with model names and colors on the right side
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    # Setting labels and title
    ax.set_xlabel("Accuracy by Type")
    ax.set_ylabel("Score")
    plt.title("Accuracy Metrics by Type and Model")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Save the graph as an image file
    graph_path = os.path.join(image_save_dir, "result_graph.png")
    plt.savefig(graph_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Charts saved at: {chart_path} and {graph_path}")


if __name__ == "__main__":
    main()
