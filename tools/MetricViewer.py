"""
Streamlit-based interactive metrics explorer for ECHI project.

This script provides a web UI for visualizing, comparing, and exploring
CSV-based evaluation metrics for audio experiments. It supports:
- Loading and parsing experiment result files.
- Visualizing distributions and scatter plots of metrics.
- Comparing metrics across experiments.
- Interactive selection of data points with audio playback for reference and enhanced
segments.

Configuration is handled via Hydra.
"""

import csv
import functools
import glob
import logging
import os
import string

import hydra
import pandas as pd
import parse
import plotly.graph_objects as go
import streamlit as st
from omegaconf import DictConfig

# --- Utility functions ---

SAMPLE_RATE = 16000  # Assuming a sample rate of 16kHz for audio files
DATASPLIT = "dev"  # Default dataset split to use in the reports


def find_files(template_str, overrides=None):
    """Find all files matching a template string with optional overrides."""
    formatter = string.Formatter()
    placeholders = [fname for _, fname, _, _ in formatter.parse(template_str) if fname]

    wildcard_dict = {placeholder: "*" for placeholder in set(placeholders)}
    if overrides:
        for key, value in overrides.items():
            wildcard_dict[key] = value
    wildcard_str = template_str.format(**wildcard_dict)

    files = glob.glob(wildcard_str)
    good_files = []
    parsed_files = []
    for file in files:
        parsed = parse.parse(template_str, file)
        if parsed is None:
            continue
        good_files.append(file)
        parsed_files.append(parsed.named)
        # convert parsed result to a dictionary
    return good_files, parsed_files


@st.cache_data
def load_metrics(filepath):
    """Load metric from CSV."""
    metrics_df = pd.read_csv(filepath, sep=",", quoting=csv.QUOTE_NONE)
    metrics_df["session_id"] = metrics_df["key"].apply(lambda x: x.split(".")[0])
    # e.g. parse names of the form: dev_03.ha.P098.117.24412238_24482318.wav
    metrics_df["start_time"] = (
        metrics_df["key"].apply(lambda x: int(x.split(".")[4].split("_")[0]))
        / SAMPLE_RATE
    )
    metrics_df["end_time"] = (
        metrics_df["key"].apply(lambda x: int(x.split(".")[4].split("_")[1]))
        / SAMPLE_RATE
    )
    metrics_df["duration"] = metrics_df["end_time"] - metrics_df["start_time"]
    return metrics_df


def get_numeric_cols(df, exclude=None):
    """Return a list of numeric columns, optionally excluding some."""
    cols = df.select_dtypes(include="number").columns.tolist()
    if exclude:
        for ex in exclude:
            if ex in cols:
                cols.remove(ex)
    return cols


def make_exp_labels(exp_params: list[dict]):
    """Create a label for an experiment based on its parameters."""
    return [f"{p['exp_name']} {p['device']} {p['segment_type']}" for p in exp_params]


# ---- Plotting functions ----


def plot_distribution(df, metric, plot_fn, session_col="session_id", session=None):
    """Return a Plotly distribution plot for the given metric and session."""
    fig = go.Figure()
    if session is None or session == "All":
        for sid in sorted(df[session_col].unique()):
            sdf = df[df[session_col] == sid]
            fig.add_trace(plot_fn(y=sdf[metric], name=sid))
    else:
        sdf = df[df[session_col] == session]
        fig.add_trace(plot_fn(y=sdf[metric], name=session))
    fig.update_layout(
        yaxis_title=metric,
        # title=f"{metric} distribution",
    )
    return fig


def plot_scatter(df, x_col, y_col, session_col="session_id", session=None):
    """Return a Plotly scatter plot for the given DataFrame and columns."""
    fig = go.Figure()

    if session is None or session == "All":
        for sid in sorted(df[session_col].unique()):
            sdf = df[df[session_col] == sid]
            fig.add_trace(
                go.Scatter(
                    x=sdf[x_col],
                    y=sdf[y_col],
                    mode="markers",
                    name=sid,
                    customdata=sdf.index.values.reshape(-1, 1),
                )
            )
    else:
        sdf = df[df[session_col] == session]
        fig.add_trace(
            go.Scatter(
                x=sdf[x_col],
                y=sdf[y_col],
                mode="markers",
                name=session,
                customdata=sdf.index.values.reshape(-1, 1),
            )
        )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Session",
        ## title=f"Scatter Plot: {x_col} vs {y_col}",
    )
    return fig


def plot_compare_scatter(df1, df2, metric_x, metric_y, file_label_x, file_label_y):
    """Return a Plotly scatter plot comparing a metric from two DataFrames,
    with y=x reference line."""
    fig = go.Figure()
    session_ids = df1["session_id"].unique()
    for sid in session_ids:
        mask = df1["session_id"] == sid
        fig.add_trace(
            go.Scatter(
                x=df1.loc[mask, metric_x],
                y=df2.loc[mask, metric_y],
                mode="markers",
                name=sid,
                customdata=df1.loc[mask].index.values.reshape(-1, 1),
            )
        )

    # Add y=x reference line (always visible, not in legend)
    all_x = pd.concat([df1[metric_x], df2[metric_y]])
    min_val = all_x.min()
    max_val = all_x.max()
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="white", width=3, dash="dash"),
            opacity=0.9,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        xaxis_title=f"{metric_x} ({file_label_x})",
        yaxis_title=f"{metric_y} ({file_label_y})",
        legend_title="Session",
        title=f"Comparison: {metric_x} vs {metric_y}",
    )
    return fig


# --- Streamlit UI functions ---


def handle_audio_selection(
    result, metrics_dfs, experiments, ref_segment_dir, segment_dir
):
    """Handle audio playback for selected points in a plotly chart result.
    metrics_dfs: list of DataFrames (one or two)
    experiments: list of experiment labels (one or two)
    cfg: config object with ref_segment_dir and segment_dir
    multi: if True, expects two DataFrames/experiments (for compare mode)
    """

    if (
        result is None
        or "selection" not in result
        or "points" not in result["selection"]
    ):
        return

    for point in result["selection"]["points"]:
        index = point["customdata"]["0"]
        show_ref = True  # Show reference audio for the first metric_df
        for metrics_df, experiment in zip(metrics_dfs, experiments):
            print(metrics_df.iloc[index])
            filename = metrics_df.iloc[index]["key"]
            show_audio(
                experiment, filename, ref_segment_dir, segment_dir, show_ref=show_ref
            )
            show_ref = False


def explore_results_file(
    csv_files,
    exp_params,
    selected_idx,
    plot_type,
    session,
    x_col=None,
    y_col=None,
):
    """Streamlit UI and plotting for exploring a single results file."""
    exp_labels = make_exp_labels(exp_params)
    selected_file = csv_files[selected_idx]
    metrics_df = load_metrics(selected_file)
    experiment = exp_params[selected_idx]

    if session == "All" or session is None:
        selected_metrics_df = metrics_df
    else:
        selected_metrics_df = metrics_df[metrics_df["session_id"] == session]

    st.write(f"## Data Preview: {exp_labels[selected_idx]}")
    st.dataframe(selected_metrics_df.head())
    result = None

    if plot_type == "Scatter":
        st.write(f"### :red[{x_col}] *vs* :blue[{y_col}]")
        fig = plot_scatter(
            metrics_df, x_col, y_col, session_col="session_id", session=session
        )
        result = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

        corr = selected_metrics_df[[x_col, y_col]].corr().iloc[0, 1]
        st.write(f"#### :green[Correlation Coefficient = {corr:.2f}]")

    elif plot_type in ("Violin", "Box"):
        st.write(f"### Distribution of :green[{x_col}]")

        plot_class = go.Violin if plot_type == "Violin" else go.Box
        kwargs = (
            dict(box_visible=True, meanline_visible=True)
            if plot_type == "Violin"
            else dict(boxpoints="all", jitter=0.3)
        )
        plot_fn = functools.partial(plot_class, **kwargs)
        fig = plot_distribution(
            metrics_df, x_col, plot_fn, session_col="session_id", session=session
        )

        st.plotly_chart(fig, use_container_width=True)

        stats = selected_metrics_df[x_col].describe()
        # stats contains: count, mean, std, min, 25%, 50%, 75%, max
        # Use columns for clearer display with centered, green headings and values
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                "<div style='text-align: center;'>"
                "<h4 style='color: #21ba45;'>Count</h4>"
                "<p style='font-size:2em; margin:0;'><b>{}</b></p>"
                "</div>".format(f"{int(stats['count'])}"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                "<div style='text-align: center;'>"
                "<h4 style='color: #21ba45;'>Mean</h4>"
                "<p style='font-size:2em; margin:0;'><b>{}</b></p>"
                "</div>".format(f"{stats['mean']:.2f}"),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                "<div style='text-align: center;'>"
                "<h4 style='color: #21ba45;'>Std</h4>"
                "<p style='font-size:2em; margin:0;'><b>{}</b></p>"
                "</div>".format(f"{stats['std']:.2f}"),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                "<div style='text-align: center;'>"
                "<h4 style='color: #21ba45;'>Min</h4>"
                "<p style='font-size:2em; margin:0;'><b>{}</b></p>"
                "</div>".format(f"{stats['min']:.2f}"),
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                "<div style='text-align: center;'>"
                "<h4 style='color: #21ba45;'>Max</h4>"
                "<p style='font-size:2em; margin:0;'><b>{}</b></p>"
                "</div>".format(f"{stats['max']:.2f}"),
                unsafe_allow_html=True,
            )

    else:
        st.error("Unsupported plot type selected.")

    return result, [metrics_df], [experiment]


def compare_results_files(
    csv_files,
    exp_params,
    idx1,
    idx2,
    metric_x,
    metric_y,
):
    """Streamlit UI and plotting for comparing two results files."""
    st.sidebar.markdown(
        "Compare a metric from one results file with another (1-to-1 rows)."
    )
    exp_labels = make_exp_labels(exp_params)
    file1, file2 = csv_files[idx1], csv_files[idx2]
    metrics1_df = load_metrics(file1)
    metrics2_df = load_metrics(file2)
    experiment1 = exp_params[idx1]
    experiment2 = exp_params[idx2]

    # Align and keep only common keys, sorted
    metrics1_df = (
        metrics1_df.merge(metrics2_df[["key"]], on="key")
        .sort_values("key")
        .reset_index(drop=True)
    )
    metrics2_df = (
        metrics2_df.merge(metrics1_df[["key"]], on="key")
        .sort_values("key")
        .reset_index(drop=True)
    )

    numeric_cols1 = get_numeric_cols(metrics1_df, exclude=["session_id"])
    numeric_cols2 = get_numeric_cols(metrics2_df, exclude=["session_id"])
    common_metrics = sorted(list(set(numeric_cols1) & set(numeric_cols2)))
    if not common_metrics:
        st.warning("No common numeric metrics found between selected files.")
        return None, None, None

    st.write(f"### Compare Results: {exp_labels[idx1]} vs {exp_labels[idx2]}")
    st.write(
        f"Scatter plot: {metric_x} ({exp_labels[idx1]}) "
        f"vs {metric_y} ({exp_labels[idx2]})"
    )

    fig = plot_compare_scatter(
        metrics1_df,
        metrics2_df,
        metric_x,
        metric_y,
        exp_labels[idx1],
        exp_labels[idx2],
    )
    result = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    return result, [metrics1_df, metrics2_df], [experiment1, experiment2]


def show_audio(experiment, filename, ref_segment_dir, segment_dir, show_ref=True):
    def audio_bytes(path):
        with open(path, "rb") as f:
            return f.read()

    segment_type = experiment["segment_type"]
    exp_name = experiment["exp_name"]
    device = experiment["device"]

    full_filename = os.path.join(
        segment_dir.format(exp_name=exp_name, device=device, segment_type=segment_type),
        filename,
    )

    st.write(f"### Playing audio: {os.path.basename(full_filename)}")

    if show_ref:
        ref_filename = os.path.join(
            ref_segment_dir.format(
                dataset=DATASPLIT, device=device, segment_type=segment_type
            ),
            filename,
        )
        print(f"Reference audio file: {ref_filename}")
        if os.path.exists(ref_filename):
            st.markdown(f"**{exp_name} Reference Audio:**")
            st.audio(audio_bytes(ref_filename), format="audio/wav", autoplay=False)

    print(f"Experiment audio file: {full_filename}")
    if os.path.exists(full_filename):
        st.markdown(f"**{exp_name} Enhanced Audio:**")
        st.audio(audio_bytes(full_filename), format="audio/wav", autoplay=show_ref)


@hydra.main(
    version_base=None, config_path="../config/tools", config_name="MetricViewer.yaml"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for the metrics explorer Streamlit app."""
    logging.info("View metrics")
    logging.info(cfg.root_dir)

    # --- Find all CSV results files ---
    csv_files, exp_params = find_files(
        cfg.report_file, overrides={"session": "_", "pid": "_"}
    )
    exp_labels = make_exp_labels(exp_params)

    st.set_page_config(page_title="Metrics Explorer", layout="wide")
    # --- Sidebar: Mode selection ---
    st.sidebar.title("Controls")
    mode = st.sidebar.radio("Mode", ["Explore Single Results", "Compare Results Files"])

    if mode == "Explore Single Results":
        selected_idx = st.sidebar.selectbox(
            "Results File", range(len(csv_files)), format_func=lambda i: exp_labels[i]
        )
        selected_file = csv_files[selected_idx]
        metrics_df = load_metrics(selected_file)
        unique_session_ids = sorted(metrics_df["session_id"].unique())
        numeric_cols = get_numeric_cols(metrics_df, exclude=["session_id"])
        plot_type = st.sidebar.selectbox("Plot Type", ["Scatter", "Violin", "Box"])

        # Show controls depending on plot type
        session = st.sidebar.selectbox("Session", ["All"] + unique_session_ids)
        if plot_type == "Scatter":
            x_col = st.sidebar.selectbox("X Axis", numeric_cols, index=0)
            y_col = st.sidebar.selectbox(
                "Y Axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0
            )
        else:  # Violin or Box
            x_col = st.sidebar.selectbox("Metric", numeric_cols)
            y_col = None

        result, metrics_dfs, experiments = explore_results_file(
            csv_files,
            exp_params,
            selected_idx=selected_idx,
            plot_type=plot_type,
            session=session,
            x_col=x_col,
            y_col=y_col,
        )
    else:
        # Place controls specific to compare mode here
        st.sidebar.markdown(
            "Compare a metric from one results file with another (1-to-1 rows)."
        )
        idx1 = st.sidebar.selectbox(
            "Results File X",
            range(len(csv_files)),
            format_func=lambda i: exp_labels[i],
            key="cmp_x",
        )
        idx2 = st.sidebar.selectbox(
            "Results File Y",
            range(len(csv_files)),
            format_func=lambda i: exp_labels[i],
            key="cmp_y",
        )
        # Load the data here to get numeric columns for the next controls
        file1, file2 = csv_files[idx1], csv_files[idx2]
        metrics1_df = load_metrics(file1)
        metrics2_df = load_metrics(file2)
        numeric_cols1 = get_numeric_cols(metrics1_df, exclude=["session_id"])
        numeric_cols2 = get_numeric_cols(metrics2_df, exclude=["session_id"])
        common_metrics = sorted(list(set(numeric_cols1) & set(numeric_cols2)))
        metric_x = st.sidebar.selectbox(
            "Metric from X file", common_metrics, key="metric_x"
        )
        metric_y = st.sidebar.selectbox(
            "Metric from Y file", common_metrics, key="metric_y"
        )
        # Call the function, passing only the controls needed
        result, metrics_dfs, experiments = compare_results_files(
            csv_files,
            exp_params,
            idx1=idx1,
            idx2=idx2,
            metric_x=metric_x,
            metric_y=metric_y,
        )

    # Check if any scatter points were selected and play corresponding audio

    handle_audio_selection(
        result, metrics_dfs, experiments, cfg.ref_segment_dir, cfg.segment_dir
    )


if __name__ == "__main__":
    main()
