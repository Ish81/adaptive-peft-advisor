
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Adaptive SLM–PEFT Deployment Advisor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Adaptive SLM–PEFT Deployment Advisor")
st.caption("Benchmark-driven SLM, PEFT and deployment recommendations")

DATA_PATH = "multi_model_deployment_benchmark.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load benchmark CSV: {e}")
    st.stop()

# Actual benchmark column names
required_cols = [
    "model",
    "training_method",
    "format",
    "accuracy",
    "f1",
    "model_memory_footprint_mb",
    "peak_gpu_memory_gb",
    "avg_batch_latency_ms",
    "throughput_examples_per_sec"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

st.sidebar.header("Deployment Requirements")

gpu_vram = st.sidebar.slider(
    "Available GPU VRAM (GB)", 1.0, 16.0, 8.0, 0.5
)

max_latency = st.sidebar.slider(
    "Maximum average batch latency (ms)", 50, 500, 150, 5
)

min_accuracy = st.sidebar.slider(
    "Minimum accuracy", 0.80, 1.00, 0.90, 0.01
)

min_f1 = st.sidebar.slider(
    "Minimum F1 score", 0.80, 1.00, 0.90, 0.01
)

st.sidebar.header("Recommendation Weights")

w_quality = st.sidebar.slider("Quality", 0.0, 1.0, 0.4, 0.05)
w_memory = st.sidebar.slider("Memory efficiency", 0.0, 1.0, 0.2, 0.05)
w_latency = st.sidebar.slider("Latency", 0.0, 1.0, 0.2, 0.05)
w_throughput = st.sidebar.slider("Throughput", 0.0, 1.0, 0.2, 0.05)

weights = np.array([
    w_quality, w_memory, w_latency, w_throughput
])

if weights.sum() == 0:
    weights = np.array([0.4, 0.2, 0.2, 0.2])

weights = weights / weights.sum()

st.sidebar.caption("Weights are automatically normalized.")

# Requirements summary
st.subheader("Deployment Requirements")

c1, c2, c3, c4 = st.columns(4)

c1.metric("GPU VRAM", f"{gpu_vram:.1f} GB")
c2.metric("Maximum latency", f"{max_latency} ms")
c3.metric("Minimum accuracy", f"{min_accuracy:.2f}")
c4.metric("Minimum F1", f"{min_f1:.2f}")

st.divider()

# Apply hard constraints
feasible = df[
    (df["peak_gpu_memory_gb"] <= gpu_vram) &
    (df["avg_batch_latency_ms"] <= max_latency) &
    (df["accuracy"] >= min_accuracy) &
    (df["f1"] >= min_f1)
].copy()

st.subheader("Recommendation")

if feasible.empty:

    st.warning(
        "No configuration satisfies all selected deployment constraints."
    )

    diagnostics = df.copy()

    diagnostics["constraint_violations"] = (
        (diagnostics["peak_gpu_memory_gb"] > gpu_vram).astype(int) +
        (diagnostics["avg_batch_latency_ms"] > max_latency).astype(int) +
        (diagnostics["accuracy"] < min_accuracy).astype(int) +
        (diagnostics["f1"] < min_f1).astype(int)
    )

    st.markdown("### Closest Available Configurations")

    st.dataframe(
        diagnostics.sort_values(
            ["constraint_violations", "f1"],
            ascending=[True, False]
        ),
        use_container_width=True
    )

else:

    # Normalize metrics within feasible candidates
    def normalize(series, higher_is_better=True):
        low = series.min()
        high = series.max()

        if high == low:
            return pd.Series(1.0, index=series.index)

        result = (series - low) / (high - low)

        if not higher_is_better:
            result = 1 - result

        return result

    feasible["quality_score"] = (
        normalize(feasible["accuracy"]) +
        normalize(feasible["f1"])
    ) / 2

    feasible["memory_score"] = normalize(
        feasible["peak_gpu_memory_gb"],
        higher_is_better=False
    )

    feasible["latency_score"] = normalize(
        feasible["avg_batch_latency_ms"],
        higher_is_better=False
    )

    feasible["throughput_score"] = normalize(
        feasible["throughput_examples_per_sec"]
    )

    feasible["advisor_score"] = (
        weights[0] * feasible["quality_score"] +
        weights[1] * feasible["memory_score"] +
        weights[2] * feasible["latency_score"] +
        weights[3] * feasible["throughput_score"]
    )

    feasible = feasible.sort_values(
        "advisor_score",
        ascending=False
    ).reset_index(drop=True)

    best = feasible.iloc[0]

    st.success("A feasible deployment configuration was found.")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{best['accuracy']:.4f}")
    m2.metric("F1 Score", f"{best['f1']:.4f}")
    m3.metric("Peak GPU Memory", f"{best['peak_gpu_memory_gb']:.3f} GB")
    m4.metric("Average Batch Latency", f"{best['avg_batch_latency_ms']:.2f} ms")

    st.markdown("### Recommended Configuration")

    r1, r2 = st.columns(2)

    with r1:
        st.write("**Model:**", best["model"])
        st.write("**Training Method:**", best["training_method"])
        st.write("**Deployment Format:**", best["format"])

    with r2:
        st.write(
            "**Model Memory Footprint:**",
            f"{best['model_memory_footprint_mb']:.2f} MB"
        )
        st.write(
            "**Throughput:**",
            f"{best['throughput_examples_per_sec']:.2f} examples/sec"
        )
        st.write("**Advisor Score:**", f"{best['advisor_score']:.4f}")

    st.markdown("### Ranked Feasible Alternatives")

    display_cols = [
        "model",
        "training_method",
        "format",
        "accuracy",
        "f1",
        "model_memory_footprint_mb",
        "peak_gpu_memory_gb",
        "avg_batch_latency_ms",
        "throughput_examples_per_sec",
        "advisor_score"
    ]

    st.dataframe(
        feasible[display_cols],
        use_container_width=True
    )

st.divider()

st.subheader("Complete Benchmark Dataset")
st.dataframe(df, use_container_width=True)

st.caption(
    "The advisor uses a weighted heuristic over measured benchmark "
    "configurations. Scores are normalized among feasible candidates."
)
