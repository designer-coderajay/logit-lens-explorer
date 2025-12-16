"""
Logit Lens Explorer - Streamlit App
====================================

An interactive tool to visualize what GPT-2 "thinks" at each layer.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from logit_lens import LogitLens
import torch


@st.cache_resource
def load_model():
    """Load model once and cache it across reruns."""
    return LogitLens("gpt2-small")


def create_heatmap(analysis: dict) -> go.Figure:
    """
    Create a heatmap showing top-1 prediction probability at each layer.

    This visualization reveals WHERE the model becomes confident.
    """
    layers = []
    top_tokens = []
    top_probs = []

    for layer_info in analysis["layers"]:
        layers.append(layer_info["layer_name"])
        token, prob = layer_info["predictions"][0]
        top_tokens.append(repr(token))
        top_probs.append(prob)

    fig = go.Figure(data=go.Heatmap(
        z=[top_probs],
        x=layers,
        y=["Top-1 Prob"],
        text=[[f"{t}<br>{p:.3f}" for t, p in zip(top_tokens, top_probs)]],
        texttemplate="%{text}",
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Probability")
    ))

    fig.update_layout(
        title="Top Prediction Confidence Across Layers",
        xaxis_title="Layer",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_probability_evolution(analysis: dict, track_token: str = None) -> go.Figure:
    """
    Track how probability of specific tokens evolves across layers.

    If track_token is None, tracks the final layer's top prediction.
    """
    lens = load_model()
    prompt = analysis["prompt"]

    # Get the token to track
    if track_token is None:
        # Use the final prediction
        final_top = analysis["layers"][-1]["predictions"][0][0]
        track_token = final_top

    # Get the token ID
    token_id = lens.model.tokenizer.encode(track_token)
    if len(token_id) == 0:
        return None
    token_id = token_id[0]

    # Get residual streams and compute probabilities for this token
    residual_streams = lens.get_residual_stream_at_all_layers(prompt)

    probs = []
    layers = []

    for layer_idx in range(lens.n_layers + 1):
        residual = residual_streams[layer_idx, 0, -1]
        logits = lens.residual_to_logits(residual)
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)

        if layer_idx == 0:
            layers.append("Embed")
        else:
            layers.append(f"L{layer_idx-1}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers,
        y=probs,
        mode='lines+markers',
        name=f'P({repr(track_token)})',
        line=dict(width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f"Probability Evolution for {repr(track_token)}",
        xaxis_title="Layer",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=300
    )

    return fig


def main():
    st.set_page_config(
        page_title="Logit Lens Explorer",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Logit Lens Explorer")
    st.markdown("""
    **See what GPT-2 is "thinking" at each layer.**

    The logit lens projects intermediate residual streams into vocabulary space,
    revealing how the model builds up its prediction layer by layer.
    """)

    # Load model
    with st.spinner("Loading GPT-2 Small..."):
        lens = load_model()

    # Sidebar controls
    st.sidebar.header("Settings")

    prompt = st.sidebar.text_input(
        "Enter a prompt:",
        value="The Eiffel Tower is in",
        help="The model will predict what comes next"
    )

    top_k = st.sidebar.slider(
        "Top-K predictions:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top predictions to show per layer"
    )

    # Example prompts
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Try these prompts:**")
    examples = [
        "The Eiffel Tower is in",
        "Barack Obama was the",
        "The capital of France is",
        "import numpy as",
        "def fibonacci(n):",
        "Once upon a time",
        "The quick brown fox",
    ]

    for ex in examples:
        if st.sidebar.button(ex, key=f"btn_{ex}"):
            st.session_state.prompt = ex
            st.rerun()

    # Check for prompt update from button
    if "prompt" in st.session_state:
        prompt = st.session_state.prompt
        del st.session_state.prompt

    # Run analysis
    if prompt:
        with st.spinner("Analyzing..."):
            analysis = lens.analyze_prompt(prompt, top_k=top_k)

        # Display tokens
        st.subheader("Tokenization")
        token_display = " | ".join([f"`{t}`" for t in analysis["tokens"]])
        st.markdown(f"Tokens: {token_display}")
        st.caption(f"Analyzing position -1 (last token: `{analysis['tokens'][-1]}`)")

        # Main results in two columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Layer-by-Layer Predictions")

            # Build dataframe for display
            rows = []
            for layer_info in analysis["layers"]:
                row = {"Layer": layer_info["layer_name"]}
                for i, (token, prob) in enumerate(layer_info["predictions"]):
                    row[f"#{i+1}"] = f"{repr(token)} ({prob:.3f})"
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Key Observations")

            # Find where the final answer first appears
            final_top = analysis["layers"][-1]["predictions"][0][0]
            first_appearance = None

            for layer_info in analysis["layers"]:
                top_token = layer_info["predictions"][0][0]
                if top_token == final_top:
                    first_appearance = layer_info["layer_name"]
                    break

            st.markdown(f"**Final prediction:** `{repr(final_top)}`")

            if first_appearance:
                st.markdown(f"**First appears at:** {first_appearance}")

            # Confidence at embedding vs final
            embed_prob = analysis["layers"][0]["predictions"][0][1]
            final_prob = analysis["layers"][-1]["predictions"][0][1]

            st.markdown(f"**Embedding confidence:** {embed_prob:.3f}")
            st.markdown(f"**Final confidence:** {final_prob:.3f}")
            st.markdown(f"**Confidence gain:** {final_prob - embed_prob:.3f}")

        # Visualizations
        st.subheader("Visualizations")

        tab1, tab2 = st.tabs(["Confidence Heatmap", "Probability Evolution"])

        with tab1:
            fig = create_heatmap(analysis)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Let user pick a token to track
            all_tokens = set()
            for layer_info in analysis["layers"]:
                for token, _ in layer_info["predictions"]:
                    all_tokens.add(token)

            track_token = st.selectbox(
                "Track token probability:",
                options=sorted(all_tokens),
                index=0 if final_top not in sorted(all_tokens) else sorted(all_tokens).index(final_top)
            )

            fig = create_probability_evolution(analysis, track_token)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Understanding section
        with st.expander("🧠 How to Interpret These Results"):
            st.markdown("""
            ### What You're Seeing

            Each row shows what the model would predict if we stopped at that layer
            and asked "what token comes next?"

            ### Key Patterns to Look For

            1. **Gradual Refinement**: The model starts uncertain and becomes confident.
               Early layers might predict common words, late layers predict the specific answer.

            2. **Early Knowing**: Sometimes the model "knows" the answer very early.
               This suggests the information is easy to extract from the embedding.

            3. **Late Corrections**: Sometimes the model changes its mind in late layers.
               This suggests complex reasoning is happening.

            4. **Confidence Jumps**: Look for layers where confidence suddenly increases.
               These are where key computations happen.

            ### Technical Details

            - **Residual Stream**: The main information highway through the model
            - **W_U (Unembedding)**: Matrix that converts hidden states to vocabulary predictions
            - **LayerNorm**: Applied before unembedding (GPT-2's architecture)
            """)


if __name__ == "__main__":
    main()
