"""
dashboard.py — Local analysis dashboard for chess-ai training runs.

Drop in any combination of CSV files, or point it at a run directory.
Supports: games.csv, training.csv, eval_games.csv, regression.csv,
          openings.csv, end_reasons.csv, snapshots.csv

Run with:
  venv/bin/streamlit run dashboard.py
"""

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="HAL-4000 Dashboard",
    layout="wide",
    page_icon="♟",
)

st.title("♟ HAL-4000 Training Dashboard")

# ---------------------------------------------------------------------------
# File detection and loading
# ---------------------------------------------------------------------------

def detect_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"w_wins", "b_move", "missing_queen"} <= cols:
        return "regression"
    if {"matchup", "result"} <= cols:
        return "eval"
    if {"white_wins", "black_wins", "avg_loss"} <= cols:
        return "training"
    if {"outcome", "end_reason", "n_moves"} <= cols:
        return "games"
    if {"checkmates", "material_resigns", "value_resigns"} <= cols:
        return "end_reasons"
    if {"position", "move1", "pct1"} <= cols:
        return "snapshots"
    if {"game", "moves"} <= cols and "outcome" not in cols and "position" not in cols:
        return "openings"
    return "unknown"

def load_run_dir(path: str) -> dict:
    p = Path(path)
    result = {}
    for fname, key in [
        ("games.csv", "games"),
        ("training.csv", "training"),
        ("eval_games.csv", "eval"),
        ("regression.csv", "regression"),
        ("openings.csv", "openings"),
        ("end_reasons.csv", "end_reasons"),
        ("snapshots.csv", "snapshots"),
    ]:
        fp = p / fname
        if fp.exists():
            result[key] = pd.read_csv(fp)
    return result

# ---------------------------------------------------------------------------
# Sidebar — data loading
# ---------------------------------------------------------------------------

st.sidebar.header("Load Data")

run_dir = st.sidebar.text_input(
    "Run directory",
    placeholder="logs/run11/",
    help="Load all CSVs from a run directory at once",
)
uploaded = st.sidebar.file_uploader(
    "Or upload CSV files",
    type="csv",
    accept_multiple_files=True,
)

dfs: dict[str, pd.DataFrame] = {}

if run_dir:
    loaded = load_run_dir(run_dir)
    if loaded:
        dfs.update(loaded)
        st.sidebar.success(f"Loaded {', '.join(loaded)} from {run_dir}")
    else:
        st.sidebar.error(f"No recognised CSVs found in {run_dir}")

for f in (uploaded or []):
    try:
        df = pd.read_csv(f, on_bad_lines="skip")
    except Exception as e:
        st.sidebar.warning(f"{f.name}: could not parse ({e})")
        continue
    ftype = detect_type(df)
    if ftype != "unknown":
        dfs[ftype] = df
    else:
        st.sidebar.warning(f"{f.name}: columns not recognised")

if not dfs:
    st.info("Enter a run directory path or upload CSV files to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Section: games.csv
# ---------------------------------------------------------------------------

if "games" in dfs:
    df = dfs["games"]
    st.header("Training Games")

    total = len(df)
    n_w = (df["outcome"] == "W").sum()
    n_b = (df["outcome"] == "B").sum()
    n_d = (df["outcome"] == "D").sum()
    n_mates = (df["end_reason"] == "checkmate").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", f"{total:,}")
    c2.metric("White wins", f"{n_w} ({n_w/total*100:.0f}%)")
    c3.metric("Black wins", f"{n_b} ({n_b/total*100:.0f}%)")
    c4.metric("Draws", str(n_d))
    c5.metric("Checkmates", f"{n_mates} ({n_mates/total*100:.1f}%)")

    window = max(50, total // 20)

    col_left, col_right = st.columns(2)

    with col_left:
        df["_w"] = (df["outcome"] == "W").astype(float)
        df["_b"] = (df["outcome"] == "B").astype(float)
        df["_w_roll"] = df["_w"].rolling(window, min_periods=1).mean() * 100
        df["_b_roll"] = df["_b"].rolling(window, min_periods=1).mean() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["game"], y=df["_w_roll"],
            name="White win %", line=dict(color="#f0c040", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["game"], y=df["_b_roll"],
            name="Black win %", line=dict(color="#8080e0", width=2),
        ))
        fig.update_layout(
            title=f"W/B win rate (rolling {window} games)",
            xaxis_title="Game", yaxis_title="%",
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        df["_loss_roll"] = df["loss"].rolling(window, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["game"], y=df["_loss_roll"],
            name="Loss", line=dict(color="#e05050", width=2),
        ))
        fig2.update_layout(
            title=f"Training loss (rolling {window} games)",
            xaxis_title="Game", yaxis_title="Loss",
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        # End reason breakdown — bin games into ~20 groups
        n_bins = max(10, total // 200)
        df["_bin"] = (df["game"] // (total // n_bins)) * (total // n_bins)
        reason_df = (
            df.groupby(["_bin", "end_reason"])
            .size()
            .reset_index(name="count")
        )
        fig3 = px.bar(
            reason_df, x="_bin", y="count", color="end_reason",
            barmode="stack",
            title="End reasons over training",
            labels={"_bin": "Game", "count": "Games", "end_reason": "Reason"},
            height=320,
            color_discrete_map={
                "checkmate":       "#50c050",
                "material_resign": "#f0c040",
                "value_resign":    "#e09000",
                "cap_draw":        "#6060c0",
                "rule_draw":       "#aaaaaa",
            },
        )
        fig3.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        fig4 = px.histogram(
            df, x="n_moves", nbins=40,
            title="Game length distribution",
            labels={"n_moves": "Moves", "count": "Games"},
            color_discrete_sequence=["#50a0c0"],
            height=320,
        )
        fig4.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# Section: regression.csv
# ---------------------------------------------------------------------------

if "regression" in dfs:
    df = dfs["regression"]
    st.header("Value Head Regression")

    for col in ["start", "w_wins", "b_move", "missing_queen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("start",          f"{latest['start']:+.4f}",          help="Expected ~0.0")
    c2.metric("w_wins",         f"{latest['w_wins']:+.4f}",         help="Expected near +1")
    c3.metric("b_move",         f"{latest['b_move']:+.4f}",         help="Expected near -1")
    c4.metric("missing_queen",  f"{latest['missing_queen']:+.4f}",  help="Expected < 0")

    fig = go.Figure()
    palette = {
        "start":         "#aaaaaa",
        "w_wins":        "#f0c040",
        "b_move":        "#8080e0",
        "missing_queen": "#e05050",
    }
    labels = {
        "start":         "start (~0.0)",
        "w_wins":        "K+Q vs K, W to move (→ +1)",
        "b_move":        "K+Q vs K, B to move (→ -1)",
        "missing_queen": "white missing queen (→ < 0)",
    }
    for col, color in palette.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["game"], y=df[col],
                name=labels[col],
                line=dict(color=color, width=2),
            ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        title="Value head signal over training",
        xaxis_title="Game", yaxis_title="Value",
        yaxis=dict(range=[-1.1, 1.1]),
        height=420, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section: training.csv
# ---------------------------------------------------------------------------

if "training" in dfs:
    df = dfs["training"]
    st.header("Training Windows")

    df["_total"] = df["white_wins"] + df["black_wins"] + df["draws"]
    df["_w_pct"] = df["white_wins"] / df["_total"] * 100
    df["_b_pct"] = df["black_wins"] / df["_total"] * 100

    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["game"], y=df["_w_pct"],
            name="White win %", line=dict(color="#f0c040", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df["game"], y=df["_b_pct"],
            name="Black win %", line=dict(color="#8080e0", width=2),
        ))
        fig.update_layout(
            title="Win rate per 50-game window",
            xaxis_title="Game", yaxis_title="%",
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["game"], y=df["avg_loss"],
            name="Avg loss", line=dict(color="#e05050", width=2),
        ))
        fig2.update_layout(
            title="Average loss per window",
            xaxis_title="Game", yaxis_title="Loss",
            height=320, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["game"], y=df["avg_game_length"],
            name="Avg game length", line=dict(color="#50c0a0", width=2),
        ))
        fig3.update_layout(
            title="Average game length per window",
            xaxis_title="Game", yaxis_title="Moves",
            height=300, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        length_cols = ["len_0_20", "len_21_40", "len_41_60", "len_61_80", "len_81plus"]
        length_labels = ["0-20", "21-40", "41-60", "61-80", "81+"]
        avail = [c for c in length_cols if c in df.columns]
        if avail:
            melt = df[["game"] + avail].melt(id_vars="game", var_name="bucket", value_name="count")
            melt["bucket"] = melt["bucket"].map(dict(zip(length_cols, length_labels)))
            fig4 = px.bar(
                melt, x="game", y="count", color="bucket",
                barmode="stack",
                title="Game length buckets per window",
                labels={"game": "Game", "count": "Games", "bucket": "Length"},
                height=300,
            )
            fig4.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# Section: openings.csv
# ---------------------------------------------------------------------------

if "openings" in dfs:
    df = dfs["openings"]
    st.header("Opening Repertoire")

    df["first_move"] = df["moves"].fillna("").str.split().str[0]
    df = df[df["first_move"] != ""]

    total = len(df)
    window_size = max(200, total // 25)

    top_moves = df["first_move"].value_counts().head(6).index.tolist()
    df["_move_group"] = df["first_move"].apply(lambda m: m if m in top_moves else "other")
    df["_bin"] = (df["game"] // window_size) * window_size

    col_left, col_right = st.columns(2)

    with col_left:
        move_bin = df.groupby(["_bin", "_move_group"]).size().reset_index(name="count")
        move_bin["pct"] = move_bin["count"] / move_bin.groupby("_bin")["count"].transform("sum") * 100
        fig = px.bar(
            move_bin, x="_bin", y="pct", color="_move_group",
            barmode="stack",
            title=f"First-move distribution (windows of {window_size} games)",
            labels={"_bin": "Game", "pct": "% games", "_move_group": "First move"},
            height=360,
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        a2a3_by_bin = (
            df.groupby("_bin")
            .apply(lambda x: (x["first_move"] == "a2a3").mean() * 100)
            .reset_index(name="pct")
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=a2a3_by_bin["_bin"], y=a2a3_by_bin["pct"],
            name="a2a3 %", line=dict(color="#e05050", width=2),
            fill="tozeroy", fillcolor="rgba(224,80,80,0.15)",
        ))
        fig2.update_layout(
            title="a2a3 lock-in over training",
            xaxis_title="Game", yaxis_title="% games starting a2a3",
            yaxis=dict(range=[0, 100]),
            height=360, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Most common opening sequences (first 12 moves)")
    top_seqs = df["moves"].value_counts().head(10).reset_index()
    top_seqs.columns = ["Sequence", "Count"]
    top_seqs["% of games"] = (top_seqs["Count"] / total * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(top_seqs, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Section: end_reasons.csv
# ---------------------------------------------------------------------------

if "end_reasons" in dfs:
    df = dfs["end_reasons"]
    st.header("End Reason Transitions")

    reason_cols = ["checkmates", "material_resigns", "value_resigns", "cap_draws", "rule_draws"]
    avail = [c for c in reason_cols if c in df.columns]
    df["_total"] = df[avail].sum(axis=1).replace(0, 1)

    col_left, col_right = st.columns(2)

    with col_left:
        if "value_resigns" in df.columns and "material_resigns" in df.columns:
            df["_val_pct"]  = df["value_resigns"]   / df["_total"] * 100
            df["_mat_pct"]  = df["material_resigns"] / df["_total"] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["game"], y=df["_mat_pct"],
                name="Material resign", line=dict(color="#f0c040", width=2),
                fill="tozeroy", fillcolor="rgba(240,192,64,0.25)",
            ))
            fig.add_trace(go.Scatter(
                x=df["game"], y=df["_val_pct"],
                name="Value resign", line=dict(color="#e09000", width=2),
                fill="tozeroy", fillcolor="rgba(224,144,0,0.25)",
            ))
            fig.update_layout(
                title="Material resign vs value resign (% of games per window)",
                xaxis_title="Game", yaxis_title="% of games",
                height=340, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "checkmates" in df.columns:
            df["_mates_pct"] = df["checkmates"] / df["_total"] * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["game"], y=df["_mates_pct"],
                name="Checkmate %", line=dict(color="#50c050", width=2),
                fill="tozeroy", fillcolor="rgba(80,192,80,0.15)",
            ))
            fig2.update_layout(
                title="Checkmate rate over training",
                xaxis_title="Game", yaxis_title="% of games",
                height=340, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Section: snapshots.csv
# ---------------------------------------------------------------------------

if "snapshots" in dfs:
    df = dfs["snapshots"]
    st.header("Policy Snapshots")

    def parse_pct(val):
        try:
            return float(str(val).replace("%", ""))
        except Exception:
            return None

    df["_pct1"] = df["pct1"].apply(parse_pct)

    start_df = df[df["position"] == "start"].copy()

    col_left, col_right = st.columns(2)

    with col_left:
        if not start_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=start_df["game"], y=start_df["_pct1"],
                name="Top-1 visit %", line=dict(color="#50a0e0", width=2),
                fill="tozeroy", fillcolor="rgba(80,160,224,0.15)",
                text=start_df["move1"],
                hovertemplate="Game %{x}<br>Top move: %{text}<br>Visit share: %{y:.1f}%",
            ))
            fig.update_layout(
                title="Policy confidence at start position (top-1 MCTS visit share)",
                xaxis_title="Game", yaxis_title="% visits on top move",
                height=340, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if not start_df.empty:
            top_opening_moves = start_df["move1"].value_counts().head(5).index.tolist()
            start_df = start_df.copy()
            start_df["_move_group"] = start_df["move1"].apply(
                lambda m: m if m in top_opening_moves else "other"
            )
            n_bins = max(10, len(start_df) // 5)
            bin_size = max(1, len(start_df) // n_bins)
            start_df["_bin"] = (start_df.index // bin_size)
            bin_game = start_df.groupby("_bin")["game"].first().to_dict()
            start_df["_bin_game"] = start_df["_bin"].map(bin_game)
            move_bin = (
                start_df.groupby(["_bin_game", "_move_group"])
                .size()
                .reset_index(name="count")
            )
            fig2 = px.bar(
                move_bin, x="_bin_game", y="count", color="_move_group",
                barmode="stack",
                title="Top move at start position over time",
                labels={"_bin_game": "Game", "count": "Snapshots", "_move_group": "Move"},
                height=340,
            )
            fig2.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Latest snapshot at each canonical position")
    latest = df.sort_values("game").groupby("position").last().reset_index()
    show_cols = ["position", "game", "move1", "pct1", "move2", "pct2", "move3", "pct3", "move4", "pct4", "move5", "pct5"]
    avail_cols = [c for c in show_cols if c in latest.columns]
    st.dataframe(latest[avail_cols], width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Section: eval_games.csv
# ---------------------------------------------------------------------------

if "eval" in dfs:
    df = dfs["eval"]
    st.header("Eval Results")

    def outcome_label(row):
        hal_white = "(W)" in str(row["matchup"]) and "HAL" in str(row["matchup"]).split("(W)")[0]
        # Simpler: check if HAL is listed first as white
        parts = str(row["matchup"]).split(" vs ")
        hal_is_white = len(parts) > 0 and "(W)" in parts[0]
        r = str(row["result"])
        if r == "1-0":
            return "HAL win" if hal_is_white else "HAL loss"
        if r == "0-1":
            return "HAL loss" if hal_is_white else "HAL win"
        if r == "*":
            return "Cap draw"
        return "Draw"

    df["_outcome"] = df.apply(outcome_label, axis=1)

    matchups = sorted(df["matchup"].unique())

    summary_rows = []
    for matchup in matchups:
        sub = df[df["matchup"] == matchup]
        total = len(sub)
        wins   = (sub["_outcome"] == "HAL win").sum()
        losses = (sub["_outcome"] == "HAL loss").sum()
        draws  = (sub["_outcome"] == "Draw").sum()
        caps   = (sub["_outcome"] == "Cap draw").sum()
        wd = (wins + draws + caps) / total * 100
        summary_rows.append({
            "Matchup":      matchup,
            "Games":        total,
            "HAL wins":     wins,
            "Losses":       losses,
            "Formal draws": draws,
            "Cap draws":    caps,
            "W/D %":        f"{wd:.0f}%",
        })

    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # If multiple evals exist (steps differs), show win rate over time
    if df["hal_steps"].nunique() > 1:
        trend = (
            df.groupby(["hal_steps", "matchup"])["_outcome"]
            .apply(lambda x: (x == "HAL win").mean() * 100)
            .reset_index(name="win_pct")
        )
        fig = px.line(
            trend, x="hal_steps", y="win_pct", color="matchup",
            title="HAL win rate over training steps",
            labels={"hal_steps": "Training steps", "win_pct": "Win %", "matchup": "Matchup"},
            height=380,
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# AI Commentary (optional)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("AI Commentary")

with st.expander("Generate analysis from Claude", expanded=False):
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Or set ANTHROPIC_API_KEY in your environment",
    )

    if st.button("Analyse", disabled=not api_key):
        try:
            import anthropic

            # Build a plain-text summary of the loaded data
            summary_parts = []

            if "games" in dfs:
                g = dfs["games"]
                total = len(g)
                n_w = (g["outcome"] == "W").sum()
                n_b = (g["outcome"] == "B").sum()
                n_d = (g["outcome"] == "D").sum()
                n_mates = (g["end_reason"] == "checkmate").sum()
                last_loss = g["loss"].iloc[-1]
                summary_parts.append(
                    f"Training games: {total} total. "
                    f"White wins: {n_w} ({n_w/total*100:.0f}%), "
                    f"Black wins: {n_b} ({n_b/total*100:.0f}%), "
                    f"Draws: {n_d}. "
                    f"Checkmates: {n_mates} ({n_mates/total*100:.1f}%). "
                    f"Latest loss: {last_loss:.4f}."
                )

            if "regression" in dfs:
                r = dfs["regression"]
                for col in ["start", "w_wins", "b_move", "missing_queen"]:
                    if col in r.columns:
                        r[col] = pd.to_numeric(r[col], errors="coerce")
                first = r.iloc[0]
                last  = r.iloc[-1]
                summary_parts.append(
                    f"Value head regression over {len(r)} readings "
                    f"(games {int(r['game'].iloc[0])}–{int(r['game'].iloc[-1])}):\n"
                    f"  start:          {first['start']:+.4f} → {last['start']:+.4f}\n"
                    f"  w_wins:         {first['w_wins']:+.4f} → {last['w_wins']:+.4f}\n"
                    f"  b_move:         {first['b_move']:+.4f} → {last['b_move']:+.4f}\n"
                    f"  missing_queen:  {first['missing_queen']:+.4f} → {last['missing_queen']:+.4f}"
                )

            if "eval" in dfs:
                e = dfs["eval"]
                for matchup in sorted(e["matchup"].unique()):
                    sub = e[e["matchup"] == matchup]
                    total = len(sub)
                    wins = (sub["_outcome"] == "HAL win").sum()
                    draws = (sub["_outcome"] == "Draw").sum()
                    caps  = (sub["_outcome"] == "Cap draw").sum()
                    summary_parts.append(
                        f"Eval vs {matchup}: {wins}/{total} wins, "
                        f"{draws} formal draws, {caps} cap draws."
                    )

            prompt = (
                "You are analysing training data for an AlphaZero-style chess agent "
                "trained from self-play using MCTS and a residual neural network. "
                "The value head outputs a scalar in [-1, +1] from the current player's perspective. "
                "Four regression positions track learning: start (balanced opening, ~0 expected), "
                "w_wins (K+Q vs K, White to move, +1 expected), "
                "b_move (K+Q vs K, Black to move, -1 expected), "
                "missing_queen (White opening with no queen, <0 expected).\n\n"
                "Here is the current training data summary:\n\n"
                + "\n\n".join(summary_parts)
                + "\n\nGive a concise plain-English analysis: what is the agent learning, "
                "what looks healthy, what looks like it needs attention. 3-5 short paragraphs."
            )

            client = anthropic.Anthropic(api_key=api_key)
            with st.spinner("Asking Claude..."):
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
            st.markdown(message.content[0].text)

        except ImportError:
            st.error("anthropic package not found. Run: venv/bin/pip install anthropic")
        except Exception as e:
            st.error(f"Error: {e}")
