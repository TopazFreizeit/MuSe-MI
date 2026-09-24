import argparse
import sqlite3
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot semantic impact timeline from Phoenix DB")
    parser.add_argument("--trace-id", required=True, help="The trace ID to analyze")
    parser.add_argument("--db-path", default="~/.phoenix/phoenix.db", help="Path to phoenix.db")
    parser.add_argument("--output-dir", default="data/analytics", help="Output directory for the plot")
    return parser.parse_args()

def extract_value(attrs, key):
    """Helper to extract a value from either the root or metadata of span attributes."""
    val = attrs.get(key)
    if val is not None:
        return val
    return attrs.get("metadata", {}).get(key)

def main():
    args = parse_args()
    db_path = os.path.expanduser(args.db_path)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Connecting to Phoenix DB at {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    trace_row = conn.execute("SELECT id FROM traces WHERE trace_id=?", (args.trace_id,)).fetchone()
    if not trace_row:
        print(f"Error: Trace ID {args.trace_id} not found in database.")
        return
    trace_rowid = trace_row["id"]

    thread_row = conn.execute(
        "SELECT attributes FROM spans WHERE trace_rowid=? AND attributes LIKE '%thread_id%' LIMIT 1", 
        (trace_rowid,)
    ).fetchone()
    
    thread_id = args.trace_id
    if thread_row:
        try:
            t_attrs = json.loads(thread_row["attributes"])
            tid = t_attrs.get("therapy", {}).get("thread_id") or extract_value(t_attrs, "thread_id")
            if tid:
                thread_id = tid
        except Exception:
            pass

    query = """
    SELECT span_id, name, start_time, end_time, attributes
    FROM spans 
    WHERE trace_rowid=? AND name IN ('run_single_session', 'cognitive_interpreter')
    ORDER BY start_time ASC;
    """
    rows = conn.execute(query, (trace_rowid,)).fetchall()
    
    sessions = []
    interpreters = []
    
    for row in rows:
        try:
            attrs = json.loads(row["attributes"])
        except Exception:
            attrs = {}
            
        name = row["name"]
        
        if name == "run_single_session":
            session_num = attrs.get("therapy", {}).get("session", {}).get("number")
            if session_num is None:
                session_num = extract_value(attrs, "session_number")
                
            if session_num is not None:
                sessions.append({
                    "session_number": session_num,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"]
                })
            else:
                # If session_number isn't explicitly in attributes, we'll assign it incrementally
                sessions.append({
                    "session_number": len(sessions) + 1,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"]
                })
                
        elif name == "cognitive_interpreter":
            patient_attrs = attrs.get("therapy", {}).get("patient", {})
            readiness = patient_attrs.get("mr_impact")
            if readiness is None: readiness = extract_value(attrs, "mr_impact")
            
            reactance = patient_attrs.get("anger_impact")
            if reactance is None: reactance = extract_value(attrs, "anger_impact")
            
            se = patient_attrs.get("se_impact")
            if se is None: se = extract_value(attrs, "se_impact")
            
            pd_val = patient_attrs.get("pr_impact")
            if pd_val is None: pd_val = extract_value(attrs, "pr_impact")
            
            # If at least one of these is found, record it
            if readiness or reactance or se or pd_val:
                interpreters.append({
                    "start_time": row["start_time"],
                    "readiness_shift": readiness,
                    "reactance_impact": reactance,
                    "se_impact": se,
                    "pd_impact": pd_val
                })

    conn.close()

    if not sessions:
        print("Warning: No run_single_session spans found. Assuming all cognitive_interpreter spans belong to Session 1.")
        sessions.append({
            "session_number": 1,
            "start_time": "1970-01-01T00:00:00",
            "end_time": "2999-12-31T23:59:59"
        })
        
    if not interpreters:
        print("No cognitive_interpreter spans with semantic impacts found for this trace.")
        return

    data = []
    for interp in interpreters:
        t = interp["start_time"]
        assigned_session = None
        for s in sessions:
            # Match the start time to be within the session span
            if s["start_time"] <= t <= s["end_time"]:
                assigned_session = s["session_number"]
                break
                
        if assigned_session is not None:
            interp["session_number"] = assigned_session
            data.append(interp)
        else:
            # If timestamps don't strictly align, fallback to first session
            interp["session_number"] = sessions[0]["session_number"]
            data.append(interp)

    if not data:
        print("Could not associate interpreters with sessions.")
        return

    df = pd.DataFrame(data)
    df["turn_index"] = df.groupby("session_number").cumcount() + 1
    
    def map_score(val):
        val = str(val).lower()
        if val in ["success", "forward"]: return 1
        if val in ["resistance", "backward"]: return -1
        # neutral or unknown -> 0
        return 0

    df["motivational_readiness_score"] = df["readiness_shift"].apply(map_score)
    df["anger_score"] = df["reactance_impact"].apply(map_score)
    df["se_score"] = df["se_impact"].apply(map_score)
    df["pd_score"] = df["pd_impact"].apply(map_score)
    
    print("\n=== Semantic Impact Distributions ===")
    for sess_num, group in df.groupby("session_number"):
        print(f"\nSession {sess_num} (Turns: {len(group)})")
        print("  Motivational Readiness:    ", group["readiness_shift"].value_counts().to_dict())
        print("  Anger:    ", group["reactance_impact"].value_counts().to_dict())
        print("  Self-Efficacy:", group["se_impact"].value_counts().to_dict())
        print("  Perceived Dis:", group["pd_impact"].value_counts().to_dict())

    sns.set_theme(style="whitegrid")
    sessions_found = sorted(df["session_number"].unique())
    num_sessions = len(sessions_found)
    
    fig, axes = plt.subplots(num_sessions, 1, figsize=(10, 2.5 * num_sessions), sharey=True)
    if num_sessions == 1:
        axes = [axes]
        
    fig.suptitle("Cognitive Interpreter - Semantic Impact", fontsize=16, y=1.02)
        
    metrics = {
        "motivational_readiness_score": ("Motivational Readiness", "tab:blue"),
        "anger_score": ("Anger", "tab:orange"),
        "se_score": ("Self-Efficacy", "tab:green"),
        "pd_score": ("Problem Recognition", "tab:red")
    }
    
    max_turn = df["turn_index"].max()
    
    for i, sess_num in enumerate(sessions_found):
        ax = axes[i]
        sess_data = df[df["session_number"] == sess_num]
        
        for col, (label, color) in metrics.items():
            # Add slight vertical offset so lines don't completely overlap when values are identical
            offset = {"motivational_readiness_score": 0.03, "anger_score": 0.01, "se_score": -0.01, "pd_score": -0.03}[col]
            
            ax.plot(sess_data["turn_index"], sess_data[col] + offset, marker='o', 
                    label=label, color=color, linewidth=2, alpha=0.7)
            
        ax.set_title(f"Session {sess_num}")
        ax.set_ylabel("Impact Score")
        ax.set_xlabel("Turn Index")
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Resistance/Backward (-1)", "Neutral (0)", "Success/Forward (1)"])
        ax.set_xticks(range(1, max_turn + 1))
        
        # Add a light zero-line
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "semantic_impact_timeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTimeline plot saved to {out_path}")

    # --- Distribution Plot (Pie Charts) ---
    metrics_list = ["motivational_readiness_score", "anger_score", "se_score", "pd_score"]
    metric_titles = ["Motivational Readiness", "Anger", "Self-Efficacy", "Problem Recognition"]
    
    fig_dist, axes_dist = plt.subplots(num_sessions, len(metrics_list), figsize=(12, 3 * num_sessions))
    # Ensure axes_dist is 2D even if num_sessions == 1
    if num_sessions == 1:
        axes_dist = [axes_dist]
        
    fig_dist.suptitle("Cognitive Interpreter - Semantic Impact Distribution", fontsize=16, y=1.05)
    
    colors_map = {1: "tab:green", 0: "tab:gray", -1: "tab:red"}
    labels_map = {1: "Success/Forward", 0: "Neutral", -1: "Resistance/Backward"}
    
    for i, sess_num in enumerate(sessions_found):
        sess_data = df[df["session_number"] == sess_num]
        
        for j, metric in enumerate(metrics_list):
            ax = axes_dist[i][j]
            counts = sess_data[metric].value_counts().sort_index()
            
            if counts.empty:
                ax.axis('off')
                continue
                
            colors = [colors_map[val] for val in counts.index]
            
            ax.pie(counts, labels=None, colors=colors, autopct='%1.1f%%', startangle=90, 
                   wedgeprops={'edgecolor': 'w', 'linewidth': 1})
            
            # Set column titles on the first row
            if i == 0:
                ax.set_title(metric_titles[j], fontweight="bold")
                
            # Set row labels on the first column
            if j == 0:
                ax.text(-1.5, 0, f"Session {sess_num}", rotation=90, va="center", ha="center", fontweight="bold", fontsize=12)
                
    # Add a global legend
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=colors_map[k], label=labels_map[k]) for k in [1, 0, -1]]
    fig_dist.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    
    plt.tight_layout()
    dist_out_path = out_dir / "semantic_impact_distribution.png"
    plt.savefig(dist_out_path, dpi=300, bbox_inches='tight')
    plt.close(fig_dist)
    print(f"Distribution plot saved to {dist_out_path}")

if __name__ == "__main__":
    main()
