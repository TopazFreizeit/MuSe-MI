import argparse
import sqlite3
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot macro shift distribution from Phoenix DB")
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

def map_score(val):
    val = str(val).lower().strip()
    if val in ["major progress", "forward"]: return 2
    if val == "minor progress": return 1
    if val in ["no change", "neutral"]: return 0
    if val == "minor regression": return -1
    if val in ["major regression", "backward"]: return -2
    return 0

def get_semantic_impact(obj):
    if not isinstance(obj, dict):
        return 0
    val = obj.get("semantic_impact")
    return map_score(val)

from datetime import datetime, timedelta

def parse_time(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        # Fallback if there are fractional seconds without Z
        return pd.to_datetime(ts_str).to_pydatetime()

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

    # Extract thread_id
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
    WHERE trace_rowid=? 
      AND (name IN ('run_single_session', 'patient_rolling_memory', 'life_event_simulator') 
           OR span_kind = 'LLM' OR name LIKE 'Chat%')
    ORDER BY start_time ASC;
    """
    rows = conn.execute(query, (trace_rowid,)).fetchall()
    
    sessions = []
    events = []
    
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
                sessions.append({
                    "session_number": len(sessions) + 1,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"]
                })
                
        else:
            tags = attrs.get("tag", {}).get("tags", [])
            metadata = attrs.get("metadata", {})
            span_tags_and_node = tags + [metadata.get("node", "")]
            
            is_rm = "patient_rolling_memory" in span_tags_and_node
            is_les = "life_event_simulator" in span_tags_and_node
            
            if is_rm or is_les:
                val_str = attrs.get("output", {}).get("value")
                macro_shift = None
                if val_str:
                    try:
                        val_json = json.loads(val_str)
                        if "choices" in val_json:
                            content_str = val_json["choices"][0]["message"]["content"]
                            content_json = json.loads(content_str)
                            macro_shift = content_json.get("macro_shift")
                        else:
                            macro_shift = val_json.get("macro_shift")
                    except Exception:
                        pass
                
                if not macro_shift:
                    macro_shift = extract_value(attrs, "macro_shift")
                    
                if macro_shift and isinstance(macro_shift, dict):
                    readiness = get_semantic_impact(macro_shift.get("motivational_readiness"))
                    se = get_semantic_impact(macro_shift.get("self_efficacy"))
                    pd_val = get_semantic_impact(macro_shift.get("problem_recognition"))
                    anger = get_semantic_impact(macro_shift.get("anger"))
                    
                    event_name = "patient_rolling_memory" if is_rm else "life_event_simulator"
                    events.append({
                        "name": event_name,
                        "start_time": row["start_time"],
                        "motivational_readiness": readiness,
                        "se": se,
                        "pd": pd_val,
                        "anger": anger
                    })

    conn.close()

    if not sessions:
        print("Warning: No run_single_session spans found. Assuming Session 1.")
        sessions.append({
            "session_number": 1,
            "start_time": "1970-01-01T00:00:00",
            "end_time": "2999-12-31T23:59:59"
        })
        
    if not events:
        print("No patient_rolling_memory or life_event_simulator spans found with macro_shift.")
        return

    sessions = sorted(sessions, key=lambda x: x["session_number"])

    data = []
    for event in events:
        t_val = parse_time(event["start_time"])
        # Default to the last session if it happens after the last session starts
        assigned_session = sessions[-1]["session_number"] if sessions else 1
        
        for i in range(len(sessions) - 1):
            if t_val < parse_time(sessions[i+1]["start_time"]):
                assigned_session = sessions[i]["session_number"]
                break
                
        event["session_number"] = assigned_session
        data.append(event)

    # Deduplicate consecutive LLM calls (e.g. retries)
    # Since events are ordered chronologically, replacing the matched event 
    # automatically ensures we "get the last one" from the chain.
    deduped_data = []
    for ev in data:
        duplicate_idx = -1
        for i, existing in enumerate(deduped_data):
            if existing["name"] == ev["name"] and existing["session_number"] == ev["session_number"]:
                time_diff = abs((parse_time(ev["start_time"]) - parse_time(existing["start_time"])).total_seconds())
                if time_diff < 120:
                    duplicate_idx = i
                    break
        if duplicate_idx != -1:
            deduped_data[duplicate_idx] = ev
        else:
            deduped_data.append(ev)

    df = pd.DataFrame(deduped_data)
    
    # Calculate sequential X-axis for timelines
    # We want RM 1, RM 2... BSE for each session.
    def get_event_label(row, idx_within_session):
        if row["name"] == "life_event_simulator":
            return "BSE"
        else:
            return f"RM {idx_within_session + 1}"
            
    # Need to compute cumulative counts of RM events per session to assign "RM 1", "RM 2"
    df["rm_count"] = df[df["name"] == "patient_rolling_memory"].groupby("session_number").cumcount()
    
    # Create the x-labels
    event_labels = []
    for idx, row in df.iterrows():
        if row["name"] == "life_event_simulator":
            event_labels.append("BSE")
        else:
            event_labels.append(f"RM {int(row['rm_count']) + 1}")
    df["event_label"] = event_labels
    
    # x-index for plotting
    df["plot_index"] = df.groupby("session_number").cumcount() + 1
    
    # --- 1. Timeline Visualization ---
    sns.set_theme(style="whitegrid")
    sessions_found = sorted(df["session_number"].unique())
    num_sessions = len(sessions_found)
    
    fig, axes = plt.subplots(num_sessions, 1, figsize=(10, 2.5 * num_sessions), sharey=True)
    if num_sessions == 1:
        axes = [axes]
        
    fig.suptitle("Macro Shift Timeline (Rolling Memory & BSE)", fontsize=16, y=1.02)
    
    metrics = {
        "motivational_readiness": ("Motivational Readiness", "tab:blue"),
        "se": ("Self-Efficacy", "tab:green"),
        "pd": ("Problem Recognition", "tab:red"),
        "anger": ("Anger", "tab:orange")
    }
    
    for i, sess_num in enumerate(sessions_found):
        ax = axes[i]
        sess_data = df[df["session_number"] == sess_num]
        
        for col, (label, color) in metrics.items():
            offset = {"motivational_readiness": 0.06, "se": 0.02, "pd": -0.02, "anger": -0.06}[col]
            ax.plot(sess_data["plot_index"], sess_data[col] + offset, marker='o', 
                    label=label, color=color, linewidth=2, alpha=0.7)
            
        ax.set_title(f"Session {sess_num}")
        ax.set_ylabel("Macro Shift")
        
        ax.set_ylim(-2.1, 2.1)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels([
            "Major Regression (-2)", 
            "Minor Regression (-1)", 
            "No Change (0)", 
            "Minor Progress (1)", 
            "Major Progress (2)"
        ])
        
        ax.set_xticks(sess_data["plot_index"])
        ax.set_xticklabels(sess_data["event_label"])
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_timeline = out_dir / "macro_shift_timeline.png"
    plt.savefig(out_path_timeline, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Timeline plot saved to {out_path_timeline}")

    # --- 2. Pie Chart Distribution ---
    event_types = [
        ("patient_rolling_memory", "Patient Rolling Memory"),
        ("life_event_simulator", "Life Simulator (BSE)")
    ]
    metric_cols = ["motivational_readiness", "se", "pd", "anger"]
    metric_titles = ["Motivational Readiness", "Self-Efficacy", "Problem Recognition", "Anger"]
    
    fig_dist, axes_dist = plt.subplots(len(event_types), len(metric_cols), figsize=(12, 3 * len(event_types)))
    # Ensure axes_dist is 2D
    if len(event_types) == 1:
        axes_dist = [axes_dist]
        
    fig_dist.suptitle("Macro Shift Distribution by LLM Call", fontsize=16, y=1.05)
    
    colors_map = {2: "darkgreen", 1: "lightgreen", 0: "tab:gray", -1: "lightcoral", -2: "darkred"}
    labels_map = {2: "Major Progress", 1: "Minor Progress", 0: "No Change", -1: "Minor Regression", -2: "Major Regression"}
    
    for i, (ev_val, ev_label) in enumerate(event_types):
        ev_data = df[df["name"] == ev_val]
        
        for j, metric in enumerate(metric_cols):
            ax = axes_dist[i][j]
            
            if ev_data.empty:
                ax.axis('off')
                if j == 0:
                    ax.text(-1.5, 0, ev_label, rotation=90, va="center", ha="center", fontweight="bold", fontsize=12)
                continue
                
            counts = ev_data[metric].value_counts().sort_index()
            if counts.empty:
                ax.axis('off')
                continue
                
            colors = [colors_map[val] for val in counts.index]
            ax.pie(counts, labels=None, colors=colors, autopct='%1.1f%%', startangle=90, 
                   wedgeprops={'edgecolor': 'w', 'linewidth': 1})
            
            if i == 0:
                ax.set_title(metric_titles[j], fontweight="bold")
                
            if j == 0:
                ax.text(-1.5, 0, ev_label, rotation=90, va="center", ha="center", fontweight="bold", fontsize=12)
                
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=colors_map[k], label=labels_map[k]) for k in [2, 1, 0, -1, -2]]
    fig_dist.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5)
    
    plt.tight_layout()
    out_path_dist = out_dir / "macro_shift_distribution.png"
    plt.savefig(out_path_dist, dpi=300, bbox_inches='tight')
    plt.close(fig_dist)
    print(f"Distribution plot saved to {out_path_dist}")

if __name__ == "__main__":
    main()