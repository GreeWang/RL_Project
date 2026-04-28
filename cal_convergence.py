import pandas as pd
import numpy as np
import json
from pathlib import Path

def calc_convergence(episode_log_path: str, summary_json_path: str, window: int = 100, success_thr: float = 0.9):
    df = pd.read_csv(episode_log_path)
    df_train = df[df["phase"] == "train"].sort_values("episode").reset_index(drop=True)
    
    if len(df_train) < window:
        return {
            "converged_episode": None,
            "converged": False,
            "final_success": None,
            "final_steps": None,
            "message": "Not enough episodes for window"
        }
    
    with open(summary_json_path, "r") as f:
        meta = json.load(f)
    shortest = meta.get("shortest_path_len", 5)
    step_thr = shortest + 2
    
    df_train["succ_ma"] = df_train["success"].rolling(window, min_periods=1).mean()
    
    succ_steps = df_train[df_train["success"] == 1]["steps"]
    aligned_steps = pd.Series(np.nan, index=df_train.index)
    aligned_steps.loc[succ_steps.index] = succ_steps
    df_train["step_ma"] = aligned_steps.rolling(window, min_periods=1).mean()
    
    cond = (df_train["succ_ma"] >= success_thr) & (df_train["step_ma"] <= step_thr) & (df_train["step_ma"].notna())
    
    if cond.any():
        conv_ep = int(df_train[cond].index[0] + 1)
        final_succ = df_train["succ_ma"].iloc[-1]
        final_step = df_train["step_ma"].iloc[-1]
        msg = f"Converged at ep {conv_ep} | MA Success: {final_succ:.3f} | MA Steps: {final_step:.1f} (Thr: ≤{step_thr})"
        return {
            "converged_episode": conv_ep,
            "converged": True,
            "final_success": final_succ,
            "final_steps": final_step,
            "message": msg
        }
    
    return {
        "converged_episode": None,
        "converged": False,
        "final_success": df_train["succ_ma"].iloc[-1],
        "final_steps": df_train["step_ma"].iloc[-1],
        "message": f"Did not converge within {len(df_train)} episodes"
    }

if __name__ == "__main__":
    base = Path("results")
    output_dir = Path("convergence_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for log in sorted(base.rglob("episode_log.csv")):
        summary = log.parent / "summary.json"
        if not summary.exists():
            continue
            
        res = calc_convergence(str(log), str(summary))
        exp_path = log.parent.parent.name
        seed = log.parent.name
        
        res["experiment"] = exp_path
        res["seed"] = seed
        results.append(res)
        
        print(f"{exp_path}/{seed}: {res['message']}")
        
    if results:
        df_res = pd.DataFrame(results)
        
        csv_path = output_dir / "convergence_summary.csv"
        df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n CSV results saved in: {csv_path}")
        
        json_path = output_dir / "convergence_summary.json"
        df_res.to_json(json_path, orient="records", indent=2)
        print(f" JSON results saved in : {json_path}")
    else:
        print("\n No valid episode logs found for convergence analysis.")
