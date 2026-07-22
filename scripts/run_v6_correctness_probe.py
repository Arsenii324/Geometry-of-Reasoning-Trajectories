"""Probe correctness timing using the V6 logit-extraction hook.

Runs a small set of representative non-noisy examples through the heavy V6 hook
to empirically measure how many recurrent unrolls it actually takes to predict
the correct answer, mapping True Logical Depth to `num_steps`.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scripts._common import load_model
from traj_geom.extraction.hook import extract_trajectory
from traj_geom.shapes.synthetic import make_three_scale_task

def run_probe():
    model, tok = load_model()
    
    # We will just scale active_len while holding neutral/irrelevant at 0
    # to find the baseline 'steps-to-correctness' for pure reasoning.
    active_lens = [1, 2, 3, 4, 5]
    num_steps = 32
    
    results = []
    
    print("Running V6 Correctness Probe...")
    for act in active_lens:
        # Generate clean task without noise
        task = make_three_scale_task(irrelevant_len=0, neutral_len=0, active_len=act, seed=42)
        ans = str(task["answer"])
        ans_token_id = tok.encode(ans, add_special_tokens=False)[-1]
        
        # Run heavy V6 extraction
        out = extract_trajectory(model, tok, task["prompt"], num_steps=num_steps, seed=0, return_logits=True)
        logits = out.get("logits")
        
        if logits is None:
            print("Warning: Logits extraction failed. Check architecture compatibility.")
            break
            
        # logits shape: [num_steps, vocab_size]
        # Find the first step where the argmax is the correct answer
        correct_at_step = -1
        
        for step in range(num_steps):
            pred = logits[step].argmax()
            if pred == ans_token_id:
                correct_at_step = step
                break
                
        results.append({
            "active_len": act,
            "correct_at_step": correct_at_step,
            "target": ans
        })
        
        print(f"Active Len: {act} | Correct At Step: {correct_at_step if correct_at_step != -1 else 'Never'}")
        
    df = pd.DataFrame(results)
    df.to_csv("results/v6_correctness_probe.csv", index=False)
    print("\nResults saved to results/v6_correctness_probe.csv")

if __name__ == "__main__":
    run_probe()
