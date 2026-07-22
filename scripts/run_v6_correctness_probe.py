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
from traj_geom.shapes.synthetic import make_counting_task

def run_probe():
    model, tok = load_model()
    
    # Scale reasoning depth (number of operations)
    depths = [2, 4, 6, 8, 10, 12, 14, 16]
    num_steps = 64
    
    results = []
    
    print("Running V6 Correctness Probe with counting task...")
    for d in depths:
        # We try a few seeds to ensure the answer isn't trivially guessed early
        for seed in range(3):
            task = make_counting_task(n_ops=d, seed=seed)
            ans = str(task["answer"])
            # encode the answer correctly
            ans_token_id = tok.encode(ans, add_special_tokens=False)[-1]
            
            # Run extraction
            out = extract_trajectory(model, tok, task["prompt"], num_steps=num_steps, seed=0, return_logits=True)
            logits = out.get("logits")
            
            if logits is None:
                print("Warning: Logits extraction failed.")
                return
                
            correct_at_step = -1
            for step in range(num_steps):
                pred = logits[step].argmax()
                if pred == ans_token_id:
                    correct_at_step = step
                    break
                    
            results.append({
                "depth": d,
                "seed": seed,
                "correct_at_step": correct_at_step,
                "target": ans
            })
            
            print(f"Depth: {d} | Seed: {seed} | Target: {ans} | Correct At Step: {correct_at_step if correct_at_step != -1 else 'Never'}")
        
    df = pd.DataFrame(results)
    df.to_csv("results/v6_correctness_probe.csv", index=False)
    print("\nResults saved to results/v6_correctness_probe.csv")

if __name__ == "__main__":
    run_probe()
