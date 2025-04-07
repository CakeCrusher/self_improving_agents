#!/usr/bin/env python
"""
Example script to run the ephemeral validator from the examples directory.

This script demonstrates how to validate a policy using ephemeral validation,
which collects state-action data, loads the latest policy checkpoint,
generates samples, and evaluates them without tracking.
"""
import json
import os
import sys
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Make sure the self_improving_agents package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_improving_agents.validation import EphemeralValidator

def main():
    """Run ephemeral validation process on example data."""
    load_dotenv()
    
    # Get the path to inputs.json relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_path = os.path.join(script_dir, "inputs.json")
    
    # Load inputs
    print("\nLoading input examples...")
    with open(inputs_path, "r") as file:
        inputs = json.load(file)
    
    # Shuffle the inputs for a diverse sample
    random.seed(42)  # For reproducibility
    random.shuffle(inputs)
    
    print(f"Loaded {len(inputs)} input examples")
    
    # Create validator
    print("\nInitializing validator...")
    validator = EphemeralValidator()
    
    # Run validation
    print("\nRunning validation process...")
    # Use a start date that's likely to have data (adjust as needed)
    start_date = datetime.now() - timedelta(days=7)
    
    sample_size = 10
    print(f"Using {sample_size} samples for validation")
    
    avg_score, evals_df = validator.run_validation(
        start_date=start_date,
        inputs=inputs[:20],  # Use first 20 inputs (will be limited by validator)
        evaluator_name="formatting_classify",
        limit=sample_size,
    )
    
    print(f"\n==== Validation Results ====")
    print(f"Average Score: {avg_score:.2f} / 5.0")
    print("\n==== Sample Evaluations ====")
    
    # Format and display a few results
    for i, (label, explanation) in enumerate(zip(evals_df["label"], evals_df["explanation"])):
        if i >= 3:  # Just show first 3 for brevity
            break
        print(f"\nExample {i+1}:")
        print(f"Score: {label}")
        print(f"Reasoning: {explanation[:200]}...")  # Truncate long explanations
    
    # Save results to file
    results_path = os.path.join(script_dir, "validation_results.json")
    evals_df.to_json(results_path, orient="records")
    print(f"\nDetailed results saved to: {results_path}")
    
    return avg_score, evals_df

if __name__ == "__main__":
    main() 