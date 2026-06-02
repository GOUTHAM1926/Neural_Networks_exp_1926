import os

source_file = "/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/error_propagation_analysis_fast_div_mod_algo_understanding_experiment.md"
target_file = "/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/understanding reduction_module.md"

with open(source_file, "r") as f:
    experiment_data = f.read()

# We will append the experiment data right after section 6.7 in the target document.
# Or just at the very end of the file as an appendix to keep it clean.
with open(target_file, "a") as f:
    f.write("\n\n### 6.7.1 Detailed Error Propagation Experiment Data (The Proof)\n\n")
    f.write("Below are the exhaustive calculations demonstrating exactly how the Rounding Error accumulates for various deep learning tensor dimensions and linear indices.\n\n")
    f.write(experiment_data)
