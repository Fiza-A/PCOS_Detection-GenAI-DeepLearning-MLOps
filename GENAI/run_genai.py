"""
Execute the existing Phase 2 experiment with faculty-oriented console output.

This file runs the already-designed Phase 2 GENAI experiment.
The main purpose of this file is to make the terminal output easy
for a faculty member to understand.

Only the way results are PRINTED on the screen is changed here.
The actual generator, preprocessing, evaluation and quality-gate logic
are kept inside their existing modules.
"""



# json is used to read the saved split_metadata.json file.
import json

# sys allows us to modify Python's module search path.
import sys

# time allows us to measure how long training and the complete pipeline take.
import time

# Path makes working with folders and files easier than using raw strings.
from pathlib import Path

# __file__ = the location of this Python file.
# .resolve() gives its absolute path.
# .parent gives the folder containing this file.
# / "src" points to the GENAI/src folder.
SRC = Path(__file__).resolve().parent / "src"

# Add GENAI/src to Python's module search path.
# This allows us to import modules such as ctgan_model.py directly.
sys.path.insert(0, str(SRC))

# Import the run() function from ctgan_model.py.
# We rename it to run_ctgan so it is obvious that it runs CTGAN.
from ctgan_model import run as run_ctgan

# Import the run() function from tvae_model.py.
# Rename it to run_tvae so it is obvious that it runs TVAE.
from tvae_model import run as run_tvae

# Import the validation function.
# It checks whether the generated synthetic data has the expected structure/schema.
from data_validation import validate

# Import the numerical/categorical fidelity evaluation function.
from evaluate_fidelity import evaluate as evaluate_fidelity

# Import the association/correlation evaluation function.
from evaluate_correlation import evaluate as evaluate_correlation

# Import diversity and privacy-related evaluation.
from evaluate_diversity_privacy import evaluate as evaluate_diversity

# Import the quality-gate function.
# It decides whether the generated data is APPROVED, REVIEW_REQUIRED, etc.
from quality_gate import assess

# Import the function that compares CTGAN and TVAE.
from compare_models import compare

# Import project-level paths and configuration functions.
from phase2_data import ROOT, RAW, load_config

# Width controls how wide our terminal headings are.
# 66 characters gives us a clean-looking terminal display.
WIDTH = 66

# This function prints a large section heading.

# Example:
# [1/6] DATASET & EXPERIMENT SETUP

# title = text we want to display.
# step = optional step number such as "1/6".
def _heading(title: str, step: str | None = None) -> None:

    # If a step number exists, put it before the title.
    # Otherwise just use the title.
    label = f"[{step}] {title}" if step else title

    # Print a blank line, then the heading, then a line of dashes.
    # flush=True makes sure the output appears immediately in the terminal.
    print(f"\n{label}\n{'-' * WIDTH}", flush=True)

# This function prints information in a neat two-column format.

# Example:
# Dataset                    : PCOS_data.csv
# Total records              : 541
def _field(name: str, value: object) -> None:

    # <29 means left-align the name and reserve 29 characters for it.
    print(f"{name:<29}: {value}", flush=True)

# This function formats numerical scores.

# Example:
# 0.683245 → "0.6832"

# If the value is not a number, it returns "not available".
def _score(value: object) -> str:

    # Check whether the value is an integer or floating-point number.
    if isinstance(value, (int, float)):

        # Display numerical values with four decimal places.
        return f"{value:.4f}"

    # If it isn't numeric, display this message.
    return "not available"

# This function prints a comparison table between CTGAN and TVAE.
def _comparison_table(results: dict) -> None:

    # These are the measurements we want to display.
    labels = (
        "Numerical fidelity",
        "Categorical fidelity",
        "Association fidelity",
        "Duplicate rows",
        "Exact overlap",
        "Quality gate"
    )

    # This dictionary tells Python how to extract each measurement
    # from the large results dictionary.
    rows = {

        # Get CTGAN/TVAE numerical fidelity.
        "Numerical fidelity":
            lambda x: _score(
                x["fidelity"]["aggregate_mean_one_minus_ks"]
            ),

        # Get categorical fidelity.
        "Categorical fidelity":
            lambda x: _score(
                x["fidelity"]["aggregate_mean_js_similarity"]
            ),

        # Get association/correlation similarity.
        "Association fidelity":
            lambda x: _score(
                x["correlation"]["aggregate_association_similarity_1_minus_error"]
            ),

        # Get number of duplicate synthetic rows.
        "Duplicate rows":
            lambda x: str(
                x["diversity"]["duplicate_rows"]
            ),

        # Get exact overlap between synthetic and training records.
        "Exact overlap":
            lambda x: str(
                x["diversity"]["exact_record_overlap_training"]
            ),

        # Get the final quality-gate status.
        "Quality gate":
            lambda x: x["gate"]["status"],
    }

    # Print the table header.
    print(
        f"{'MEASURE':<29}{'CTGAN':>17}{'TVAE':>19}",
        flush=True
    )

    # Go through each measurement.
    for label in labels:

        # Print the measurement name,
        # followed by CTGAN value,
        # followed by TVAE value.
        print(
            f"{label:<29}"
            f"{rows[label](results['ctgan']):>17}"
            f"{rows[label](results['tvae']):>19}",
            flush=True
        )

    # Explain what "exact overlap" means.
    print(
        "Exact overlap reports training records; "
        "validation/test overlap counts are also saved in each privacy report.",
        flush=True
    )


# This function prints the important conclusions at the end.
def _print_findings(results: dict) -> None:

    # Print the section title.
    print("\nKEY FINDINGS", flush=True)

    # Print a separator line.
    print("-" * WIDTH, flush=True)

    # Check both generators one by one.
    for g in ("ctgan", "tvae"):

        # Get that generator's results.
        r = results[g]

        # Tell us that training, generation and evaluation completed.
        print(
            f"[OK] {g.upper()} fit, synthetic generation, and evaluation completed.",
            flush=True
        )

        # Show how many synthetic rows were generated.
        print(
            f"     Synthetic rows: {r['run']['synthetic_rows']}.",
            flush=True
        )

        # Show whether validation/test data was passed into training.
        print(
            f"     Validation/test passed to fit: "
            f"{r['run']['validation_or_test_passed_to_fit']}.",
            flush=True
        )

    # These are the three aggregate metrics we want to compare.
    for metric, title in (
        ("aggregate_mean_one_minus_ks", "numerical fidelity"),
        ("aggregate_mean_js_similarity", "categorical fidelity"),
        ("aggregate_association_similarity_1_minus_error", "association fidelity")
    ):

        # For the first two metrics, obtain values from "fidelity".
        #
        # For association similarity, obtain values from "correlation".
        left = (
            results["ctgan"]["fidelity"].get(metric)
            if metric != "aggregate_association_similarity_1_minus_error"
            else results["ctgan"]["correlation"].get(metric)
        )

        # Do the same thing for TVAE.
        right = (
            results["tvae"]["fidelity"].get(metric)
            if metric != "aggregate_association_similarity_1_minus_error"
            else results["tvae"]["correlation"].get(metric)
        )

        # Make sure both values exist and are different.
        if left is not None and right is not None and left != right:

            # Whichever value is larger is called the leader
            # for that particular metric.
            leader = "CTGAN" if left > right else "TVAE"

            # Important:
            # This explicitly says that winning one metric
            # does NOT mean the model is universally better.
            print(
                f"[INFO] {leader} has the higher aggregate {title} "
                f"value in this run; this is not a universal model ranking.",
                flush=True
            )

    # Collect the quality-gate statuses of both generators.
    statuses = [
        results[g]["gate"]["status"]
        for g in ("ctgan", "tvae")
    ]

    # If both models require review...
    if all(s == "REVIEW_REQUIRED" for s in statuses):

        # Tell the user that neither is approved.
        print(
            "[REVIEW] Neither generator is approved for unrestricted downstream use.",
            flush=True
        )

    # If at least one generator is not approved...
    elif any(s != "APPROVED" for s in statuses):

        # Tell the user that at least one requires review.
        print(
            "[REVIEW] At least one generator is not approved for downstream use.",
            flush=True
        )

    # Otherwise both were approved.
    else:

        # Print successful approval message.
        print(
            "[OK] The existing quality-gate logic approved both generators.",
            flush=True
        )


# This is the main function.
# It controls the entire GENAI experiment execution.
def main() -> dict:

    # Start a timer for the complete pipeline.
    total_start = time.perf_counter()

    # Load the project configuration.
    cfg = load_config()

    # Print the main title.
    print("=" * WIDTH)

    # This comment explains why the output uses simple ASCII characters.
    # It helps Windows terminals display the output consistently.
    # ASCII = basic characters like letters, numbers and symbols.
    # print("...") is used below for the actual title.
    print("          PCOS-MLOps - GENERATIVE AI EXECUTION")

    # Print another separator.
    print("=" * WIDTH)


    # ============================================================
    # STEP 1: DATASET
    # ============================================================

    # Display the first section heading.
    _heading("DATASET & EXPERIMENT SETUP", "1/6")

    # Read the original PCOS CSV file into a pandas DataFrame.
    raw = pd.read_csv(RAW)

    # Get the name of the target column from the configuration.
    # In this project it is "PCOS (Y/N)".
    target = cfg["target_column"]

    # Print the raw dataset filename.
    _field("Dataset", RAW.name)

    # Print number of records/rows.
    _field("Total records", len(raw))

    # Print number of columns/features.
    _field("Total features", len(raw.columns))

    # Print the target column.
    _field("Target", target)


    # ============================================================
    # STEP 2: DATA PARTITIONING
    # ============================================================

    # Print the second section heading.
    _heading("DATA PARTITIONING", "2/6")

    # Build the path to the saved split metadata.
    split_path = ROOT / "GENAI/data/processed/split_metadata.json"

    # Check whether the split metadata file actually exists.
    if not split_path.is_file():

        # If it doesn't exist, stop execution with an explanation.
        raise FileNotFoundError(
            "Partition metadata is missing. "
            "Run the configured GENAI pipeline setup before execution."
        )

    # Read the JSON file as text and convert it into a Python dictionary.
    split = json.loads(
        split_path.read_text(encoding="utf-8")
    )

    # Display number of training records.
    _field(
        "Training records",
        split["row_counts"]["train"]
    )

    # Display number of validation records.
    _field(
        "Validation records",
        split["row_counts"]["validation"]
    )

    # Display number of test records.
    _field(
        "Test records",
        split["row_counts"]["test"]
    )

    # Display how the data was split and the random seed.
    _field(
        "Split",
        f"Stratified 70/15/15 (seed {split['seed']})"
    )

    # Very important:
    # only the training partition is used to train the generators.
    _field(
        "Generator training data",
        "Training partition ONLY"
    )


    # ============================================================
    # STEP 3 AND STEP 4: CTGAN + TVAE
    # ============================================================

    # Create a tuple containing:
    # 1. the generator name
    # 2. the function that runs that generator.
    runners = (
        ("ctgan", run_ctgan),
        ("tvae", run_tvae)
    )

    # Create an empty dictionary.
    # We will store all CTGAN and TVAE results here.
    results: dict = {}

    # Create a dictionary to store training time.
    training_seconds: dict[str, float] = {}


    # Run each generator one after another.
    #
    # enumerate(..., start=3) gives:
    # CTGAN → index 3
    # TVAE  → index 4
    for index, (generator, train_fn) in enumerate(
        runners,
        start=3
    ):

        # Convert "ctgan" → "CTGAN".
        label = generator.upper()

        # Print either:
        # [3/6] CTGAN TRAINING
        # or
        # [4/6] TVAE TRAINING
        _heading(
            f"{label} TRAINING",
            f"{index}/6"
        )

        # Tell the user which model is running.
        print(
            f"Model                       : {label}",
            flush=True
        )

        # Tell the user exactly which dataset is used.
        print(
            "Training data               : train.csv ONLY",
            flush=True
        )

        # Explicitly confirm that validation/test data is not used for fitting.
        print(
            "Validation/test used?       : NO (per pipeline fit contract)",
            flush=True
        )

        # Start timer for this generator.
        stage_start = time.perf_counter()

        # Actually execute the CTGAN or TVAE run() function.
        #
        # The first returned value is ignored using "_".
        # The second value contains metadata about the run.
        _, run_metadata = train_fn()

        # Calculate how many seconds the generator stage took.
        training_seconds[generator] = (
            time.perf_counter() - stage_start
        )

        # Tell the user training completed.
        print(
            "Training status             : COMPLETED",
            flush=True
        )

        # Show number of synthetic rows generated.
        print(
            f"Synthetic records           : "
            f"{run_metadata['synthetic_rows']}",
            flush=True
        )

        # Show training duration.
        print(
            f"Training stage elapsed      : "
            f"{_elapsed(training_seconds[generator])}",
            flush=True
        )

        # Explain what the timing includes.
        print(
            "(Stage time includes fit, sampling, and model/artifact saving.)",
            flush=True
        )


        # Tell the terminal that evaluation is starting.
        print(
            f"{label} evaluation           : running",
            flush=True
        )

        # Validate the generated synthetic dataset.
        validation = validate(generator)

        # Evaluate numerical and categorical fidelity.
        fidelity = evaluate_fidelity(generator)

        # Evaluate relationships/correlations between features.
        correlation = evaluate_correlation(generator)

        # Evaluate duplicates, diversity and privacy-related distances.
        diversity = evaluate_diversity(generator)

        # Combine the evaluation results and apply the quality gate.
        gate = assess(
            generator,
            validation,
            fidelity,
            correlation,
            diversity
        )

        # Store all results for this generator.
        results[generator] = {
            "run": run_metadata,
            "validation": validation,
            "fidelity": fidelity,
            "correlation": correlation,
            "diversity": diversity,
            "gate": gate
        }

        # Tell the user evaluation finished.
        print(
            f"{label} evaluation           : COMPLETED",
            flush=True
        )


    # ============================================================
    # STEP 5: SYNTHETIC DATA EVALUATION
    # ============================================================

    # Print evaluation section heading.
    _heading(
        "SYNTHETIC DATA EVALUATION",
        "5/6"
    )

    # Tell the faculty that numerical fidelity was evaluated.
    print(
        "Numerical fidelity          : COMPLETED",
        flush=True
    )

    # Tell the faculty categorical fidelity was evaluated.
    print(
        "Categorical fidelity        : COMPLETED",
        flush=True
    )

    # Tell the faculty association analysis was performed.
    print(
        "Association analysis        : COMPLETED",
        flush=True
    )

    # Tell the faculty duplicate detection was performed.
    print(
        "Duplicate detection         : COMPLETED",
        flush=True
    )

    # Tell the faculty exact-overlap detection was performed.
    print(
        "Exact-overlap detection     : COMPLETED",
        flush=True
    )

    # Tell the faculty privacy-distance analysis was performed.
    # "Preliminary; no guarantee" is important because
    # distance analysis is NOT a formal privacy guarantee.
    print(
        "Privacy-distance diagnostic : COMPLETED (preliminary; no guarantee)",
        flush=True
    )

    # Compare CTGAN and TVAE overall.
    comparison = compare(results)

    # Print the comparison table.
    _comparison_table(results)


    # ============================================================
    # STEP 6: QUALITY GATE
    # ============================================================

    # Print the final quality-gate section.
    _heading(
        "QUALITY GATE",
        "6/6"
    )

    # Check CTGAN and TVAE separately.
    for generator in ("ctgan", "tvae"):

        # Get that generator's gate result.
        gate = results[generator]["gate"]

        # Print its status and reason.
        print(
            f"{generator.upper():<8}: "
            f"{gate['status']} - {gate['reason']}",
            flush=True
        )


    # If either CTGAN or TVAE requires review...
    if any(
        results[g]["gate"]["status"] == "REVIEW_REQUIRED"
        for g in ("ctgan", "tvae")
    ):

        # Explain what REVIEW_REQUIRED means.
        print(
            "\nQUALITY GATE MEANING",
            flush=True
        )

        print(
            "REVIEW_REQUIRED means synthetic data was generated, but the available",
            flush=True
        )

        print(
            "evidence is not sufficient for unrestricted downstream use. It does",
            flush=True
        )

        print(
            "not mean generator training failed; evidence thresholds are not defined",
            flush=True
        )

        print(
            "by the project sources, so automatic approval is not justified.",
            flush=True
        )


    # ============================================================
    # FINAL DECISION
    # ============================================================

    # Print final-decision heading.
    _heading("FINAL DECISION")

    # Print CTGAN's final status.
    print(
        f"CTGAN : {results['ctgan']['gate']['status']}",
        flush=True
    )

    # Print TVAE's final status.
    print(
        f"TVAE  : {results['tvae']['gate']['status']}",
        flush=True
    )

    # Print the overall comparison decision.
    print(
        comparison["decision"],
        flush=True
    )

    # Show CTGAN training time.
    print(
        f"CTGAN training stage time  : "
        f"{_elapsed(training_seconds['ctgan'])}",
        flush=True
    )

    # Show TVAE training time.
    print(
        f"TVAE training stage time   : "
        f"{_elapsed(training_seconds['tvae'])}",
        flush=True
    )

    # Calculate total time from the beginning of the pipeline.
    total_seconds = time.perf_counter() - total_start

    # Display total pipeline time.
    print(
        f"Total pipeline time        : "
        f"{_elapsed(total_seconds)}",
        flush=True
    )

    # Print the human-readable key findings.
    _print_findings(results)

    # Print the closing separator.
    print("\n" + "=" * WIDTH, flush=True)

    # Tell the user the experiment is finished.
    print(
        "                  EXECUTION COMPLETE"
    )

    # Print the closing separator again.
    print("=" * WIDTH, flush=True)

    # Return the complete results dictionary.
    return results


# This condition checks whether this file was executed directly.
#
# If you run:
# python GENAI/run_genai.py
#
# then __name__ becomes "__main__",
# so the main() function is executed.
if __name__ == "__main__":

    # Start the complete GENAI pipeline.
    main()