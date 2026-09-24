import os
import subprocess
import logging
import json
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# List of target directory indices (e.g., [1] means process 'transcripts/1')
# Use ["*"] to scan all subdirectories
TARGET_INDICES = ["*"]  # Example: process directories 'transcripts/9' and 'transcripts/16'

# Optional: Only process files whose filename contains ANY of these substrings.
# Set to an empty list ([]) to disable filtering.
TARGET_SUBSTRINGS = ["0cc2ab65842f46bd835687b2af29621a"]

# Optional: Exclude files whose filename contains ANY of these substrings.
# Set to an empty list ([]) to disable filtering.
EXCLUDE_SUBSTRINGS = ["2cb47c81dc5878a6d0e50ea9aeb0927a","63e69ff5513d1aa34a14c7d78597f60c","fe693cbd61684ed0b2e3096ed5574324"]

# Maximum number of concurrent automisc_coder scripts to run
MAX_CONCURRENT_RUNS = 6

# Base directory for transcripts
TRANSCRIPTS_BASE_DIR = Path("transcripts_2")

# Path to the coder script
CODER_SCRIPT_PATH = Path("src_automisc/automisc_coder.py")

# Default LLM configuration
DEFAULT_MODEL_NAME = "google/gemma-4-31b-it"
DEFAULT_CONTEXT_TURNS = 1
DEFAULT_MAX_TOKENS = 500
DEFAULT_PROVIDER_KWARGS = {
    "order": ["deepinfra"],
    "ignore": ["siliconflow", "parasail"],
    "sort": "price"
}

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def process_file(
    txt_file_path: Path,
    model: str,
    context_turns: int,
    provider_kwargs: dict,
    max_tokens: Optional[int] = None,
    force: bool = False,
    output_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None
) -> bool:
    """
    Runs the automisc_coder script on a single file.
    Returns True if successful, False if skipped or failed.
    """
    # Extract model short name (e.g., 'llama-3.3-70b-instruct' from 'meta-llama/llama-3.3-70b-instruct')
    model_short_name = model.split("/")[-1].replace(":", "_").replace(".", "_")

    # Define the output path in the result folder (equal to input dir by default)
    output_file_name = f"{txt_file_path.stem}_{model_short_name}.yaml"
    target_output_dir = output_dir if output_dir is not None else base_dir
    if target_output_dir is not None:
        if base_dir is not None:
            try:
                rel_dir = txt_file_path.relative_to(base_dir).parent
                target_dir = target_output_dir / rel_dir
            except ValueError:
                target_dir = txt_file_path.parent if target_output_dir == base_dir else target_output_dir
        else:
            target_dir = target_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = target_dir / output_file_name
    else:
        output_file_path = txt_file_path.parent / output_file_name

    # Check if the output file already exists
    if output_file_path.exists() and not force:
        logger.info(f"Skipping: Output file already exists for {txt_file_path.name} -> {output_file_path}")
        return False

    logger.info(f"Starting automisc_coder for: {txt_file_path}")
    
    try:
        cmd = [
            "python",
            str(CODER_SCRIPT_PATH),
            "--input-file", str(txt_file_path),
            "--output-file", str(output_file_path),
            "--model", str(model),
            "--context-turns", str(context_turns),
            "--provider-kwargs", json.dumps(provider_kwargs)
        ]
        if max_tokens is not None:
            cmd.extend(["--max-tokens", str(max_tokens)])
            
        # Run the coder script as a subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully finished processing {txt_file_path.name}. Output saved to: {output_file_path}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed processing {txt_file_path.name}. Exit code: {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {txt_file_path.name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process transcripts in batch using AutoMISC")
    parser.add_argument(
        "--input-dir", "--input-folder", "--transcripts-dir",
        dest="input_dir",
        type=str,
        default=str(TRANSCRIPTS_BASE_DIR),
        help=f"Base directory containing transcripts (default: {TRANSCRIPTS_BASE_DIR})"
    )
    parser.add_argument(
        "--output-dir", "--results-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save output files (default: equal to --input-dir)"
    )
    parser.add_argument("--indices", nargs="+", default=None, help="Target indices (e.g. '*' or '1 20 89')")
    parser.add_argument("--target-substrings", nargs="+", default=None, help="Only process files matching any substring")
    parser.add_argument("--exclude-substrings", nargs="+", default=None, help="Exclude files matching any substring")
    parser.add_argument("--max-concurrent-runs", type=int, default=MAX_CONCURRENT_RUNS, help="Maximum concurrent runs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name to use")
    parser.add_argument("--context-turns", type=int, default=DEFAULT_CONTEXT_TURNS, help="Number of context turns")
    parser.add_argument("--provider-kwargs", type=str, default=json.dumps(DEFAULT_PROVIDER_KWARGS), help="JSON-encoded string of provider kwargs")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum number of output tokens")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing output files")
    parser.add_argument("--dry-run", action="store_true", help="List discovered files to process without executing")
    
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    target_indices = args.indices if args.indices is not None else TARGET_INDICES
    target_substrings = args.target_substrings if args.target_substrings is not None else TARGET_SUBSTRINGS
    exclude_substrings = args.exclude_substrings if args.exclude_substrings is not None else EXCLUDE_SUBSTRINGS
    max_concurrent_runs = args.max_concurrent_runs
    force = args.force
    dry_run = args.dry_run

    model = args.model
    context_turns = args.context_turns
    try:
        provider_kwargs = json.loads(args.provider_kwargs)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse provider-kwargs JSON: {e}")
        return
    max_tokens = args.max_tokens

    if not CODER_SCRIPT_PATH.exists():
        logger.error(f"Coder script not found at {CODER_SCRIPT_PATH}. Please verify the path.")
        return

    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {base_dir}")
        return

    files_to_process = []

    # Check if base_dir itself directly contains .txt files
    for item in sorted(base_dir.iterdir()):
        if item.is_file() and item.suffix == '.txt':
            if target_substrings and not any(sub in item.name for sub in target_substrings):
                continue
            if exclude_substrings and any(sub in item.name for sub in exclude_substrings):
                continue
            files_to_process.append(item)

    # Check for subdirectories in base_dir
    if "*" in target_indices or "*" in map(str, target_indices):
        logger.info(f"Wildcard '*' detected in target indices. Scanning all subdirectories in {base_dir}.")
        target_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    else:
        target_dirs = [base_dir / str(idx) for idx in target_indices]

    # Gather all target files from subdirectories
    for target_dir in target_dirs:
        if not target_dir.exists() or not target_dir.is_dir():
            if "*" not in target_indices: # Only warn if we were looking for a specific index
                logger.warning(f"Target directory does not exist or is not a directory: {target_dir}")
            continue

        # Find all .txt files in the directory
        for item in sorted(target_dir.iterdir()):
            if item.is_file() and item.suffix == '.txt':
                if target_substrings and not any(sub in item.name for sub in target_substrings):
                    continue
                if exclude_substrings and any(sub in item.name for sub in exclude_substrings):
                    continue
                files_to_process.append(item)

    if not files_to_process:
        logger.info("No .txt files found to process in the specified target indices.")
        return

    logger.info(f"Found {len(files_to_process)} .txt files in total.")

    if dry_run:
        logger.info(f"[Dry Run] Result folder: {output_dir}")
        logger.info("[Dry Run] Discovered candidate files:")
        model_short_name = model.split("/")[-1].replace(":", "_").replace(".", "_")
        for f in files_to_process:
            try:
                rel_dir = f.relative_to(base_dir).parent
                tgt = output_dir / rel_dir
            except ValueError:
                tgt = f.parent if output_dir == base_dir else output_dir
            expected_output = tgt / f"{f.stem}_{model_short_name}.yaml"
            status = "EXISTS (will skip unless --force)" if expected_output.exists() else "PENDING"
            logger.info(f"  [{status}] {f} -> {expected_output}")
        return

    logger.info(f"Starting execution with a maximum of {max_concurrent_runs} concurrent runs.")

    # Execute concurrently
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_concurrent_runs) as executor:
        # Submit all tasks to the executor
        future_to_file = {
            executor.submit(
                process_file,
                f,
                model,
                context_turns,
                provider_kwargs,
                max_tokens,
                force,
                output_dir,
                base_dir
            ): f
            for f in files_to_process
        }
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                success = future.result()
                if success:
                    processed_count += 1
            except Exception as exc:
                logger.error(f"{file_path.name} generated an exception: {exc}")

    logger.info(f"Batch run complete. Successfully processed (or re-processed) {processed_count} files.")

if __name__ == "__main__":
    main()
