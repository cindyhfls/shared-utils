import psutil,os,json,sys
from datetime import datetime

def print_memory_usage(message=""):
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"{datetime.now().strftime('%H:%M:%S')} | {mem_gb:.2f} GB {message}")

def load_args_from_json(json_file):
    """Load arguments from JSON file"""
    with open(json_file, 'r') as f:
        config = json.load(f)

    # Convert back to proper types where needed
    # Handle special types that might be lost in JSON serialization
    if 'cv_alphas' in config and isinstance(config['cv_alphas'], list):
        config['cv_alphas'] = [float(x) for x in config['cv_alphas']]
    if 'run' in config and isinstance(config['run'], list):
        config['run'] = [int(x) for x in config['run']]
    if 'subj' in config and isinstance(config['subj'], list):
        config['subj'] = [int(x) for x in config['subj']]
    return config

def save_args_to_json(args, output_dir=None):
    """Save all arguments to a JSON file"""
    if output_dir is None:
        output_dir = args.outdir if args.outdir else "."

    output_dir = os.path.join(output_dir,'config')
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{args.stage}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert args to dictionary
    args_dict = vars(args).copy()

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(args_dict, f, indent=2, default=str)

    print(f"Configuration saved to: {filepath}")
    return filepath

def merge_args(json_dict, cli_ns):
    merged = json_dict.copy()
    cli_dict = vars(cli_ns)

    # args actually typed on CLI
    typed = set(
        a.lstrip('-').replace('-', '_')
        for a in sys.argv[1:]
        if a.startswith('-')
    )

    for key, value in cli_dict.items():
        if key in typed:   # only override if actually typed
            merged[key] = value
        else: # if it is not typed, replace it only if it does not exist already
            if key not in merged:
                merged[key] = value
    return merged
