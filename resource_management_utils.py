import psutil,os,json
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
    if 'run' in config and isinstance(config['run'], list):
        config['run'] = [int(x) for x in config['run']]
    if 'train_run' in config and isinstance(config['train_run'], list):
        config['train_run'] = [int(x) for x in config['train_run']]
    if 'test_run' in config and isinstance(config['test_run'], list):
        config['test_run'] = [int(x) for x in config['test_run']]
    if 'cv_alphas' in config and isinstance(config['cv_alphas'], list):
        config['cv_alphas'] = [float(x) for x in config['cv_alphas']]
    
    return config

def save_args_to_json(args, output_dir=None):
    """Save all arguments to a JSON file"""
    if output_dir is None:
        output_dir = args.outdir if args.outdir else "."
    
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