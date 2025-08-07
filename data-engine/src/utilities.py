import time
import threading

def run_with_loading_animation(func, *args, process_name="Processing", file_name=""):
    """
    Run a function with a loading animation and elapsed time display.
    
    Args:
        func: The function to execute
        *args: Arguments to pass to the function
        process_name: Name of the process for display (default: "Processing")
        file_name: File name for display (default: "")
    """
    start_time = time.time()
    loading_event = threading.Event()
    loading_event.set()
    
    def loading_animation():
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while loading_event.is_set():
            elapsed = time.time() - start_time
            display_name = f" {file_name}" if file_name else ""
            print(f"\r{spinner[i % len(spinner)]} {process_name}:{display_name} - Elapsed: {elapsed:.1f}s", end="", flush=True)
            time.sleep(0.1)
            i += 1
    
    # Start loading animation in background thread
    animation_thread = threading.Thread(target=loading_animation, daemon=True)
    animation_thread.start()
    
    try:
        result = func(*args)
        return result
    finally:
        loading_event.clear()
        elapsed = time.time() - start_time
        display_name = f" {file_name}" if file_name else ""
        print(f"\r✓ {process_name}:{display_name} - Completed in {elapsed:.1f}s")
        time.sleep(0.1)  # Brief pause to see completion message