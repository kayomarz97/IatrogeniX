#!/usr/bin/python3
import time
import os
import sys
import re

LOG_FILE = "evaluation/LIVE_LOG.txt"
TOTAL_CASES = 1000

def get_eta():
    if not os.path.exists(LOG_FILE):
        return "Waiting for log..."
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    
    if not lines:
        return "Log empty..."
    
    # Extract [X/1000] and timestamps
    # Assuming log format: [X/1000] ✅ ... | 24.5s | ...
    # We will estimate based on the last 10 entries
    progress_pattern = re.compile(r"\[(\d+)/1000\]")
    time_pattern = re.compile(r"\|\s+(\d+\.\d+)s\s+\|")
    
    completed = 0
    times = []
    
    for line in lines[-50:]:
        m_prog = progress_pattern.search(line)
        m_time = time_pattern.search(line)
        if m_prog:
            completed = int(m_prog.group(1))
        if m_time:
            times.append(float(m_time.group(1)))
            
    if not times or completed == 0:
        return "Calculating ETA..."
    
    avg_time = sum(times) / len(times)
    remaining = TOTAL_CASES - completed
    eta_seconds = remaining * avg_time
    
    m, s = divmod(int(eta_seconds), 60)
    h, m = divmod(m, 60)
    
    return f"Progress: {completed}/{TOTAL_CASES} | Avg: {avg_time:.1f}s | ETA: {h:d}h {m:02d}m {s:02d}s"

def monitor():
    print("\033[1;36m--- IatrogeniX Real-Time Clinical Audit Dashboard ---\033[0m")
    print("Press Ctrl+C to stop monitoring.\n")
    
    # We will tail the file but print the ETA at the top
    last_pos = 0
    while True:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                f.seek(last_pos)
                new_data = f.read()
                if new_data:
                    # Colorize
                    new_data = new_data.replace("✅", "\033[32m✅\033[0m")
                    new_data = new_data.replace("⚠️", "\033[33m⚠️\033[0m")
                    new_data = new_data.replace("🚨", "\033[31m🚨\033[0m")
                    new_data = new_data.replace("Status: safe", "\033[32mStatus: safe\033[0m")
                    new_data = new_data.replace("Status: warning", "\033[33mStatus: warning\033[0m")
                    new_data = new_data.replace("Status: blocked", "\033[31mStatus: blocked\033[0m")
                    
                    sys.stdout.write(new_data)
                    sys.stdout.flush()
                    last_pos = f.tell()
                    
                    # Print ETA line (overwriting)
                    eta = get_eta()
                    sys.stdout.write(f"\r\033[1;33m{eta}\033[0m\n")
                    sys.stdout.flush()
                    
        time.sleep(2)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
