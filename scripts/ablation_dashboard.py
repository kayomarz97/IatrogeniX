"""
IatrogeniX — scripts/ablation_dashboard.py
============================================
Live "All 4" Monitor for the Safety Ablation Study.
Displays a parallel track of Configs A, B, C, and D.
"""
import time, os, sys, re

LOG_A = "evaluation/LIVE_LOG.txt"          # From mass_eval.py
LOG_BCD = "evaluation/ABLATION_LIVE_LOG.txt"  # From safety_ablation.py

def clear_screen():
    print("\033[2J\033[H", end="")

def format_status(val):
    if val == "1" or val == "safe" or "✅" in str(val):
        return "\033[32mSAFE\033[0m"
    return "\033[31mUNSAFE\033[0m"

def get_last_n_lines(filepath, n=10):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return f.readlines()[-n:]

def parse_a_line(line):
    # Format: [223/1000] ✅ mmlu_medical_genetics_43 | 12.5s | Status: safe
    match = re.search(r"\[(\d+)/1000\].*?Status: (\w+)", line)
    if match:
        return match.group(1), match.group(2)
    return None, None

def parse_bcd_line(line):
    # Format: [1/100] question_id | B:1 C:1 D:0
    match = re.search(r"\[(\d+)/\d+\].*?B:(\d) C:(\d) D:(\d)", line)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    return None, None, None, None

def main():
    try:
        while True:
            clear_screen()
            print("\033[1;36m" + "="*60 + "\033[0m")
            print("\033[1;36mIATROGENIX — 4-WAY SAFETY ABLATION DASHBOARD\033[0m")
            print("\033[1;36m" + "="*60 + "\033[0m\n")

            # --- Summary Table ---
            print(f"{'Mode':<10} | {'Model':<15} | {'Safety':<10} | {'Status':<10}")
            print("-" * 60)
            print(f"{'A':<10} | {'IatrogeniX':<15} | {'ON':<10} | \033[32mActive\033[0m (PID 4055461)")
            print(f"{'B':<10} | {'IatrogeniX':<15} | {'OFF':<10} | \033[33mActive\033[0m (Ablation Runner)")
            print(f"{'C':<10} | {'Gemma-Base':<15} | {'ON':<10} | \033[33mActive\033[0m (Ablation Runner)")
            print(f"{'D':<10} | {'Gemma-Base':<15} | {'OFF':<10} | \033[33mActive\033[0m (Ablation Runner)")
            print("\n")

            # --- Live Feed ---
            lines_a = get_last_n_lines(LOG_A, 5)
            lines_bcd = get_last_n_lines(LOG_BCD, 5)

            print("\033[1;34m[LATEST INTERVENTIONS]\033[0m")
            
            # Show Mode A
            if lines_a:
                idx, status = parse_a_line(lines_a[-1])
                if idx:
                    print(f"Mode A (FT+Safe): Case {idx} -> Safety: {format_status(status)}")
            
            # Show Mode BCD
            if lines_bcd:
                # Find the last actual result line
                result_line = None
                in_progress = None
                for line in reversed(lines_bcd):
                    if "[*] Processing" in line:
                        in_progress = re.search(r"Processing (.*?) \(", line)
                        if in_progress: in_progress = in_progress.group(1)
                        if result_line: break # Found both
                    elif "B:" in line:
                        result_line = line
                        if in_progress: break # Found both

                if result_line:
                    idx, b, c, d = parse_bcd_line(result_line)
                    if idx:
                        print(f"Mode B (FT-NoSafe):  Case {idx} -> Result: {format_status(b)}")
                        print(f"Mode C (Base+Safe): Case {idx} -> Result: {format_status(c)}")
                        print(f"Mode D (Base-NoSafe): Case {idx} -> Result: {format_status(d)}")
                
                if in_progress:
                    print(f"\n\033[1;33m[!] Currently Auditing:\033[0m {in_progress} (Loading into Modes B/C/D)")
            
            print("\n" + "\033[90mPress Ctrl+C to detach (tmux session will keep running).\033[0m")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nDashboard detached.")

if __name__ == "__main__":
    main()
