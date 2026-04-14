"""Start trading bot with stdout captured to live_output.log."""
import sys, os, subprocess

os.chdir(r'C:\Users\convo\trade')
sys.path.insert(0, r'C:\Users\convo\trade')
os.environ['PYTHONUNBUFFERED'] = '1'

log_path = os.path.join('logs', 'live_output.log')

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except:
                pass

log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"[LOGGED] Output captured to {log_path}", flush=True)

# Run the actual bot
from src.main import main
main()
