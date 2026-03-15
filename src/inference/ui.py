import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import queue

from src.inference.scheduler import IDSScheduler, load_config

class MLIDS_Dashboard(tk.Tk):
    def __init__(self, demo_mode=False):
        super().__init__()
        
        # Configure Main Window
        self.title(f"ML-Enhanced IDS Dashboard [{'DEMO' if demo_mode else 'LIVE'}]")
        self.geometry("1000x600")
        self.minsize(800, 500)
        self.configure(bg="#1E1E2E")
        
        # State variables
        self.demo_mode = demo_mode
        self.is_monitoring = False
        self.scheduler = None
        self.monitor_thread = None
        
        # UI Queue for thread-safe GUI updates
        self.ui_queue = queue.Queue()
        
        # Setup modern styling
        self._setup_styles()
        
        # Build layout
        self._build_ui()
        
        # Start queue processor
        self.after(100, self._process_queue)
        
        # Handle Window Close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        
        # Colors
        self.bg_color = "#1E1E2E"
        self.fg_color = "#CDD6F4"
        self.acc_color = "#89B4FA"
        
        # Treeview (Table) Styling
        style.configure("Treeview", 
                        background="#181825", 
                        foreground=self.fg_color, 
                        rowheight=25,
                        fieldbackground="#181825")
        
        style.map('Treeview', background=[('selected', self.acc_color)])
        
        style.configure("Treeview.Heading", 
                        background="#313244", 
                        foreground=self.fg_color, 
                        font=('Arial', 10, 'bold'))

        # Add tag configurations for colors
        # (This must be done per-instance via tree.tag_configure, but we set up the concept here)

    def _build_ui(self):
        # --- Top Header ---
        header_frame = tk.Frame(self, bg="#11111B", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        title_lbl = tk.Label(header_frame, text="🛡️ ML-Enhanced Intrusion Detection", 
                             font=("Arial", 16, "bold"), bg="#11111B", fg=self.acc_color)
        title_lbl.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Controls
        self.toggle_btn = tk.Button(header_frame, text="▶ Start Monitoring", 
                                    font=("Arial", 11, "bold"), bg="#A6E3A1", fg="#11111B",
                                    command=self.toggle_monitoring, relief=tk.FLAT, padx=15)
        self.toggle_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # --- Stats Bar ---
        stats_frame = tk.Frame(self, bg=self.bg_color)
        stats_frame.pack(fill=tk.X, pady=10, padx=20)
        
        self.lbl_processed = tk.Label(stats_frame, text="Processed: 0", font=("Arial", 11), bg=self.bg_color, fg=self.fg_color)
        self.lbl_processed.pack(side=tk.LEFT, padx=10)
        
        self.lbl_intrusions = tk.Label(stats_frame, text="Intrusions: 0", font=("Arial", 11, "bold"), bg=self.bg_color, fg="#F38BA8")
        self.lbl_intrusions.pack(side=tk.LEFT, padx=30)
        
        self.lbl_rate = tk.Label(stats_frame, text="Detection Rate: 0.0%", font=("Arial", 11), bg=self.bg_color, fg=self.fg_color)
        self.lbl_rate.pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = tk.Label(stats_frame, text="Status: IDLE", font=("Arial", 11, "italic"), bg=self.bg_color, fg="#F9E2AF")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # --- Table (Treeview) ---
        table_frame = tk.Frame(self, bg="#181825")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        columns = ("time", "rule", "severity", "src", "dst", "class", "conf")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        self.tree.heading("time", text="Time")
        self.tree.heading("rule", text="Rule Description")
        self.tree.heading("severity", text="Severity")
        self.tree.heading("src", text="Source IP")
        self.tree.heading("dst", text="Dest IP")
        self.tree.heading("class", text="ML Class")
        self.tree.heading("conf", text="Confidence")
        
        self.tree.column("time", width=80, anchor=tk.CENTER)
        self.tree.column("rule", width=300, anchor=tk.W)
        self.tree.column("severity", width=80, anchor=tk.CENTER)
        self.tree.column("src", width=120, anchor=tk.CENTER)
        self.tree.column("dst", width=120, anchor=tk.CENTER)
        self.tree.column("class", width=100, anchor=tk.CENTER)
        self.tree.column("conf", width=80, anchor=tk.CENTER)
        
        # Set row colors
        self.tree.tag_configure("critical", foreground="#F38BA8", font=("Arial", 10, "bold")) # Red
        self.tree.tag_configure("high", foreground="#FAB387") # Orange
        self.tree.tag_configure("medium", foreground="#F9E2AF") # Yellow
        self.tree.tag_configure("low", foreground="#A6E3A1") # Green
        self.tree.tag_configure("info", foreground="#94E2D5") # Teal
        self.tree.tag_configure("normal", foreground=self.fg_color) # Default text
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        self.is_monitoring = True
        self.toggle_btn.config(text="⏹ Stop Monitoring", bg="#F38BA8", fg="#11111B")
        self.lbl_status.config(text="Status: RUNNING", fg="#A6E3A1")
        
        # Initialize scheduler
        config = load_config()
        if self.demo_mode:
            config["demo_mode"] = True
            
        self.scheduler = IDSScheduler(config, ui_callback=self._handle_alerts)
        
        if not self.scheduler.setup():
            messagebox.showerror("Error", "Failed to load ML models. Please train them first.")
            self.stop_monitoring()
            return

        # Run scheduler loop in background thread
        self.monitor_thread = threading.Thread(target=self.scheduler.run_background, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        if self.scheduler:
            self.scheduler.stop()
            
        self.is_monitoring = False
        self.toggle_btn.config(text="▶ Start Monitoring", bg="#A6E3A1", fg="#11111B")
        self.lbl_status.config(text="Status: IDLE", fg="#F9E2AF")

    def _handle_alerts(self, alerts, stats):
        """Callback from scheduler thread. Puts data onto UI queue."""
        self.ui_queue.put({"alerts": alerts, "stats": stats})

    def _process_queue(self):
        """Process GUI updates in the main thread."""
        try:
            while True:
                data = self.ui_queue.get_nowait()
                
                # Update stats
                stats = data["stats"]
                self.lbl_processed.config(text=f"Processed: {stats['total_processed']}")
                self.lbl_intrusions.config(text=f"Intrusions: {stats['intrusions_detected']}")
                
                total = max(stats['total_processed'], 1)
                rate = (stats['intrusions_detected'] / total) * 100
                self.lbl_rate.config(text=f"Detection Rate: {rate:.1f}%")
                
                # Add rows to treeview
                for alert in data["alerts"]:
                    ml = alert["ml_analysis"]
                    orig = alert["original_alert"]
                    severity = alert["severity"]
                    time_str = os.path.basename(alert["timestamp"]).split("T")[1][:8]
                    
                    # Determine row tag
                    if not ml["is_intrusion"]:
                        tag = "normal"
                    else:
                        tag = severity
                        
                    values = (
                        time_str,
                        f"[{orig['rule_id']}] {orig['rule_description'][:40]}",
                        severity.upper(),
                        orig['source_ip'],
                        orig['destination_ip'],
                        ml['ensemble_class'],
                        f"{ml['confidence']:.1%}"
                    )
                    
                    # Insert at top
                    self.tree.insert("", 0, values=values, tags=(tag,))
                    
                # Keep only last 1000 items
                children = self.tree.get_children()
                if len(children) > 1000:
                    for child in children[1000:]:
                        self.tree.delete(child)
                        
        except queue.Empty:
            pass
            
        # Re-schedule check
        self.after(100, self._process_queue)

    def _on_closing(self):
        self.stop_monitoring()
        self.destroy()

if __name__ == "__main__":
    demo = "--demo" in sys.argv
    app = MLIDS_Dashboard(demo_mode=demo)
    app.mainloop()
