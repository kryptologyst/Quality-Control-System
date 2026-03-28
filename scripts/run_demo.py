#!/usr/bin/env python3
"""
Quality Control System - Demo Launcher

Simple script to launch the Streamlit demo application.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo application."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Error: Demo application not found at {demo_path}")
        sys.exit(1)
    
    print("🚀 Launching Quality Control System Demo...")
    print("📊 Open your browser to view the interactive dashboard")
    print("⚠️  Remember: This is for educational purposes only")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
