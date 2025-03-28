
def get_app_path():
    import os
    import sys
    
    if getattr(sys, 'frozen', False):
        # We are running in a bundle
        base_dir = sys._MEIPASS
    else:
        # We are running in a normal Python environment
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return base_dir

# Add app path to sys.path to find modules
import sys
sys.path.insert(0, get_app_path())
