from pathlib import Path
from src.common.app_utils import PROJECT_ROOT

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"PROJECT_ROOT type: {type(PROJECT_ROOT)}")
print(f"Expected path: /Users/auslei/dev/property_adviser")
print(f"Is correct: {PROJECT_ROOT == Path('/Users/auslei/dev/property_adviser')}")

# Also test that Overview.py can import app_utils properly
print("\nTesting import from Overview.py:")
try:
    import sys
    sys.path.append(str(PROJECT_ROOT))
    from app.Overview import PROJECT_ROOT as OVERVIEW_PROJECT_ROOT
    print(f"Successfully imported from Overview.py")
    print(f"OVERVIEW_PROJECT_ROOT: {OVERVIEW_PROJECT_ROOT}")
except Exception as e:
    print(f"Import error: {e}")