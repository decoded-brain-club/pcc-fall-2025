import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src/loaders")) # Add src/loaders directory to path

from tuh_loader import TUHLoader # type: ignore

def main():
    pass

if __name__ == "__main__":
    main()