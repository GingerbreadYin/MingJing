"""Configure and start the single-process Gateway inside Compose."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.launch import prepare_environment


def main():
    port = int(os.environ.get('RAGOPT_PORT', '8033'))
    subprocess.run([sys.executable, str(ROOT / 'scripts/configure.py'), '--port', str(port)],
                   cwd=ROOT, check=True)
    prepare_environment()
    import uvicorn

    uvicorn.run('app.gateway.app:app', host='0.0.0.0', port=port, workers=1)


if __name__ == '__main__':
    main()
