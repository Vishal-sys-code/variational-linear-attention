import json
from pathlib import Path

fp = Path(r'd:\Github\variational-linear-attention\notebooks\Second_Version_VLA\12e_Hetero_vs_Uniform.ipynb')
nb = json.load(open(fp, 'r', encoding='utf-8'))

new_setup = (
    "import math, time, gc, json\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "import torch.utils.checkpoint as ckpt\n"
    "\n"
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "\n"
    "# ── GPU compatibility guard ───────────────────────────────────────────\n"
    "if DEVICE == 'cuda':\n"
    "    _cap = torch.cuda.get_device_capability()\n"
    "    _gpu = torch.cuda.get_device_name()\n"
    "    _min = (7, 0)  # PyTorch >=2.4 dropped sm_60 (P100)\n"
    "    print(f'GPU: {_gpu}  CUDA capability: {_cap[0]}.{_cap[1]}')\n"
    "    if _cap < _min:\n"
    "        raise RuntimeError(\n"
    "            f'\\n{\"=\"*60}\\n'\n"
    "            f'  GPU {_gpu} (sm_{_cap[0]}{_cap[1]}) is NOT supported by this PyTorch.\\n'\n"
    "            f'  Minimum required: sm_{_min[0]}{_min[1]}\\n\\n'\n"
    "            f'  FIX: In Kaggle -> Settings -> Accelerator -> select \"GPU T4 x2\".\\n'\n"
    "            f'  T4 (sm_75) is supported and faster for this workload.\\n'\n"
    "            f'{\"=\"*60}'\n"
    "        )\n"
    "\n"
    "OUT    = Path('/kaggle/working/nb12e')\n"
    "for s in ['plots','logs']: (OUT/s).mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'font.family':'DejaVu Serif','font.size':11,\n"
    "    'axes.spines.top':False,'axes.spines.right':False,\n"
    "    'axes.grid':True,'grid.alpha':0.25,'figure.dpi':150,\n"
    "})\n"
    "C = {'uniform':'#55EFC4','hetero':'#00B894','deltanet':'#FDCB6E'}\n"
    "\n"
    "print(f'Device: {DEVICE}  Torch: {torch.__version__}')"
)

for c in nb['cells']:
    if c.get('id') == 'setup':
        c['source'] = new_setup
        break

json.dump(nb, open(fp, 'w', encoding='utf-8'), indent=1)
print("Notebook patched: added GPU compatibility guard to setup cell.")
