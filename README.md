# VeichiTrainer — M1 (Manual labeling + ArUco scale)

A simple desktop tool to:
- Open a photo of your bench
- Draw **boxes** for components and **measurement lines** for trunking/DIN rail/busbar
- Detect **ArUco** corner tags to convert **pixels → millimetres**
- Save per-image annotations as JSON (for later training + costing)

---

## Install (macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
