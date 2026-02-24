# High-Speed Washer Core Thickness Inspection (OK/NOK)

This repository contains a high-speed machine vision system for washer-type parts.  
The system measures core thickness regions on the part with high precision and performs automatic OK/NOK classification based on tolerance limits.

## Key Features
- Live acquisition via **Baumer NeoAPI**
- Real-time thickness measurement from **6 predefined regions (No 1..6)**
- Robust binary processing:
  - Largest connected component extraction
  - Hole filling
  - Border clearing (inner region isolation)
- Edge-based distance estimation:
  - Canny edges
  - Two-line fitting (upper/lower edges)
  - Perpendicular distance computation
- Adjustable parameters from UI:
  - mm/px scale
  - threshold
  - canny thresholds
  - LSL/USL tolerance limits
- Large **OK / NOT OK** visual indicator

## Demo
- The application provides a live video view and prints measurements (No 1..6) with OK/NOK status.
- Global result is **OK** only if all 6 measurements are within tolerance.

## Requirements
- Windows (recommended, due to camera SDK)
- Python 3.10+ (tested with typical industrial setups)
- Baumer NeoAPI installed (Camera Explorer / GenTL components)

### Python Packages
- numpy
- opencv-python
- pillow
- neoapi
- scipy (optional; improves hole-filling)

## Installation
```bash
pip install numpy opencv-python pillow
# Optional
pip install scipy
