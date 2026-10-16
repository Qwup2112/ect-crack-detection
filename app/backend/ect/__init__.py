"""ECT scan processing: reading TDMS raster scans, border cropping, contrast enhancement,
crack detection and rendering. Pure functions, no web or agent code - the chat agent's
tools call into this package.

Ported from the legacy tools/crack_heatmap backend (git tag legacy-app): the reading,
serpentine fix, filters, auto border crop, threshold detector, YOLO input and crack
measurement are the versions that were verified on the full dataset there.
"""
