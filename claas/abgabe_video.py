"""Abgabe-Video aus mehreren Clips zusammensetzen (Claas, 20.11.).
  python abgabe_video.py <ziel.mp4> <clip1.mp4> <clip2.mp4> ...
Clips in der angegebenen Reihenfolge, fehlende werden uebersprungen."""
import os
import sys

try:
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy.editor import VideoFileClip, concatenate_videoclips

out, parts = sys.argv[1], [p for p in sys.argv[2:] if os.path.exists(p)]
print("Clips:", *parts, sep="\n  ")
clips = [VideoFileClip(p) for p in parts]
concatenate_videoclips(clips, method="compose").write_videofile(out, fps=30, logger=None)
print("ABGABE_OK", out, round(sum(c.duration for c in clips), 1), "s")
