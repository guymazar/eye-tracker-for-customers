import cv2
import numpy as np
import os

class GazeAnalyzer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.heatmap = None
        self.frame_count = 0

        # Ensure output directories exist
        os.makedirs(os.path.join(self.output_dir, 'gaze_heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)

    def update_heatmap(self, eyes):
        if self.heatmap is None:
            self.heatmap = np.zeros((480, 640), dtype=np.float32)  # Assuming 640x480 resolution

        for (ex, ey, ew, eh) in eyes:
            self.heatmap[ey:ey+eh, ex:ex+ew] += 1

    def save_results(self):
        # Normalize the heatmap
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # Save the heatmap
        heatmap_path = os.path.join(self.output_dir, 'gaze_heatmaps', f'heatmap_{self.frame_count}.png')
        cv2.imwrite(heatmap_path, heatmap_colored)

        # Log the heatmap path
        log_path = os.path.join(self.output_dir, 'logs', 'gaze_log.txt')
        with open(log_path, 'a') as log_file:
            log_file.write(f'Frame {self.frame_count}: {heatmap_path}\n')

        self.frame_count += 1 