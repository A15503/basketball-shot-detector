import matplotlib.pyplot as plt
import io
from PIL import Image

def plot_wrist_graph(wrist_y_per_frame, release_frame):
    frames = [f[0] for f in wrist_y_per_frame]
    y_vals = [f[1] for f in wrist_y_per_frame]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.plot(frames, y_vals, color='#00ff88', linewidth=2, label='Wrist Height')

    release_y = next(y for f, y in wrist_y_per_frame if f == release_frame)
    ax.axvline(x=release_frame, color='red', linestyle='--', linewidth=2, label=f'Release Frame {release_frame}')
    ax.plot(release_frame, release_y, 'ro', markersize=10)

    ax.set_xlabel('Frame', color='white')
    ax.set_ylabel('Wrist Y Position', color='white')
    ax.set_title('Wrist Arc During Shot', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.invert_yaxis()
    ax.legend(facecolor='#262730', labelcolor='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    return Image.open(buf)