import io
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")   # non-GUI backend — required for web servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TARGET_SR = 500   # must match preprocess.py


def generate_graph(audio: np.ndarray, sr: int, result: str, bpm: int) -> str:
    """
    Generate a two-panel analysis chart:
      - Top:    Mel spectrogram (colour-mapped, dB scale)
      - Bottom: Waveform (amplitude vs time)

    Returns the chart as a base64-encoded PNG string so the
    frontend can render it directly in an <img> tag:
        <img src="data:image/png;base64,{graph}" />
    """
    label_colour = "#2ecc71" if result == "normal" else "#e74c3c"
    label_text   = result.upper()

    fig = plt.figure(figsize=(9, 6), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.4, 1], hspace=0.45)

    # ── Panel 1: Mel Spectrogram ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#16213e")

    S      = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=500)
    S_db   = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax0,
        cmap="magma",
    )

    ax0.set_title(
        f"Heart Sound Analysis  ·  {label_text}  ·  {bpm} BPM",
        fontsize=11,
        fontweight="bold",
        color=label_colour,
        pad=8,
    )
    ax0.set_xlabel("Time (s)", color="#aaaacc", fontsize=9)
    ax0.set_ylabel("Frequency (Hz)", color="#aaaacc", fontsize=9)
    ax0.tick_params(colors="#aaaacc")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#333355")

    cbar = fig.colorbar(img, ax=ax0, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc", fontsize=8)

    # ── Panel 2: Waveform ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_facecolor("#16213e")

    times = np.linspace(0, len(audio) / sr, num=len(audio))
    ax1.plot(times, audio, color="#4fc3f7", linewidth=0.6, alpha=0.9)
    ax1.axhline(0, color="#555577", linewidth=0.5, linestyle="--")
    ax1.set_title("Waveform", fontsize=10, color="#aaaacc", pad=6)
    ax1.set_xlabel("Time (s)", color="#aaaacc", fontsize=9)
    ax1.set_ylabel("Amplitude", color="#aaaacc", fontsize=9)
    ax1.tick_params(colors="#aaaacc")
    ax1.set_xlim(0, times[-1])
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333355")

    # ── Encode to base64 ─────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return encoded
