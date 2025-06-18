import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, group_delay
import pandas as pd

# ---------- 信号生成と移動平均 ----------
def generate_signal(fs, freq, noise_amp, duration=5.0):
    t = np.arange(0, duration, 1.0/fs)
    signal = np.sin(2 * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    return t, signal + noise, signal

def moving_average(signal, N):
    return np.convolve(signal, np.ones(N)/N, mode='same')

# ---------- フィルタ特性 ----------
def plot_frequency_response(N, fs):
    h = np.ones(N) / N
    w, H = freqz(h, worN=1024, fs=fs)

    plt.figure()
    plt.title("Amplitude Response")
    plt.plot(w, 20*np.log10(np.abs(H)))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid()

    plt.figure()
    plt.title("Phase Response")
    plt.plot(w, np.unwrap(np.angle(H)))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [rad]")
    plt.grid()

def plot_group_delay(N, fs):
    b = np.ones(N) / N
    w, gd = group_delay((b, 1), fs=fs)

    plt.figure()
    plt.title("Group Delay")
    plt.plot(w, gd)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Delay [samples]")
    plt.grid()

def plot_step_response(N):
    h = np.ones(N) / N
    step = np.ones(100)
    y = np.convolve(step, h)
    plt.figure()
    plt.title("Step Response")
    plt.plot(y)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("移動平均フィルタ 設計&適用")
        root.geometry("300x200")

        # 入力フォーム
        self.create_input("サンプリング周波数 [Hz]", "1000", 0)
        self.create_input("正弦波周波数 [Hz]", "5", 1)
        self.create_input("ノイズ振幅", "0.5", 2)
        self.create_input("移動平均サンプル数", "5", 3)

        # 実行ボタン
        ttk.Button(root, text="フィルタ設計&フィルタ適用", command=self.run).grid(column=1, row=4, columnspan=2, pady=10)
        ttk.Button(root, text="フィルタ後信号保存", command=self.save_csv).grid(column=1, row=5, columnspan=2, pady=5)
        ttk.Button(root, text="フィルタ特性表示", command=self.show_filter_characteristics).grid(column=1, row=6, columnspan=2, pady=5)

        self.t = None
        self.original = None
        self.filtered = None

    def create_input(self, label, default, row):
        ttk.Label(self.root, text=label).grid(column=0, row=row, sticky=tk.W)
        entry = ttk.Entry(self.root)
        entry.insert(0, default)
        entry.grid(column=2, row=row)
        setattr(self, f'entry_{row}', entry)

    def run(self):
        fs = int(self.entry_0.get())
        freq = float(self.entry_1.get())
        noise_amp = float(self.entry_2.get())
        N = int(self.entry_3.get())

        self.t, noisy, self.original = generate_signal(fs, freq, noise_amp)
        self.filtered = moving_average(noisy, N)
        self.noisy = noisy

        plt.figure()
        plt.title("Noisy vs Filtered")
        plt.plot(self.t, noisy, label="Noisy Signal", color="g", linewidth=2)
        plt.plot(self.t, self.filtered, label=f"Filtered (N={N})", color="b", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.ylim([-4, 4])
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()

        plt.figure()
        plt.title("Filtered vs Org Sine")
        plt.plot(self.t, self.filtered, label=f"Filtered (N={N})", color="b", linewidth=2)
        plt.plot(self.t, self.original, label="Org Sine", color='#f781bf', linewidth=2.5)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.ylim([-4, 4])
        plt.legend(loc='upper left')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def save_csv(self):
        if self.t is None or self.filtered is None:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.DataFrame({
                "Time [s]": self.t,
                "Org Sine": self.original,
                "Filtered Signal": self.filtered,
                "Noisy": self.noisy
            })
            df.to_csv(file_path, index=False)

    def show_filter_characteristics(self):
        fs = int(self.entry_0.get())
        N = int(self.entry_3.get())

        plot_frequency_response(N, fs)
        plot_group_delay(N, fs)
        plot_step_response(N)
        plt.show()

# ---------- 実行 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
