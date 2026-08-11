#!/usr/bin/env python3
"""
Generate Interactive HTML Results Dashboard for Dual-Cue Feature Fusion GPU Training
Produces outputs/LOTA_Training_Results.html with Chart.js curves, GPU hardware specs, and test metrics.
"""

import json
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.logger import get_logger

logger = get_logger("dashboard_generator")


def generate_results_dashboard():
    output_html = root_path / "outputs" / "LOTA_Training_Results.html"
    
    # 10-Epoch Anti-Overfitting Full GPU Capacity Training Trajectory
    epochs_data = [
        {"epoch": 1, "train_loss": 0.6067, "train_acc": 0.6770, "train_auc": 0.7581, "val_loss": 0.4190, "val_acc": 0.8155, "val_auc": 0.8905, "action": "Saved Checkpoint (137s)"},
        {"epoch": 2, "train_loss": 0.2899, "train_acc": 0.9120, "train_auc": 0.9719, "val_loss": 0.3343, "val_acc": 0.8550, "val_auc": 0.9307, "action": "Saved Checkpoint (113s)"},
        {"epoch": 3, "train_loss": 0.1793, "train_acc": 0.9778, "train_auc": 0.9976, "val_loss": 0.3616, "val_acc": 0.8520, "val_auc": 0.9293, "action": "Patience 1/5 (110s)"},
        {"epoch": 4, "train_loss": 0.1548, "train_acc": 0.9895, "train_auc": 0.9994, "val_loss": 0.3576, "val_acc": 0.8615, "val_auc": 0.9301, "action": "Patience 2/5 (67s)"},
        {"epoch": 5, "train_loss": 0.1453, "train_acc": 0.9955, "train_auc": 0.9998, "val_loss": 0.3265, "val_acc": 0.8680, "val_auc": 0.9387, "action": "RECORD HIGH CHECKPOINT (66s)"},
        {"epoch": 6, "train_loss": 0.1370, "train_acc": 0.9987, "train_auc": 1.0000, "val_loss": 0.3489, "val_acc": 0.8605, "val_auc": 0.9360, "action": "Patience 1/5 (109s)"},
        {"epoch": 7, "train_loss": 0.1372, "train_acc": 0.9988, "train_auc": 1.0000, "val_loss": 0.3320, "val_acc": 0.8625, "val_auc": 0.9365, "action": "Patience 2/5 (66s)"},
        {"epoch": 8, "train_loss": 0.1347, "train_acc": 0.9993, "train_auc": 1.0000, "val_loss": 0.3469, "val_acc": 0.8535, "val_auc": 0.9342, "action": "Patience 3/5 (66s)"},
        {"epoch": 9, "train_loss": 0.1338, "train_acc": 0.9995, "train_auc": 1.0000, "val_loss": 0.3306, "val_acc": 0.8640, "val_auc": 0.9380, "action": "Patience 4/5 (66s)"},
        {"epoch": 10, "train_loss": 0.1335, "train_acc": 0.9997, "train_auc": 1.0000, "val_loss": 0.3559, "val_acc": 0.8560, "val_auc": 0.9351, "action": "Early Stopping Triggered (66s)"},
    ]
    
    # Test Set Metrics (2,000 images)
    test_metrics = {
        "loss": 0.3063,
        "accuracy": 0.8750,
        "precision": 0.8827,
        "recall": 0.8650,
        "f1_score": 0.8737,
        "roc_auc": 0.9441,
        "tn": 885,
        "fp": 115,
        "fn": 135,
        "tp": 865
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual-Cue Classifier — 100% GPU Capacity Results Dashboard</title>
    <meta name="description" content="GPU Training and evaluation results for Dual-Cue Feature Fusion AI-Generated Image Classifier on NVIDIA GeForce RTX 3050 Laptop GPU.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #060a13;
            --bg-secondary: #0d1321;
            --bg-card: rgba(16, 24, 48, 0.65);
            --glass-border: rgba(100, 160, 255, 0.12);
            --glass-border-hover: rgba(100, 180, 255, 0.25);
            --accent-cyan: #00d4ff;
            --accent-blue: #4f8cff;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-orange: #fb923c;
            --text-main: #e8ecf4;
            --text-muted: #7a8baa;
            --gradient-hero: linear-gradient(135deg, #0d1321 0%, #1a1040 50%, #0d1321 100%);
            --gradient-accent: linear-gradient(135deg, #00d4ff, #4f8cff, #a855f7);
            --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.4);
            --shadow-glow: 0 0 40px rgba(0, 212, 255, 0.08);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Outfit', sans-serif; background: var(--bg-primary); color: var(--text-main); min-height: 100vh; line-height: 1.6; overflow-x: hidden; }}

        .bg-anim {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; overflow: hidden; }}
        .bg-anim .orb {{ position: absolute; border-radius: 50%; filter: blur(120px); opacity: 0.15; animation: float 20s ease-in-out infinite; }}
        .bg-anim .orb:nth-child(1) {{ width: 600px; height: 600px; background: var(--accent-cyan); top: -10%; left: -5%; }}
        .bg-anim .orb:nth-child(2) {{ width: 500px; height: 500px; background: var(--accent-purple); top: 50%; right: -10%; }}

        .container {{ position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 2.5rem 1.5rem; }}

        header {{
            background: var(--gradient-hero);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            margin-bottom: 2.5rem;
            box-shadow: var(--shadow-card), var(--shadow-glow);
            position: relative;
            overflow: hidden;
        }}

        .title-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: var(--accent-cyan);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 1.25rem;
        }}

        h1 {{
            font-size: 2.75rem;
            font-weight: 800;
            background: var(--gradient-accent);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            line-height: 1.15;
        }}

        .subtitle {{ color: var(--text-muted); font-size: 1.15rem; max-width: 800px; }}

        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; margin-bottom: 2.5rem; }}
        .stat-card {{ background: var(--bg-card); border: 1px solid var(--glass-border); border-radius: 18px; padding: 1.5rem; backdrop-filter: blur(12px); transition: all 0.3s ease; }}
        .stat-card:hover {{ transform: translateY(-4px); border-color: var(--glass-border-hover); box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5); }}

        .stat-label {{ color: var(--text-muted); font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }}
        .stat-value {{ font-size: 2.25rem; font-weight: 800; color: #fff; line-height: 1.1; }}
        .stat-value.cyan {{ color: var(--accent-cyan); }}
        .stat-value.emerald {{ color: var(--accent-emerald); }}
        .stat-value.purple {{ color: var(--accent-purple); }}
        .stat-value.rose {{ color: var(--accent-rose); }}

        .stat-desc {{ color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem; }}

        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(540px, 1fr)); gap: 1.75rem; margin-bottom: 2.5rem; }}
        .chart-card {{ background: var(--bg-card); border: 1px solid var(--glass-border); border-radius: 20px; padding: 1.75rem; backdrop-filter: blur(12px); }}
        .chart-title {{ font-size: 1.2rem; font-weight: 700; color: var(--text-main); margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.6rem; }}

        .table-card {{ background: var(--bg-card); border: 1px solid var(--glass-border); border-radius: 20px; padding: 1.75rem; backdrop-filter: blur(12px); margin-bottom: 2.5rem; }}

        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: rgba(255, 255, 255, 0.03); color: var(--text-muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; padding: 1rem; border-bottom: 1px solid var(--glass-border); }}
        td {{ padding: 1rem; border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.95rem; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

        .badge-best {{ background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); color: var(--accent-emerald); padding: 0.25rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}

        .cm-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
        .cm-box {{ padding: 1.5rem; border-radius: 14px; text-align: center; }}
        .cm-tn {{ background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); }}
        .cm-fp {{ background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); }}
        .cm-fn {{ background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); }}
        .cm-tp {{ background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3); }}
        .cm-num {{ font-size: 2rem; font-weight: 800; margin-top: 0.25rem; }}

        footer {{ text-align: center; color: var(--text-muted); font-size: 0.9rem; padding: 2rem 0; border-top: 1px solid var(--glass-border); }}
    </style>
</head>
<body>

<div class="bg-anim">
    <div class="orb"></div>
    <div class="orb"></div>
</div>

<div class="container">
    <header>
        <div class="title-badge">⚡ 100% GPU Capacity (TF32 + Batch Size 64) — RTX 3050</div>
        <h1>Dual-Cue Anti-Overfitting GPU Results</h1>
        <p class="subtitle">Dual ResNet-50 Feature Fusion with 15.2M Trainable Parameters, 10x Weight Decay (1e-2), 0.6 Dropout, and 0.05 Label Smoothing on 10,000 Benchmark Images (dataset10000).</p>
    </header>

    <!-- Top Key Metrics Cards -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Test Accuracy</div>
            <div class="stat-value emerald">{test_metrics['accuracy']*100:.2f}%</div>
            <div class="stat-desc">Surged from 51.45% baseline (+36.05%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Test ROC-AUC</div>
            <div class="stat-value cyan">{test_metrics['roc_auc']:.4f}</div>
            <div class="stat-desc">Surged from 0.5354 baseline (+0.4087)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Epoch Speed</div>
            <div class="stat-value purple">66.5s</div>
            <div class="stat-desc">16x GPU Speedup via TF32 & PCIe DMA</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Test Loss</div>
            <div class="stat-value rose">{test_metrics['loss']:.4f}</div>
            <div class="stat-desc">Reduced from 0.6913 baseline</div>
        </div>
    </div>

    <!-- Chart Cards -->
    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-title">📊 Anti-Overfitting Accuracy & ROC-AUC Trajectory</div>
            <canvas id="accChart" height="260"></canvas>
        </div>
        <div class="chart-card">
            <div class="chart-title">📉 Anti-Overfitting Training vs Validation Loss</div>
            <canvas id="lossChart" height="260"></canvas>
        </div>
    </div>

    <!-- Confusion Matrix & Hardware Specs -->
    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-title">🎯 Test Confusion Matrix (2,000 Images)</div>
            <div class="cm-grid">
                <div class="cm-box cm-tn">
                    <div class="stat-label">True Negatives (Real)</div>
                    <div class="cm-num emerald">{test_metrics['tn']}</div>
                </div>
                <div class="cm-box cm-fp">
                    <div class="stat-label">False Positives</div>
                    <div class="cm-num rose">{test_metrics['fp']}</div>
                </div>
                <div class="cm-box cm-fn">
                    <div class="stat-label">False Negatives</div>
                    <div class="cm-num rose">{test_metrics['fn']}</div>
                </div>
                <div class="cm-box cm-tp">
                    <div class="stat-label">True Positives (AI)</div>
                    <div class="cm-num cyan">{test_metrics['tp']}</div>
                </div>
            </div>
        </div>
        <div class="chart-card">
            <div class="chart-title">💻 100% GPU Capacity Hardware Configuration</div>
            <div style="margin-top: 0.5rem;">
                <p style="margin-bottom: 0.6rem;"><strong>Architecture:</strong> Dual-Cue Feature Fusion (2x ResNet-50)</p>
                <p style="margin-bottom: 0.6rem;"><strong>Parameter Count:</strong> 15,224,833 Trainable (Stem, Layer1, Layer2 Frozen)</p>
                <p style="margin-bottom: 0.6rem;"><strong>Anti-Overfitting:</strong> Dropout(0.6) + Weight Decay (1e-2) + Label Smoothing (0.05)</p>
                <p style="margin-bottom: 0.6rem;"><strong>GPU Capacity:</strong> Batch Size 64 + TF32 Matrix Math + Pinned Memory DMA</p>
                <p style="margin-bottom: 0.6rem;"><strong>Epoch Speedup:</strong> 66.5s per epoch (16x Acceleration on RTX 3050 GPU)</p>
            </div>
        </div>
    </div>

    <!-- Epoch Progress Table -->
    <div class="table-card">
        <div class="chart-title">📜 Full 100% GPU Capacity Training Progression</div>
        <table>
            <thead>
                <tr>
                    <th>Epoch</th>
                    <th>Train Loss</th>
                    <th>Train Acc</th>
                    <th>Train AUC</th>
                    <th>Val Loss</th>
                    <th>Val Acc</th>
                    <th>Val AUC</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
"""

    for ep in epochs_data:
        badge_cls = "badge-best" if "RECORD" in ep["action"] or "Saved" in ep["action"] else ""
        html_content += f"""
                <tr>
                    <td><strong>Epoch {ep['epoch']}</strong></td>
                    <td>{ep['train_loss']:.4f}</td>
                    <td>{ep['train_acc']*100:.2f}%</td>
                    <td>{ep['train_auc']:.4f}</td>
                    <td>{ep['val_loss']:.4f}</td>
                    <td><strong>{ep['val_acc']*100:.2f}%</strong></td>
                    <td><strong>{ep['val_auc']:.4f}</strong></td>
                    <td><span class="{badge_cls}">{ep['action']}</span></td>
                </tr>
"""

    html_content += f"""
            </tbody>
        </table>
    </div>

    <footer>
        <p>Dual-Cue AI-Generated Image Detection Project — ICCV 2025 Architecture Specification</p>
    </footer>
</div>

<script>
    const labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Accuracy Chart
    new Chart(document.getElementById('accChart'), {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
                {{
                    label: 'Train Accuracy',
                    data: [67.70, 91.20, 97.78, 98.95, 99.55, 99.87, 99.88, 99.93, 99.95, 99.97],
                    borderColor: '#4f8cff',
                    backgroundColor: 'rgba(79, 140, 255, 0.1)',
                    tension: 0.3,
                    fill: true
                }},
                {{
                    label: 'Val Accuracy',
                    data: [81.55, 85.50, 85.20, 86.15, 86.80, 86.05, 86.25, 85.35, 86.40, 85.60],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.3,
                    fill: true
                }},
                {{
                    label: 'Val ROC-AUC',
                    data: [89.05, 93.07, 92.93, 93.01, 93.87, 93.60, 93.65, 93.42, 93.80, 93.51],
                    borderColor: '#00d4ff',
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false
                }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#e8ecf4' }} }} }},
            scales: {{
                x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#7a8baa' }} }},
                y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#7a8baa' }} }}
            }}
        }}
    }});

    // Loss Chart
    new Chart(document.getElementById('lossChart'), {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
                {{
                    label: 'Train Loss',
                    data: [0.6067, 0.2899, 0.1793, 0.1548, 0.1453, 0.1370, 0.1372, 0.1347, 0.1338, 0.1335],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    tension: 0.3,
                    fill: true
                }},
                {{
                    label: 'Val Loss',
                    data: [0.4190, 0.3343, 0.3616, 0.3576, 0.3265, 0.3489, 0.3320, 0.3469, 0.3306, 0.3559],
                    borderColor: '#fb923c',
                    backgroundColor: 'rgba(251, 146, 60, 0.1)',
                    tension: 0.3,
                    fill: true
                }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#e8ecf4' }} }} }},
            scales: {{
                x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#7a8baa' }} }},
                y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#7a8baa' }} }}
            }}
        }}
    }});
</script>

</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Successfully generated updated 100% GPU Capacity HTML results dashboard at {output_html}")

if __name__ == "__main__":
    generate_results_dashboard()
