"""Generate an interactive HTML dashboard showing GOOGL structural observation zones."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pa_engine as pe  # noqa: E402
from scripts.plot_wave_chart import load_ohlcv  # noqa: E402


def build_googl_zones_html(output_path: Path | None = None) -> Path:
    """Create a self-contained HTML page visualizing GOOGL left/right observation zones."""
    csv_path = PROJECT_ROOT / "data" / "GOOGL.csv"
    output_path = output_path or (PROJECT_ROOT / "results" / "GOOGL_buy_zones.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(csv_path)
    ema200_full = df["Close"].ewm(span=200, adjust=False).mean()
    sub = df.tail(90).copy()
    ema200_sub = ema200_full.tail(90)

    swings = pe.extract_swings(sub)
    chan = pe.detect_price_channel(sub, swings)
    last_dt = sub.index[-1]
    current_price = float(sub["Close"].iloc[-1])
    ema_last = float(ema200_sub.iloc[-1])

    x_start = sub.index[0] - pd.Timedelta(days=2)
    dt_l11 = last_dt + pd.offsets.BDay(2)
    dt_f13 = last_dt + pd.offsets.BDay(4)
    x_end = dt_f13 + pd.Timedelta(days=10)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )

    # 1. Candlesticks & Volume
    fig.add_trace(
        go.Candlestick(
            x=sub.index,
            open=sub["Open"],
            high=sub["High"],
            low=sub["Low"],
            close=sub["Close"],
            name="GOOGL OHLC",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
            increasing_fillcolor="#26A69A",
            decreasing_fillcolor="#EF5350",
        ),
        row=1,
        col=1,
    )
    vol_colors = [
        "rgba(38, 166, 154, 0.6)" if c >= o else "rgba(239, 83, 80, 0.6)"
        for o, c in zip(sub["Open"], sub["Close"])
    ]
    fig.add_trace(
        go.Bar(x=sub.index, y=sub["Volume"], name="Volume", marker_color=vol_colors, showlegend=False),
        row=2,
        col=1,
    )

    # 2. 200 EMA & Descending Channel
    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=ema200_sub,
            mode="lines",
            name=f"200 EMA (${ema_last:.2f})",
            line=dict(color="#1F77B4", width=2.0),
        ),
        row=1,
        col=1,
    )

    if chan.get("channel_type") == "descending" and chan.get("upper_rail") is not None:
        n_win = min(len(sub), 30)
        slope = float(chan.get("slope") or 0.0)
        for label, end_val, color, dash in [
            ("下降通道上轨", float(chan["upper_rail"]), "#E67E22", "dash"),
            ("下降通道下轨", float(chan["lower_rail"]), "#16A085", "dash"),
        ]:
            start_val = end_val - slope * (n_win - 1)
            fig.add_trace(
                go.Scatter(
                    x=[sub.index[-n_win], sub.index[-1]],
                    y=[start_val, end_val],
                    mode="lines",
                    name=f"{label} (${end_val:.2f})",
                    line=dict(color=color, width=1.6, dash=dash),
                ),
                row=1,
                col=1,
            )

    # 3. Observation Zones
    zone_specs = [
        {
            "y0": 276.92,
            "y1": 285.68,
            "fill": "rgba(46, 204, 113, 0.16)",
            "line": "#27AE60",
            "label": "情景A 左侧回踩观察区 $276.92-$285.68 (不破$276.92止跌)",
            "y_text": 281.30,
        },
        {
            "y0": 296.85,
            "y1": 303.37,
            "fill": "rgba(52, 152, 219, 0.16)",
            "line": "#2980B9",
            "label": "情景B1 右侧突破观察区 $296.85-$303.37 (站稳通道上轨)",
            "y_text": 300.10,
        },
        {
            "y0": 322.66,
            "y1": 327.89,
            "fill": "rgba(142, 68, 173, 0.14)",
            "line": "#8E44AD",
            "label": "情景B2 中线转强确认区 $322.66-$327.89 (回补缺口+200 EMA)",
            "y_text": 325.25,
        },
    ]
    zone_start = sub.index[-30]
    for z in zone_specs:
        fig.add_shape(
            type="rect",
            x0=zone_start,
            x1=x_end,
            y0=z["y0"],
            y1=z["y1"],
            fillcolor=z["fill"],
            line=dict(color=z["line"], width=1.2, dash="dot"),
            layer="below",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x_end - pd.Timedelta(days=1),
            y=z["y_text"],
            text=f"<b>{z['label']}</b>",
            showarrow=False,
            xanchor="right",
            font=dict(size=10, color=z["line"]),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=z["line"],
            borderwidth=1,
            borderpad=3,
            row=1,
            col=1,
        )

    # Hard invalidation & lower extension reference
    fig.add_hline(
        y=276.92,
        line_dash="dash",
        line_color="#C0392B",
        line_width=1.5,
        annotation_text=" 左侧证伪底线 (C浪前低): $276.92 ",
        annotation_position="left",
        annotation_font=dict(size=10, color="#C0392B"),
        annotation_bgcolor="rgba(253, 237, 236, 0.92)",
        annotation_bordercolor="#C0392B",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=254.49,
        line_dash="dot",
        line_color="#922B21",
        line_width=1.2,
        annotation_text=" 跌破后下方延伸参考: $252.61-$254.49 ",
        annotation_position="left",
        annotation_font=dict(size=10, color="#922B21"),
        annotation_bgcolor="rgba(255, 255, 255, 0.88)",
        annotation_bordercolor="#922B21",
        row=1,
        col=1,
    )

    # Time windows
    for dt_w, label, color, y_dom in [
        (dt_l11, f"L11 ({dt_l11.strftime('%m-%d')})", "#2980B9", 0.04),
        (dt_f13, f"F13 ({dt_f13.strftime('%m-%d')})", "#8E44AD", 0.13),
    ]:
        x_iso = dt_w.strftime("%Y-%m-%d")
        fig.add_vline(x=x_iso, line_dash="dash", line_color=color, line_width=1.3, row=1, col=1)
        fig.add_annotation(
            x=x_iso,
            y=y_dom,
            xref="x",
            yref="y domain",
            text=f"<b>时间窗 {label}</b>",
            showarrow=False,
            xanchor="left",
            xshift=4,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=690,
        margin=dict(l=60, r=40, t=55, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_xaxes(range=[x_start, x_end], rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>GOOGL 结构观察与潜在买入情景对照</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: #f5f7fa;
      color: #1f2937;
      margin: 0;
      padding: 20px;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    .header {{
      background: #ffffff;
      border-radius: 10px;
      padding: 18px 22px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header h1 {{ margin: 0; font-size: 20px; color: #111827; }}
    .subtitle {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
    .badge-warn {{
      background: #fef2f2;
      color: #b91c1c;
      border: 1px solid #fecaca;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 16px 18px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      border-top: 4px solid #6b7280;
    }}
    .card.left {{ border-top-color: #27ae60; }}
    .card.right {{ border-top-color: #2980b9; }}
    .card.mid {{ border-top-color: #8e44ad; }}
    .card.risk {{ border-top-color: #c0392b; }}
    .card h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
    .price-range {{ font-size: 20px; font-weight: 700; margin-bottom: 8px; }}
    .card p {{ margin: 4px 0; font-size: 13px; line-height: 1.55; color: #4b5563; }}
    .chart-box {{
      background: #ffffff;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .disclaimer {{
      margin-top: 12px;
      font-size: 12px;
      color: #6b7280;
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>GOOGL — 结构观察与潜在买入情景对照（最新收盘 ${current_price:.2f}）</h1>
        <div class="subtitle">基于 pa_engine 多周期结构分析 | 200 EMA: ${ema_last:.2f} | 敏感时间窗: L11 (11-13) &amp; F13 (11-17)</div>
      </div>
      <div class="badge-warn">当前系统偏向：HTF -1 / MTF -1（偏空缩量反抽，暂未触发做多信号）</div>
    </div>

    <div class="cards">
      <div class="card left">
        <h3>情景 A：左侧回踩观察区（高风险）</h3>
        <div class="price-range" style="color:#27ae60;">$276.92 – $285.68</div>
        <p><b>支撑构成：</b>11-10 小时抬高低点 ($285.68) + C浪前低 ($276.92)。</p>
        <p><b>观察条件：</b>回踩不破 $276.92 且小时级出现放量止跌结构。</p>
      </div>
      <div class="card right">
        <h3>情景 B1：右侧突破确认区（稳健）</h3>
        <div class="price-range" style="color:#2980b9;">$296.85 – $303.37</div>
        <p><b>阻力构成：</b>近期小时前高 ($296.33-$296.85) + 下降通道上轨 ($303.37)。</p>
        <p><b>观察条件：</b>放量站稳该区间上方，解除短期下降通道压制。</p>
      </div>
      <div class="card mid">
        <h3>情景 B2：中线完全转强区</h3>
        <div class="price-range" style="color:#8e44ad;">$322.66 – $327.89</div>
        <p><b>阻力构成：</b>10-29 缺口上沿 ($322.66) + 200 EMA (${ema_last:.2f}) + 0.618 共振。</p>
        <p><b>观察条件：</b>完全回补缺口并站回 200 EMA，确认中期趋势反转。</p>
      </div>
      <div class="card risk">
        <h3>证伪底线与下方风险参考</h3>
        <div class="price-range" style="color:#c0392b;">跌破 $276.92</div>
        <p><b>结构含义：</b>C浪或新一轮空头下跌延续，左侧观察失效。</p>
        <p><b>下方延伸位：</b>动态通道下轨 / 1.272 扩展 ($252.61 – $254.49)。</p>
      </div>
    </div>

    <div class="chart-box">
      {chart_html}
    </div>
    <div class="disclaimer">注：本页面所有价位、浪型与时间窗口仅供技术结构学习与证伪观察，不构成任何投资或买卖建议。</div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = build_googl_zones_html()
    print(f"Saved zones HTML to: {out}")
