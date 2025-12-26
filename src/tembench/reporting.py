"""HTML report generation for TempoBench."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil


def get_system_info() -> dict:
    """Gather system information for reproducibility."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "hostname": platform.node(),
    }
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info["cpu_freq_mhz"] = round(cpu_freq.current, 1)
    except Exception:
        pass
    return info


def _generate_summary_table_html(df: pd.DataFrame) -> str:
    """Generate an HTML table from the summary DataFrame."""
    if df.empty:
        return "<p>No summary data available.</p>"
    
    html = ['<table class="summary-table">']
    html.append('<thead><tr>')
    for col in df.columns:
        html.append(f'<th>{col}</th>')
    html.append('</tr></thead>')
    html.append('<tbody>')
    for _, row in df.iterrows():
        html.append('<tr>')
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                html.append(f'<td>{val:.3f}</td>')
            else:
                html.append(f'<td>{val}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)


def _generate_stats_cards(df: pd.DataFrame) -> str:
    """Generate statistics cards for the report."""
    cards = []
    
    if 'wall_ms_median' in df.columns:
        fastest = df['wall_ms_median'].min()
        slowest = df['wall_ms_median'].max()
        avg = df['wall_ms_median'].mean()
        cards.append(f'''
        <div class="stat-card">
            <div class="stat-value">{fastest:.2f}ms</div>
            <div class="stat-label">Fastest</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{slowest:.2f}ms</div>
            <div class="stat-label">Slowest</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg:.2f}ms</div>
            <div class="stat-label">Average</div>
        </div>
        ''')
    
    if 'peak_rss_mb_median' in df.columns:
        max_mem = df['peak_rss_mb_median'].max()
        cards.append(f'''
        <div class="stat-card">
            <div class="stat-value">{max_mem:.2f}MB</div>
            <div class="stat-label">Peak Memory</div>
        </div>
        ''')
    
    total_runs = len(df)
    cards.append(f'''
    <div class="stat-card">
        <div class="stat-value">{total_runs}</div>
        <div class="stat-label">Total Configurations</div>
    </div>
    ''')
    
    return '\n'.join(cards)


def generate_report(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    fits_csv: Optional[Path] = None,
    chart_html: Optional[Path] = None,
    title: str = "TempoBench Report",
    output_path: Optional[Path] = None,
) -> str:
    """Generate a comprehensive HTML report.
    
    Args:
        summary_csv: Path to summary CSV file
        runs_jsonl: Optional path to raw JSONL runs
        fits_csv: Optional path to complexity fits CSV
        chart_html: Optional path to pre-generated chart HTML
        title: Report title
        output_path: If provided, save the report to this path
        
    Returns:
        The generated HTML as a string
    """
    df = pd.read_csv(summary_csv)
    system_info = get_system_info()
    
    # Load fits if available
    fits_html = ""
    if fits_csv and fits_csv.exists():
        fits_df = pd.read_csv(fits_csv)
        fits_html = f'''
        <section class="fits-section">
            <h2>Complexity Analysis</h2>
            <p>Best-fit complexity models determined via AIC selection:</p>
            {_generate_summary_table_html(fits_df)}
        </section>
        '''
    
    # Load chart if available
    chart_embed = ""
    if chart_html and chart_html.exists():
        chart_content = chart_html.read_text()
        # Extract just the vegaEmbed part if it's a full HTML
        if '<script>' in chart_content and 'vegaEmbed' in chart_content:
            chart_embed = f'''
            <section class="chart-section">
                <h2>Performance Charts</h2>
                <div id="vis-container">
                    {chart_content}
                </div>
            </section>
            '''
        else:
            chart_embed = f'''
            <section class="chart-section">
                <h2>Performance Charts</h2>
                <iframe src="{chart_html.name}" style="width:100%;height:500px;border:none;"></iframe>
            </section>
            '''
    
    # Generate raw runs summary if available
    runs_summary = ""
    if runs_jsonl and runs_jsonl.exists():
        rows = []
        with runs_jsonl.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            total_runs = len(rows)
            ok_count = sum(1 for r in rows if r.get('status') == 'ok')
            failed_count = sum(1 for r in rows if r.get('status') == 'failed')
            timeout_count = sum(1 for r in rows if r.get('status') == 'timeout')
            runs_summary = f'''
            <section class="runs-summary">
                <h2>Run Statistics</h2>
                <div class="stat-cards">
                    <div class="stat-card success">
                        <div class="stat-value">{ok_count}</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-value">{timeout_count}</div>
                        <div class="stat-label">Timeouts</div>
                    </div>
                    <div class="stat-card error">
                        <div class="stat-value">{failed_count}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_runs}</div>
                        <div class="stat-label">Total Runs</div>
                    </div>
                </div>
            </section>
            '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 12px;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header .subtitle {{
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        
        section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        section h2 {{
            color: var(--primary-color);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .stat-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: var(--bg-color);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid var(--primary-color);
        }}
        
        .stat-card.success {{
            border-left-color: var(--success-color);
        }}
        
        .stat-card.warning {{
            border-left-color: var(--warning-color);
        }}
        
        .stat-card.error {{
            border-left-color: var(--error-color);
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-color);
        }}
        
        .stat-label {{
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .summary-table th,
        .summary-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .summary-table th {{
            background: var(--bg-color);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .summary-table tr:hover {{
            background: var(--bg-color);
        }}
        
        .system-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }}
        
        .system-info-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px dashed var(--border-color);
        }}
        
        .system-info-label {{
            color: #64748b;
        }}
        
        .system-info-value {{
            font-weight: 500;
        }}
        
        .chart-section iframe,
        .chart-section .vega-embed {{
            width: 100%;
            min-height: 450px;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #64748b;
            font-size: 0.875rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            header {{
                padding: 2rem 1rem;
            }}
            
            header h1 {{
                font-size: 1.75rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="subtitle">Generated by TempoBench on {system_info['timestamp'][:10]}</div>
        </header>
        
        <section class="overview">
            <h2>Performance Overview</h2>
            <div class="stat-cards">
                {_generate_stats_cards(df)}
            </div>
        </section>
        
        {runs_summary}
        
        {chart_embed}
        
        <section class="summary-section">
            <h2>Detailed Results</h2>
            {_generate_summary_table_html(df)}
        </section>
        
        {fits_html}
        
        <section class="system-section">
            <h2>System Information</h2>
            <div class="system-info">
                <div>
                    <div class="system-info-item">
                        <span class="system-info-label">Platform</span>
                        <span class="system-info-value">{system_info['platform']}</span>
                    </div>
                    <div class="system-info-item">
                        <span class="system-info-label">Python</span>
                        <span class="system-info-value">{system_info['python_version']}</span>
                    </div>
                    <div class="system-info-item">
                        <span class="system-info-label">Processor</span>
                        <span class="system-info-value">{system_info['processor'] or 'N/A'}</span>
                    </div>
                </div>
                <div>
                    <div class="system-info-item">
                        <span class="system-info-label">CPU Cores</span>
                        <span class="system-info-value">{system_info['cpu_count_physical']} physical / {system_info['cpu_count_logical']} logical</span>
                    </div>
                    <div class="system-info-item">
                        <span class="system-info-label">Memory</span>
                        <span class="system-info-value">{system_info['memory_total_gb']} GB</span>
                    </div>
                    <div class="system-info-item">
                        <span class="system-info-label">Architecture</span>
                        <span class="system-info-value">{system_info['architecture']}</span>
                    </div>
                </div>
            </div>
        </section>
        
        <footer>
            <p>Generated by TempoBench - Language-agnostic benchmarking orchestrator</p>
            <p>Report timestamp: {system_info['timestamp']}</p>
        </footer>
    </div>
</body>
</html>'''
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
    
    return html


def compare_summaries(
    current_csv: Path,
    baseline_csv: Path,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Compare current results against a baseline.
    
    Args:
        current_csv: Path to current summary CSV
        baseline_csv: Path to baseline summary CSV  
        threshold_pct: Threshold percentage for flagging regressions
        
    Returns:
        DataFrame with comparison results including delta and regression flags
    """
    current = pd.read_csv(current_csv)
    baseline = pd.read_csv(baseline_csv)
    
    # Identify common group columns
    group_cols = [c for c in ['bench', 'impl', 'n'] if c in current.columns and c in baseline.columns]
    if not group_cols:
        group_cols = ['bench'] if 'bench' in current.columns else []
    
    if not group_cols:
        return pd.DataFrame()
    
    # Merge on group columns
    merged = current.merge(
        baseline,
        on=group_cols,
        suffixes=('_current', '_baseline'),
        how='outer'
    )
    
    # Calculate deltas for time metrics
    result_cols = list(group_cols)
    
    for metric in ['wall_ms_median', 'wall_ms_mean']:
        curr_col = f'{metric}_current'
        base_col = f'{metric}_baseline'
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f'{metric}_delta'] = merged[curr_col] - merged[base_col]
            merged[f'{metric}_delta_pct'] = (
                (merged[curr_col] - merged[base_col]) / merged[base_col] * 100
            ).round(2)
            merged[f'{metric}_regression'] = merged[f'{metric}_delta_pct'] > threshold_pct
            result_cols.extend([
                curr_col, base_col, f'{metric}_delta', f'{metric}_delta_pct', f'{metric}_regression'
            ])
    
    # Calculate deltas for memory metrics  
    for metric in ['peak_rss_mb_median', 'peak_rss_mb_mean']:
        curr_col = f'{metric}_current'
        base_col = f'{metric}_baseline'
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f'{metric}_delta'] = merged[curr_col] - merged[base_col]
            merged[f'{metric}_delta_pct'] = (
                (merged[curr_col] - merged[base_col]) / merged[base_col] * 100
            ).round(2)
            result_cols.extend([
                curr_col, base_col, f'{metric}_delta', f'{metric}_delta_pct'
            ])
    
    # Select only relevant columns
    result_cols = [c for c in result_cols if c in merged.columns]
    return merged[result_cols]


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    title: str = "TempoBench Comparison Report",
    threshold_pct: float = 5.0,
    output_path: Optional[Path] = None,
) -> str:
    """Generate an HTML comparison report.
    
    Args:
        comparison_df: DataFrame from compare_summaries
        title: Report title
        threshold_pct: Threshold used for regression detection
        output_path: If provided, save the report to this path
        
    Returns:
        The generated HTML as a string
    """
    system_info = get_system_info()
    
    # Count regressions
    regression_cols = [c for c in comparison_df.columns if c.endswith('_regression')]
    total_regressions = 0
    for col in regression_cols:
        total_regressions += comparison_df[col].sum()
    
    total_configs = len(comparison_df)
    improvements = 0
    
    # Count improvements (negative delta)
    delta_pct_cols = [c for c in comparison_df.columns if c.endswith('_delta_pct')]
    for col in delta_pct_cols:
        improvements += (comparison_df[col] < -threshold_pct).sum()
    
    # Generate table with styling for regressions
    def style_row(row):
        styles = []
        for col in comparison_df.columns:
            if col.endswith('_regression') and row.get(col, False):
                styles.append('background-color: #fef2f2; color: #dc2626;')
            elif col.endswith('_delta_pct'):
                val = row.get(col, 0)
                if pd.notna(val):
                    if val > threshold_pct:
                        styles.append('background-color: #fef2f2; color: #dc2626;')
                    elif val < -threshold_pct:
                        styles.append('background-color: #f0fdf4; color: #16a34a;')
                    else:
                        styles.append('')
                else:
                    styles.append('')
            else:
                styles.append('')
        return styles
    
    # Build table HTML
    table_html = ['<table class="comparison-table">']
    table_html.append('<thead><tr>')
    for col in comparison_df.columns:
        table_html.append(f'<th>{col}</th>')
    table_html.append('</tr></thead>')
    table_html.append('<tbody>')
    
    for _, row in comparison_df.iterrows():
        table_html.append('<tr>')
        for col in comparison_df.columns:
            val = row[col]
            cell_class = ''
            if col.endswith('_regression'):
                if val:
                    cell_class = 'class="regression"'
                    val = '⚠️ YES'
                else:
                    val = '✓ NO'
            elif col.endswith('_delta_pct'):
                if pd.notna(val):
                    if val > threshold_pct:
                        cell_class = 'class="regression"'
                        val = f'+{val:.1f}%'
                    elif val < -threshold_pct:
                        cell_class = 'class="improvement"'
                        val = f'{val:.1f}%'
                    else:
                        val = f'{val:.1f}%'
            elif isinstance(val, float):
                val = f'{val:.3f}'
            table_html.append(f'<td {cell_class}>{val}</td>')
        table_html.append('</tr>')
    table_html.append('</tbody></table>')
    
    status_class = 'success' if total_regressions == 0 else 'warning'
    status_text = 'No regressions detected' if total_regressions == 0 else f'{int(total_regressions)} regressions detected'
    status_icon = '✓' if total_regressions == 0 else '⚠️'
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        
        header {{
            background: linear-gradient(135deg, #7c3aed, #5b21b6);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 12px;
        }}
        
        header h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
        
        .status-banner {{
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
        }}
        
        .status-banner.success {{
            background: #f0fdf4;
            color: var(--success-color);
            border: 2px solid var(--success-color);
        }}
        
        .status-banner.warning {{
            background: #fef2f2;
            color: var(--error-color);
            border: 2px solid var(--error-color);
        }}
        
        section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        section h2 {{
            color: var(--primary-color);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .stat-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: var(--bg-color);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid var(--primary-color);
        }}
        
        .stat-card.success {{ border-left-color: var(--success-color); }}
        .stat-card.warning {{ border-left-color: var(--warning-color); }}
        .stat-card.error {{ border-left-color: var(--error-color); }}
        
        .stat-value {{ font-size: 2rem; font-weight: 700; }}
        .stat-label {{ font-size: 0.875rem; color: #64748b; margin-top: 0.25rem; }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            overflow-x: auto;
            display: block;
        }}
        
        .comparison-table th, .comparison-table td {{
            padding: 0.6rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            white-space: nowrap;
        }}
        
        .comparison-table th {{
            background: var(--bg-color);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .comparison-table tr:hover {{ background: var(--bg-color); }}
        
        .comparison-table .regression {{
            background-color: #fef2f2;
            color: #dc2626;
            font-weight: 600;
        }}
        
        .comparison-table .improvement {{
            background-color: #f0fdf4;
            color: #16a34a;
            font-weight: 600;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #64748b;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="subtitle">Regression threshold: {threshold_pct}%</div>
        </header>
        
        <div class="status-banner {status_class}">
            {status_icon} {status_text}
        </div>
        
        <section>
            <h2>Summary</h2>
            <div class="stat-cards">
                <div class="stat-card">
                    <div class="stat-value">{total_configs}</div>
                    <div class="stat-label">Configurations Compared</div>
                </div>
                <div class="stat-card {'error' if total_regressions > 0 else 'success'}">
                    <div class="stat-value">{int(total_regressions)}</div>
                    <div class="stat-label">Regressions</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">{int(improvements)}</div>
                    <div class="stat-label">Improvements</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Detailed Comparison</h2>
            <p style="margin-bottom: 1rem; color: #64748b;">
                Values highlighted in <span style="color: #dc2626;">red</span> indicate regressions.
                Values highlighted in <span style="color: #16a34a;">green</span> indicate improvements.
            </p>
            {''.join(table_html)}
        </section>
        
        <footer>
            <p>Generated by TempoBench - Language-agnostic benchmarking orchestrator</p>
            <p>Report timestamp: {system_info['timestamp']}</p>
        </footer>
    </div>
</body>
</html>'''
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
    
    return html
