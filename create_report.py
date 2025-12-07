import os
import base64
from datetime import datetime


def create_html_report():
    """Create an HTML report with all plots"""
    output_dir = "experiment_results"

    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist!")
        return

    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

    if not files:
        print("No plot files found!")
        return

    # Group files by experiment
    experiments = {}
    for f in files:
        # Extract experiment name (fig2, fig3a, etc.)
        if f.startswith('fig'):
            exp_name = f.split('_')[0]  # fig2, fig3a, etc.
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(f)

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DPC Clustering Experiment Results</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1400px; 
                margin: 0 auto; 
                background-color: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px; 
                padding-bottom: 20px; 
                border-bottom: 2px solid #e0e0e0;
            }}
            h1 {{ 
                color: #2c3e50; 
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            .subtitle {{ 
                color: #7f8c8d; 
                font-size: 1.2em;
                margin-bottom: 20px;
            }}
            .paper-info {{
                background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                border-left: 5px solid #3498db;
            }}
            h2 {{ 
                color: #2980b9; 
                margin-top: 40px; 
                padding-bottom: 10px; 
                border-bottom: 2px solid #3498db;
                font-size: 1.8em;
            }}
            .experiment {{ 
                margin-bottom: 50px; 
                padding: 25px; 
                background-color: #f8f9fa; 
                border-radius: 10px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .experiment-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}
            .experiment-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .experiment-description {{
                color: #546e7a;
                margin-bottom: 20px;
                line-height: 1.6;
            }}
            .plot-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
                gap: 25px; 
                margin-top: 20px; 
            }}
            .plot-container {{ 
                text-align: center; 
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                transition: transform 0.3s ease;
            }}
            .plot-container:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .plot-title {{ 
                font-weight: bold; 
                margin: 15px 0; 
                color: #2c3e50;
                font-size: 1.1em;
            }}
            .plot-container img {{ 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #e0e0e0; 
                border-radius: 5px; 
            }}
            .plot-description {{
                font-size: 0.9em;
                color: #666;
                margin-top: 10px;
                line-height: 1.4;
            }}
            .timestamp {{ 
                color: #7f8c8d; 
                font-size: 0.9em; 
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
            }}
            .nav-bar {{
                background: #2c3e50;
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
            }}
            .nav-link {{
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 4px;
                transition: background 0.3s;
            }}
            .nav-link:hover {{
                background: #3498db;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 DPC Clustering Experiment Results</h1>
                <div class="subtitle">Reproduction of "Clustering by fast search and find of density peaks"</div>
            </div>

            <div class="paper-info">
                <strong>📄 Paper:</strong> "Clustering by fast search and find of density peaks"<br>
                <strong>👥 Authors:</strong> Alex Rodriguez and Alessandro Laio<br>
                <strong>🏛️ Journal:</strong> Science 344, 1492 (2014)<br>
                <strong>📈 Algorithm:</strong> Density Peaks Clustering (DPC)<br>
                <strong>🎯 Key Idea:</strong> Cluster centers are characterized by high density and large distance from points with higher density
            </div>

            <div class="nav-bar">
                <a href="#fig2" class="nav-link">Figure 2</a>
                <a href="#fig3a" class="nav-link">Figure 3A</a>
                <a href="#fig3b" class="nav-link">Figure 3B</a>
                <a href="#fig3c" class="nav-link">Figure 3C</a>
                <a href="#fig3d" class="nav-link">Figure 3D</a>
                <a href="#comparison" class="nav-link">Comparison</a>
            </div>
    """

    # Experiment descriptions
    exp_descriptions = {
        'fig2': "Synthetic data with 5 density peaks of varying shapes and densities. Tests the algorithm's ability to detect non-spherical clusters with different densities.",
        'fig3a': "Two crescent-shaped clusters (moons dataset). Tests ability to detect non-convex clusters.",
        'fig3b': "15 highly overlapping Gaussian clusters. Tests resolution in distinguishing many clusters.",
        'fig3c': "Three concentric circular clusters. Tests ability to detect nested structures.",
        'fig3d': "Three curved, non-linearly separable clusters. Tests performance on complex shapes."
    }

    # Plot descriptions
    plot_descriptions = {
        'clusters': "Shows the final clustering results. Different colors represent different clusters. Gray 'X' marks are halo/noise points. Yellow stars are cluster centers.",
        'decision': "Decision graph plotting δ (minimum distance to higher density point) vs ρ (local density). Cluster centers appear in the top-right corner (high δ and high ρ).",
        'gamma': "Gamma values (γ = ρ × δ) sorted in descending order. Clear clusters show a sharp drop in γ values after the true centers."
    }

    # Add each experiment
    for exp_name in sorted(experiments.keys()):
        # Get experiment title
        if exp_name == 'fig2':
            title = "Figure 2: Synthetic Data with 5 Density Peaks"
            anchor = "fig2"
        elif exp_name == 'fig3a':
            title = "Figure 3A: Two Crescent Moons"
            anchor = "fig3a"
        elif exp_name == 'fig3b':
            title = "Figure 3B: 15 Overlapping Clusters"
            anchor = "fig3b"
        elif exp_name == 'fig3c':
            title = "Figure 3C: Three Concentric Circles"
            anchor = "fig3c"
        elif exp_name == 'fig3d':
            title = "Figure 3D: Three Curved Clusters"
            anchor = "fig3d"
        else:
            title = f"Experiment: {exp_name.upper()}"
            anchor = exp_name

        html_content += f"""
            <div class="experiment" id="{anchor}">
                <div class="experiment-header">
                    <div class="experiment-title">{title}</div>
                </div>
                <div class="experiment-description">
                    {exp_descriptions.get(exp_name, 'No description available.')}
                </div>
                <div class="plot-grid">
        """

        for plot_file in sorted(experiments[exp_name]):
            filepath = os.path.join(output_dir, plot_file)

            # Encode image to base64 for HTML embedding
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Extract plot type
            plot_type = plot_file.replace(f"{exp_name}_", "").replace(".png", "")

            # Get plot title
            if 'cluster' in plot_type:
                plot_title = "Clustering Results"
            elif 'decision' in plot_type:
                plot_title = "Decision Graph"
            elif 'gamma' in plot_type:
                plot_title = "Gamma Values"
            else:
                plot_title = plot_type.replace("_", " ").title()

            html_content += f"""
                    <div class="plot-container">
                        <div class="plot-title">{plot_title}</div>
                        <img src="data:image/png;base64,{img_data}" alt="{plot_file}">
                        <div class="plot-description">
                            {plot_descriptions.get(plot_type.split('_')[0] if '_' in plot_type else plot_type, '')}
                        </div>
                    </div>
            """

        html_content += """
                </div>
            </div>
        """

    # Add comparison plot if exists
    comparison_file = "all_test_cases_comparison.png"
    comparison_path = os.path.join(output_dir, comparison_file)
    if os.path.exists(comparison_path):
        with open(comparison_path, "rb") as img_file:
            comparison_data = base64.b64encode(img_file.read()).decode('utf-8')

        html_content += f"""
            <div class="experiment" id="comparison">
                <div class="experiment-header">
                    <div class="experiment-title">Comparison: All Test Cases</div>
                </div>
                <div class="experiment-description">
                    Side-by-side comparison of DPC algorithm performance on all test cases from Figure 3.
                </div>
                <div class="plot-container" style="max-width: 900px; margin: 0 auto;">
                    <div class="plot-title">Algorithm Performance Comparison</div>
                    <img src="data:image/png;base64,{comparison_data}" alt="Comparison">
                    <div class="plot-description">
                        Summary plot showing DPC clustering results on all four test cases from Figure 3.
                    </div>
                </div>
            </div>
        """

    html_content += f"""
            <div class="timestamp">
                Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>

            <div class="footer">
                <p>Density Peaks Clustering (DPC) Algorithm | Rodriguez & Laio, 2014</p>
                <p>Experiment reproduction for educational purposes</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML file
    report_file = "experiment_report.html"
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"HTML report created: {report_file}")
    print(f"Open it in your browser to view all annotated plots!")

    return report_file


if __name__ == "__main__":
    print("Creating enhanced HTML report...")
    report_file = create_html_report()

    print(f"\n✅ Report created: {report_file}")
    print("\n📊 To view the report:")
    print("   firefox experiment_report.html")
    print("   OR google-chrome experiment_report.html")
    print("   OR xdg-open experiment_report.html")