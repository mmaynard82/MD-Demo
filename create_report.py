# 03_create_report.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path

OUT_DIR = Path("output")
doc = SimpleDocTemplate(OUT_DIR / "Insights_Report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Maynard Dynamics — Analytics Jumpstart: Executive Summary", styles['Title']))
story.append(Spacer(1,12))
story.append(Paragraph("Top takeaways:", styles['Heading2']))
story.append(Paragraph(
    "- We found X top selling products and Y regions driving revenue. "
    "- Profit hotspots and low-margin SKUs are identified. "
    "- Short term forecast suggests Z% trend (see chart).", styles['Normal']
))
story.append(Spacer(1,12))
story.append(Image(str(OUT_DIR / "sales_trend.png"), width=400, height=150))
story.append(Spacer(1,6))
story.append(Image(str(OUT_DIR / "top_products.png"), width=400, height=150))
story.append(Spacer(1,12))
story.append(Paragraph("Next steps: Deploy dashboard, automate reporting, run targeted promotions for top products.", styles['Normal']))
doc.build(story)
print("Saved", OUT_DIR / "Insights_Report.pdf")
