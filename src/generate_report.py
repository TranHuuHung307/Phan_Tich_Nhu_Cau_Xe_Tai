"""Generate the final multi-page PDF report from processed data and figures."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from .utils import (
        FIGURES_DIR,
        PROCESSED_CSV_PATH,
        PROJECT_ROOT,
        RAW_CSV_PATH,
        build_weekly_monthly_tables,
    )
except ImportError:
    from utils import (
        FIGURES_DIR,
        PROCESSED_CSV_PATH,
        PROJECT_ROOT,
        RAW_CSV_PATH,
        build_weekly_monthly_tables,
    )


REPORT_PATH = PROJECT_ROOT / "reports" / "final_report.pdf"
FONT_CANDIDATES = [
    (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
]


def register_fonts() -> tuple[str, str]:
    """Register a Unicode font pair for Vietnamese report text."""
    for regular, bold in FONT_CANDIDATES:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            return "ReportRegular", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    """Create an image scaled to fit a report page."""
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def build_report(output_path: Path = REPORT_PATH) -> Path:
    """Build a complete report containing metrics, conclusions, and charts."""
    regular_font, bold_font = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=regular_font,
            fontSize=10,
            leading=15,
            spaceAfter=6,
        )
    )

    raw = pd.read_csv(RAW_CSV_PATH)
    cleaned = pd.read_csv(PROCESSED_CSV_PATH, parse_dates=["Date"])
    _, monthly = build_weekly_monthly_tables(cleaned)
    series = monthly.set_index("Date")["Avg_Weekly_Volume"].asfreq("MS")
    decomposition = seasonal_decompose(
        series, model="additive", period=12, extrapolate_trend="freq"
    )
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    seasonal_cycle = seasonal.iloc[:12]

    growth = (trend.iloc[-1] / trend.iloc[0] - 1) * 100
    highest_month = int(seasonal_cycle.idxmax().month)
    lowest_month = int(seasonal_cycle.idxmin().month)
    candidate_count = int(cleaned["Is_Outlier_Candidate"].sum())
    peak_week_count = cleaned.loc[
        cleaned["Is_Peak_Week"].eq(1), ["Year", "Week"]
    ].drop_duplicates().shape[0]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Phân tích nhu cầu xe tải mùa vụ",
        author="Data Analysis Project",
    )
    story = [
        Paragraph("BÁO CÁO PHÂN TÍCH NHU CẦU XE TẢI MÙA VỤ", styles["ReportTitle"]),
        Paragraph(
            "Báo cáo được tạo từ notebook đã thực thi đầy đủ, dữ liệu cleaned và "
            "các biểu đồ trong thư mục reports/figures.",
            styles["ReportBody"],
        ),
        Paragraph("1. Tổng quan dữ liệu", styles["ReportHeading"]),
    ]

    summary_data = [
        ["Chỉ tiêu", "Kết quả"],
        ["Số dòng dữ liệu raw", f"{len(raw):,}"],
        ["Số dòng dữ liệu cleaned", f"{len(cleaned):,}"],
        ["Số cột cleaned", f"{len(cleaned.columns):,}"],
        ["Missing value sau cleaning", f"{int(cleaned.isna().sum().sum()):,}"],
        ["Số tuần cao điểm phân biệt", f"{peak_week_count:,}"],
        ["Ứng viên ngoại lai residual", f"{candidate_count:,}"],
    ]
    summary_table = Table(summary_data, colWidths=[8 * cm, 5 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), regular_font),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            summary_table,
            Spacer(1, 12),
            Paragraph("2. Phương pháp cleaning và phân tích", styles["ReportHeading"]),
            Paragraph(
                "Raw data được giữ nguyên và xác minh bằng SHA-256. Missing metric "
                "chỉ được điền trong cùng Route + Fleet_Type. Ngoại lai được gắn cờ "
                "trên residual sau khi kiểm soát trend và seasonality; không tự động "
                "xóa hoặc winsorize.",
                styles["ReportBody"],
            ),
            Paragraph(
                "Phân rã chuỗi thời gian sử dụng mô hình cộng: "
                "Observed = Trend + Seasonality + Residual. Dữ liệu decomposition là "
                "khối lượng trung bình mỗi tuần trong tháng để tránh thiên lệch giữa "
                "tháng có 4 và 5 tuần.",
                styles["ReportBody"],
            ),
            Paragraph("3. Kết quả chính", styles["ReportHeading"]),
        ]
    )

    result_data = [
        ["Chỉ tiêu", "Kết quả"],
        ["Tăng trưởng trend toàn kỳ", f"{growth:+.2f}%"],
        ["Tháng có mùa vụ cao nhất", f"Tháng {highest_month}"],
        ["Tháng có mùa vụ thấp nhất", f"Tháng {lowest_month}"],
        ["Biên độ mùa vụ", f"{seasonal.max() - seasonal.min():,.2f} tấn"],
        ["Residual trung bình", f"{residual.mean():,.4f} tấn"],
        ["Độ lệch chuẩn residual", f"{residual.std():,.2f} tấn"],
        ["Residual lớn nhất", f"{residual.max():+,.2f} tấn"],
        ["Residual nhỏ nhất", f"{residual.min():+,.2f} tấn"],
    ]
    result_table = Table(result_data, colWidths=[8 * cm, 5 * cm])
    result_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), regular_font),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            result_table,
            Spacer(1, 12),
            Paragraph("4. Kết luận", styles["ReportHeading"]),
            Paragraph(
                f"Nhu cầu vận tải có xu hướng tăng rõ rệt, với trend tăng "
                f"{growth:.2f}% trong giai đoạn phân tích. Thành phần mùa vụ cao nhất "
                f"ở tháng {highest_month} và thấp nhất ở tháng {lowest_month}. "
                "Residual trung bình gần 0 cho thấy decomposition tách được phần lớn "
                "trend và seasonality; các residual lớn vẫn cần xác minh nghiệp vụ.",
                styles["ReportBody"],
            ),
        ]
    )

    chart_sections = [
        ("Kiểm tra ngoại lai theo tuyến và loại xe", "01_Outlier_Boxplot.png"),
        ("Line chart: xu hướng theo thời gian", "02_LineChart_XuHuong.png"),
        ("Bar chart: so sánh tuyến và tháng", "03_BarChart_SoSanh.png"),
        ("Pie chart: tỷ trọng khối lượng theo loại xe", "04_PieChart_LoaiXe.png"),
        ("Scatter plot: số chuyến và khối lượng", "05_ScatterPlot_TuongQuan.png"),
        ("Phân rã Trend, Seasonality và Residual", "06_Decomposition_ChuoiThoiGian.png"),
    ]
    for title, filename in chart_sections:
        story.extend(
            [
                PageBreak(),
                Paragraph(title, styles["ReportHeading"]),
                scaled_image(FIGURES_DIR / filename, 25 * cm, 15.5 * cm),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    return output_path


def main() -> None:
    output_path = build_report()
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
