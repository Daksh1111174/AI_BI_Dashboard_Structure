"""
===========================================================
Export Utilities
AI Business Intelligence Dashboard

Author : Daksh Shah
===========================================================
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def export_dashboard(df):
    """
    Export dashboard data and summaries.
    """

    st.header("📤 Export Center")

    tab1, tab2, tab3 = st.tabs(
        [
            "CSV",
            "Excel",
            "PDF"
        ]
    )

    # ============================================
    # CSV
    # ============================================

    with tab1:

        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="dashboard_data.csv",
            mime="text/csv"
        )

    # ============================================
    # Excel
    # ============================================

    with tab2:

        buffer = io.BytesIO()

        with pd.ExcelWriter(
            buffer,
            engine="xlsxwriter"
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Dashboard"
            )

            workbook = writer.book
            worksheet = writer.sheets["Dashboard"]

            header = workbook.add_format(
                {
                    "bold": True,
                    "bg_color": "#2563EB",
                    "font_color": "white"
                }
            )

            for col_num, value in enumerate(df.columns):
                worksheet.write(
                    0,
                    col_num,
                    value,
                    header
                )

            worksheet.autofilter(
                0,
                0,
                len(df),
                len(df.columns) - 1
            )

        st.download_button(
            "📥 Download Excel",
            buffer.getvalue(),
            file_name="dashboard.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    # ============================================
    # PDF
    # ============================================

    with tab3:

        revenue = (
            df["Sales"].sum()
            if "Sales" in df.columns
            else 0
        )

        profit = (
            df["Profit"].sum()
            if "Profit" in df.columns
            else 0
        )

        orders = (
            df["Order ID"].nunique()
            if "Order ID" in df.columns
            else 0
        )

        customers = (
            df["Customer ID"].nunique()
            if "Customer ID" in df.columns
            else 0
        )

        pdf_buffer = io.BytesIO()

        doc = SimpleDocTemplate(pdf_buffer)

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "<b>AI Business Intelligence Dashboard</b>",
                styles["Title"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"Generated: {datetime.now():%d-%m-%Y %H:%M}",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"<b>Total Revenue:</b> ₹ {revenue:,.2f}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Total Profit:</b> ₹ {profit:,.2f}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Total Orders:</b> {orders}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Total Customers:</b> {customers}",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 20))

        story.append(
            Paragraph(
                "<b>Executive Summary</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                (
                    "This report summarizes the uploaded business "
                    "dataset. Use the dashboard to explore revenue, "
                    "profitability, customer behavior, product "
                    "performance, forecasting, and AI-generated insights."
                ),
                styles["Normal"]
            )
        )

        doc.build(story)

        st.download_button(
            "📥 Download PDF",
            pdf_buffer.getvalue(),
            file_name="dashboard_report.pdf",
            mime="application/pdf"
        )

    st.success("Export options are ready.")
