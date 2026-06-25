"""
===========================================================
AI Business Intelligence

Base Visualization

Author : Daksh Shah
===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# ============================================================
# Base Visualization
# ============================================================

class BaseVisualization:
    """
    Base visualization class for AI Business Intelligence.

    All dashboard pages should use this class.

    Supports:
        • Line Charts
        • Area Charts
        • Bar Charts
        • Pie Charts
        • Donut Charts
    """

    # --------------------------------------------------------

    def __init__(
        self,
        template: str = "plotly_white"
    ):

        self.template = template

        self.color_sequence = px.colors.qualitative.Set2

    # ========================================================
    # Theme
    # ========================================================

    def set_theme(
        self,
        template: str
    ):

        self.template = template

    # ========================================================
    # Line Chart
    # ========================================================

    def line_chart(
        self,
        dataframe: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        color: Optional[str] = None,
        markers: bool = False
    ):

        fig = px.line(

            dataframe,

            x=x,

            y=y,

            color=color,

            title=title,

            markers=markers,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        fig.update_layout(

            hovermode="x unified"

        )

        return fig

    # ========================================================
    # Area Chart
    # ========================================================

    def area_chart(
        self,
        dataframe,
        x,
        y,
        title=""
    ):

        fig = px.area(

            dataframe,

            x=x,

            y=y,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Bar Chart
    # ========================================================

    def bar_chart(
        self,
        dataframe,
        x,
        y,
        title="",
        color=None,
        orientation="v"
    ):

        fig = px.bar(

            dataframe,

            x=x,

            y=y,

            color=color,

            orientation=orientation,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Horizontal Bar Chart
    # ========================================================

    def horizontal_bar_chart(
        self,
        dataframe,
        x,
        y,
        title=""
    ):

        return self.bar_chart(

            dataframe,

            x=x,

            y=y,

            title=title,

            orientation="h"

        )

    # ========================================================
    # Pie Chart
    # ========================================================

    def pie_chart(
        self,
        dataframe,
        names,
        values,
        title=""
    ):

        fig = px.pie(

            dataframe,

            names=names,

            values=values,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Donut Chart
    # ========================================================

    def donut_chart(
        self,
        dataframe,
        names,
        values,
        title=""
    ):

        fig = px.pie(

            dataframe,

            names=names,

            values=values,

            hole=0.55,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Multi-Line Chart
    # ========================================================

    def multi_line_chart(
        self,
        dataframe,
        x,
        y_columns: List[str],
        title=""
    ):

        fig = go.Figure()

        for column in y_columns:

            fig.add_trace(

                go.Scatter(

                    x=dataframe[x],

                    y=dataframe[column],

                    mode="lines",

                    name=column

                )

            )

        fig.update_layout(

            template=self.template,

            title=title,

            hovermode="x unified"

        )

        return fig

    # ========================================================
    # KPI Indicator
    # ========================================================

    def indicator(
        self,
        value,
        title,
        suffix=""
    ):

        fig = go.Figure(

            go.Indicator(

                mode="number",

                value=value,

                title={"text": title},

                number={

                    "suffix": suffix

                }

            )

        )

        fig.update_layout(

            template=self.template,

            height=250

        )

        return fig

    # ========================================================
    # Multi KPI Dashboard
    # ========================================================

    def kpi_dashboard(
        self,
        metrics: dict
    ):

        fig = make_subplots(

            rows=1,

            cols=len(metrics),

            specs=[[

                {"type": "indicator"}

                for _ in metrics

            ]]

        )

        col = 1

        for name, value in metrics.items():

            fig.add_trace(

                go.Indicator(

                    mode="number",

                    value=float(value),

                    title={

                        "text": name

                    }

                ),

                row=1,

                col=col

            )

            col += 1

        fig.update_layout(

            template=self.template,

            height=250

        )

        return fig
          # ========================================================
    # Scatter Plot
    # ========================================================

    def scatter_chart(
        self,
        dataframe,
        x,
        y,
        color=None,
        size=None,
        title=""
    ):

        fig = px.scatter(

            dataframe,

            x=x,

            y=y,

            color=color,

            size=size,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Bubble Chart
    # ========================================================

    def bubble_chart(
        self,
        dataframe,
        x,
        y,
        size,
        color=None,
        hover_name=None,
        title=""
    ):

        fig = px.scatter(

            dataframe,

            x=x,

            y=y,

            size=size,

            color=color,

            hover_name=hover_name,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Histogram
    # ========================================================

    def histogram(
        self,
        dataframe,
        column,
        bins=30,
        color=None,
        title=""
    ):

        fig = px.histogram(

            dataframe,

            x=column,

            nbins=bins,

            color=color,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Box Plot
    # ========================================================

    def box_plot(
        self,
        dataframe,
        x=None,
        y=None,
        color=None,
        title=""
    ):

        fig = px.box(

            dataframe,

            x=x,

            y=y,

            color=color,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Violin Plot
    # ========================================================

    def violin_plot(
        self,
        dataframe,
        x,
        y,
        color=None,
        title=""
    ):

        fig = px.violin(

            dataframe,

            x=x,

            y=y,

            color=color,

            box=True,

            points="outliers",

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Heatmap
    # ========================================================

    def heatmap(
        self,
        matrix,
        title="Heatmap"
    ):

        fig = px.imshow(

            matrix,

            text_auto=".2f",

            aspect="auto",

            title=title,

            color_continuous_scale="RdBu_r"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # ========================================================
    # Correlation Matrix
    # ========================================================

    def correlation_matrix(
        self,
        dataframe
    ):

        corr = dataframe.corr(

            numeric_only=True

        )

        return self.heatmap(

            corr,

            title="Correlation Matrix"

        )

    # ========================================================
    # Density Contour
    # ========================================================

    def density_contour(
        self,
        dataframe,
        x,
        y,
        title=""
    ):

        fig = px.density_contour(

            dataframe,

            x=x,

            y=y,

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Pair Plot
    # ========================================================

    def scatter_matrix(
        self,
        dataframe,
        dimensions,
        color=None,
        title=""
    ):

        fig = px.scatter_matrix(

            dataframe,

            dimensions=dimensions,

            color=color,

            title=title

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # ========================================================
    # Distribution Comparison
    # ========================================================

    def distribution_comparison(
        self,
        dataframe,
        column,
        category,
        title=""
    ):

        fig = px.histogram(

            dataframe,

            x=column,

            color=category,

            marginal="box",

            barmode="overlay",

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Time Series Scatter
    # ========================================================

    def time_series_scatter(
        self,
        dataframe,
        x,
        y,
        color=None,
        title=""
    ):

        fig = px.scatter(

            dataframe,

            x=x,

            y=y,

            color=color,

            trendline="ols",

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Missing Values Heatmap
    # ========================================================

    def missing_values_heatmap(
        self,
        dataframe
    ):

        missing = dataframe.isna().astype(int)

        fig = px.imshow(

            missing,

            aspect="auto",

            title="Missing Values",

            color_continuous_scale="Blues"

        )

        fig.update_layout(

            template=self.template

        )

        return fig
          # ========================================================
    # Treemap
    # ========================================================

    def treemap(
        self,
        dataframe,
        path,
        values,
        color=None,
        title=""
    ):

        fig = px.treemap(

            dataframe,

            path=path,

            values=values,

            color=color,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Sunburst
    # ========================================================

    def sunburst(
        self,
        dataframe,
        path,
        values,
        color=None,
        title=""
    ):

        fig = px.sunburst(

            dataframe,

            path=path,

            values=values,

            color=color,

            title=title,

            template=self.template,

            color_discrete_sequence=self.color_sequence

        )

        return fig

    # ========================================================
    # Funnel Chart
    # ========================================================

    def funnel_chart(
        self,
        dataframe,
        x,
        y,
        title=""
    ):

        fig = px.funnel(

            dataframe,

            x=x,

            y=y,

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Waterfall Chart
    # ========================================================

    def waterfall_chart(
        self,
        dataframe,
        x,
        y,
        title=""
    ):

        fig = go.Figure(

            go.Waterfall(

                x=dataframe[x],

                y=dataframe[y]

            )

        )

        fig.update_layout(

            template=self.template,

            title=title

        )

        return fig

    # ========================================================
    # Radar Chart
    # ========================================================

    def radar_chart(
        self,
        dataframe,
        category_column,
        value_column,
        title=""
    ):

        fig = go.Figure()

        fig.add_trace(

            go.Scatterpolar(

                r=dataframe[value_column],

                theta=dataframe[category_column],

                fill="toself",

                name=title

            )

        )

        fig.update_layout(

            template=self.template,

            polar=dict(

                radialaxis=dict(

                    visible=True

                )

            ),

            title=title

        )

        return fig

    # ========================================================
    # Sankey Diagram
    # ========================================================

    def sankey_chart(
        self,
        labels,
        source,
        target,
        values,
        title=""
    ):

        fig = go.Figure(

            data=[

                go.Sankey(

                    node=dict(

                        label=labels

                    ),

                    link=dict(

                        source=source,

                        target=target,

                        value=values

                    )

                )

            ]

        )

        fig.update_layout(

            template=self.template,

            title=title

        )

        return fig

    # ========================================================
    # Geographic Scatter Map
    # ========================================================

    def geo_scatter(
        self,
        dataframe,
        lat,
        lon,
        color=None,
        size=None,
        hover_name=None,
        title=""
    ):

        fig = px.scatter_geo(

            dataframe,

            lat=lat,

            lon=lon,

            color=color,

            size=size,

            hover_name=hover_name,

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Choropleth Map
    # ========================================================

    def choropleth(
        self,
        dataframe,
        locations,
        color,
        locationmode="country names",
        title=""
    ):

        fig = px.choropleth(

            dataframe,

            locations=locations,

            color=color,

            locationmode=locationmode,

            title=title,

            template=self.template

        )

        return fig

    # ========================================================
    # Calendar Heatmap (Monthly)
    # ========================================================

    def calendar_heatmap(
        self,
        dataframe,
        date_column,
        value_column,
        title=""
    ):

        df = dataframe.copy()

        df[date_column] = pd.to_datetime(
            df[date_column]
        )

        df["Month"] = df[date_column].dt.month_name()

        pivot = (

            df

            .groupby("Month")[value_column]

            .sum()

            .reindex(

                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December"
                ]

            )

            .fillna(0)

        )

        fig = px.imshow(

            [pivot.values],

            x=pivot.index,

            y=["Sales"],

            text_auto=".0f",

            title=title,

            color_continuous_scale="Viridis"

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # ========================================================
    # Gauge Chart
    # ========================================================

    def gauge_chart(
        self,
        value,
        min_value,
        max_value,
        title=""
    ):

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=value,

                title={"text": title},

                gauge=dict(

                    axis=dict(

                        range=[

                            min_value,

                            max_value

                        ]

                    )

                )

            )

        )

        fig.update_layout(

            template=self.template

        )

        return fig

    # ========================================================
    # Table
    # ========================================================

    def table(
        self,
        dataframe,
        title=""
    ):

        fig = go.Figure(

            data=[

                go.Table(

                    header=dict(

                        values=list(

                            dataframe.columns

                        )

                    ),

                    cells=dict(

                        values=[

                            dataframe[col]

                            for col in dataframe.columns

                        ]

                    )

                )

            ]

        )

        fig.update_layout(

            template=self.template,

            title=title

        )

        return fig
          # ========================================================
    # Two Column Dashboard
    # ========================================================

    def two_column_dashboard(
        self,
        left_figure,
        right_figure,
        left_title="",
        right_title=""
    ):

        return {
            "layout": "two_column",
            "left": {
                "title": left_title,
                "figure": left_figure
            },
            "right": {
                "title": right_title,
                "figure": right_figure
            }
        }

    # ========================================================
    # Four KPI Layout
    # ========================================================

    def four_kpi_layout(
        self,
        metrics: dict
    ):

        figures = {}

        for name, value in metrics.items():

            figures[name] = self.indicator(
                value=value,
                title=name
            )

        return figures

    # ========================================================
    # Dashboard Package
    # ========================================================

    def dashboard_package(
        self,
        figures: dict,
        title="Dashboard"
    ):

        return {

            "title": title,

            "created": pd.Timestamp.now(),

            "charts": figures,

            "count": len(figures)

        }

    # ========================================================
    # Save HTML
    # ========================================================

    def save_html(
        self,
        figure,
        path
    ):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.write_html(path)

        return path

    # ========================================================
    # Save Image
    # ========================================================

    def save_image(
        self,
        figure,
        path
    ):
        """
        Requires Plotly image export support
        (for example, Kaleido).
        """

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.write_image(path)

        return path

    # ========================================================
    # Save PDF
    # ========================================================

    def save_pdf(
        self,
        figure,
        path
    ):
        """
        Requires Plotly image export support.
        """

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.write_image(path)

        return path

    # ========================================================
    # Export Dashboard
    # ========================================================

    def export_dashboard(
        self,
        dashboard,
        folder="reports/dashboard"
    ):

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        exported = {}

        for name, fig in dashboard["charts"].items():

            filename = folder / f"{name}.html"

            fig.write_html(filename)

            exported[name] = str(filename)

        return exported

    # ========================================================
    # Apply Layout Defaults
    # ========================================================

    def apply_defaults(
        self,
        figure
    ):

        figure.update_layout(

            template=self.template,

            font=dict(

                family="Arial",

                size=14

            ),

            margin=dict(

                l=40,
                r=40,
                t=60,
                b=40

            )

        )

        return figure

    # ========================================================
    # Streamlit Helper
    # ========================================================

    @staticmethod
    def show(
        st,
        figure,
        use_container_width=True
    ):

        st.plotly_chart(

            figure,

            use_container_width=use_container_width

        )

    # ========================================================
    # Available Themes
    # ========================================================

    @staticmethod
    def available_themes():

        return [

            "plotly",

            "plotly_white",

            "plotly_dark",

            "ggplot2",

            "seaborn",

            "simple_white",

            "presentation",

            "xgridoff",

            "ygridoff"

        ]

    # ========================================================
    # Reset Theme
    # ========================================================

    def reset_theme(self):

        self.template = "plotly_white"

    # ========================================================
    # Validate DataFrame
    # ========================================================

    @staticmethod
    def validate_dataframe(
        dataframe
    ):

        if dataframe is None:

            raise ValueError(
                "DataFrame is None."
            )

        if dataframe.empty:

            raise ValueError(
                "DataFrame is empty."
            )

        return True

    # ========================================================
    # Figure Information
    # ========================================================

    @staticmethod
    def figure_info(
        figure
    ):

        return {

            "type":

                type(figure).__name__,

            "traces":

                len(figure.data),

            "layout":

                figure.layout.to_plotly_json()

        }

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self):

        return (

            f"BaseVisualization("

            f"theme='{self.template}')"

        )
      
