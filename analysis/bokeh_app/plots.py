# bokeh_app/plots.py
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, CategoricalColorMapper, CDSView, GroupFilter, DataTable, TableColumn, Div, Select, Panel, Tabs
from bokeh.layouts import column, row
from bokeh.transform import linear_cmap
from bokeh.palettes import Category10, Viridis256, magma
from scipy.stats import gaussian_kde

import pandas as pd

from .data import (
    get_uranium_by_depth, 
    get_element_correlations_for_granite, 
    get_lithology_uranium_distribution, 
    get_major_elements_vs_si02,
    get_trace_elements_vs_si02,
    get_scatter_data_for_element,
    get_stats_uranium_concentration
)

# Color palette for lithologies (expand as needed)
lithology_palette = Category10[10]  # Adjust as needed

def plot_uranium_by_depth():
    df = get_uranium_by_depth()
    if df.empty:
        return Div(text="No data available for Uranium by Depth plot.")

    source = ColumnDataSource(df)
    
    # Create a new plot
    p = figure(width=800, height=400, title="Average Uranium Concentration by Depth",
               x_axis_label="Average Depth (m)", y_axis_label="Uranium (ppm)")

    # Add a circle renderer with a size, color, and alpha
    p.circle(x='depth', y='U_ppm', size=8, source=source, alpha=0.5)
    
    # Calculate the average uranium concentration for each 1-meter interval
    depth_intervals = np.arange(start=min(df['depth']), stop=max(df['depth']) + 1, step=1)
    avg_u_by_depth = []
    for i in range(len(depth_intervals) - 1):
        interval_data = df[(df['depth'] >= depth_intervals[i]) & (df['depth'] < depth_intervals[i+1])]
        avg_u = interval_data['U_ppm'].mean() if not interval_data.empty else None
        avg_u_by_depth.append(avg_u)
    
    # Create a ColumnDataSource for the average line
    avg_source = ColumnDataSource(data={
        'depth': depth_intervals[:-1],
        'avg_U_ppm': avg_u_by_depth
    })
    
    # Add a line for the average uranium concentration
    p.line(x='depth', y='avg_U_ppm', line_width=2, color="red", source=avg_source, legend_label="Average U")

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [
        ("Depth", "@depth m"),
        ("Uranium", "@U_ppm ppm"),
        ("Lithology", "@lithology"),
        ("Hole ID", "@hole_id")
    ]
    p.add_tools(hover)

    return p

def plot_element_correlations(element='U'):
    df = get_element_correlations_for_granite(element)
    if df.empty:
        return Div(text=f"No data available for {element} correlations.")

    source = ColumnDataSource(df)
    
    # Create a color mapper based on the 'Correlation' column
    min_corr, max_corr = df['Correlation'].min(), df['Correlation'].max()
    mapper = linear_cmap('Correlation', Viridis256, low=min_corr, high=max_corr)
    
    p = figure(
        width=800, height=500, title=f"Correlation of Elements with {element} in Granite",
        x_range=list(df['Element']), y_axis_label="Correlation Coefficient",
        toolbar_location="above"
    )
    
    p.vbar(x='Element', top='Correlation', width=0.9, source=source, line_color="white", fill_color=mapper)
    
    # Add hover tool to display correlation values
    hover = HoverTool()
    hover.tooltips = [("Element", "@Element"), ("Correlation", "@Correlation{0.000}")]
    p.add_tools(hover)
    
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1.2  # Rotate x-axis labels for better readability
    
    return p

def plot_lithology_uranium_distribution():
    df = get_lithology_uranium_distribution()
    if df.empty:
        return Div(text="No data available for Uranium Distribution by Lithology plot.")
    
    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    
    # Determine unique lithologies and assign colors
    unique_lithologies = df['lithology'].unique()
    lithology_mapper = CategoricalColorMapper(factors=unique_lithologies, palette=Category10[len(unique_lithologies)])
    
    # Create a plot
    p = figure(width=800, height=400, x_axis_label="Uranium (ppm)", y_axis_label="Density",
               title="Uranium Distribution by Lithology")
    
    # Add KDE plots for each lithology
    for lithology in unique_lithologies:
        lith_df = df[df['lithology'] == lithology]
        
        # Check if there is enough data to calculate KDE
        if len(lith_df) > 1:
            density = gaussian_kde(lith_df['U_ppm'])
            x = np.linspace(0, lith_df['U_ppm'].max(), 1000)
            y = density(x)
            
            # Create a ColumnDataSource for the current lithology
            lith_source = ColumnDataSource(data={'x': x, 'y': y, 'lithology': [lithology] * len(x)})
            
            # Add the KDE plot for the current lithology
            p.line(x='x', y='y', source=lith_source, line_width=2,
                   color=lithology_mapper.palette[unique_lithologies.tolist().index(lithology)],
                   legend_label=lithology)
        else:
            # Handle cases with insufficient data (e.g., display a message or skip)
            print(f"Insufficient data for KDE calculation for lithology: {lithology}")
    
    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [("Uranium", "@x ppm"), ("Density", "@y"), ("Lithology", "@lithology")]
    p.add_tools(hover)
    
    # Hide the legend if it's too crowded
    if len(unique_lithologies) > 10:
        p.legend.visible = False
    
    return p

def plot_major_elements_vs_si02():
    df = get_major_elements_vs_si02()
    if df.empty:
        return Div(text="No data available for Major Elements vs SiO2 plot.")
    
    plots = []
    major_elements = ['Al2O3', 'CaO', 'FeOT', 'MgO', 'Na2O', 'K2O', 'TiO2']
    
    for element in major_elements:
        if element in df.columns:
            source = ColumnDataSource(df)
            
            # Create a new plot for each element
            p = figure(width=400, height=300, title=f"{element} vs SiO2 in Granite",
                       x_axis_label="SiO2 (%)", y_axis_label=f"{element} (%)")
            
            # Scatter plot
            p.circle(x='SiO2', y=element, source=source, size=8, color="blue", alpha=0.5)
            
            # Add hover tool
            hover = HoverTool()
            hover.tooltips = [(f"{element}", f"@{element} %"), ("SiO2", "@SiO2 %")]
            p.add_tools(hover)
            
            plots.append(p)
    
    # Arrange plots in a grid layout
    grid = column([row(plots[i], plots[i+1]) for i in range(0, len(plots), 2)])
    
    return grid

def plot_trace_elements_vs_si02():
    df = get_trace_elements_vs_si02()
    if df.empty:
        return Div(text="No data available for Trace Elements vs SiO2 plot.")

    plots = []
    trace_elements = ['Ba', 'Nb', 'Rb', 'Sr', 'Zn', 'Zr']  # Example trace elements

    for element in trace_elements:
        if element in df.columns:
            source = ColumnDataSource(df)

            # Create a new plot for each element
            p = figure(width=400, height=300, title=f"{element} vs SiO2 in Granite",
                       x_axis_label="SiO2 (%)", y_axis_label=f"{element} (ppm)")

            # Scatter plot
            p.circle(x='SiO2', y=element, source=source, size=8, color="green", alpha=0.5)

            # Add hover tool
            hover = HoverTool()
            hover.tooltips = [(f"{element}", f"@{element} ppm"), ("SiO2", "@SiO2 %")]
            p.add_tools(hover)

            plots.append(p)

    # Arrange plots in a grid layout
    grid = column([row(plots[i], plots[i+1]) for i in range(0, len(plots), 2)])

    return grid

def create_scatter_plot_panel():
    # Get the list of available elements
    df_elements = get_element_correlations_for_granite('U')
    available_elements = df_elements['Element'].tolist() if not df_elements.empty else []

    # Create dropdown widgets for element selection
    select_element1 = Select(title="Element 1:", options=available_elements, value='U')
    select_element2 = Select(title="Element 2:", options=available_elements, value='Th')

    def update_plot(attrname, old, new):
        # Get the selected elements
        element1 = select_element1.value
        element2 = select_element2.value

        # Get the data for the selected elements
        df = get_scatter_data_for_element(element1, element2)

        # Update the plot's title and source
        if not df.empty:
            p.title.text = f"Scatter Plot of {element1} vs {element2} in Granite"
            source.data = ColumnDataSource.from_df(df)
        else:
            p.title.text = "No data available for selected elements"
            source.data = {}

    # Set up callbacks to update the plot when dropdown values change
    select_element1.on_change('value', update_plot)
    select_element2.on_change('value', update_plot)

    # Create the initial plot with default elements
    df_initial = get_scatter_data_for_element('U', 'Th')
    source = ColumnDataSource(df_initial)
    p = figure(width=800, height=600, title=f"Scatter Plot of U vs Th in Granite",
               x_axis_label="U (ppm)", y_axis_label="Th (ppm)")

    # Add a scatter renderer
    p.circle(x='U', y='Th', source=source, size=8, color="navy", alpha=0.5)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [("U", "@U ppm"), ("Th", "@Th ppm")]
    p.add_tools(hover)

    # Layout the dropdowns and the plot
    layout = column(row(select_element1, select_element2), p)

    # Create a panel with the layout
    panel = Panel(child=layout, title="Scatter Plot")

    return panel

def display_uranium_statistics_table():
    stats = get_stats_uranium_concentration()
    if stats is None:
        return Div(text="No data available for Uranium Statistics table.")

    # Reset index to make 'lithology' a column
    stats_df = stats.reset_index()

    # Create a ColumnDataSource from the statistics DataFrame
    source = ColumnDataSource(stats_df)

    # Define the columns for the DataTable
    columns = [
        TableColumn(field="lithology", title="Lithology"),
        TableColumn(field="Average U (ppm)", title="Average U (ppm)"),
        TableColumn(field="Standard Deviation", title="Standard Deviation"),
        TableColumn(field="Min U (ppm)", title="Min U (ppm)"),
        TableColumn(field="Max U (ppm)", title="Max U (ppm)")
    ]

    # Create the DataTable
    data_table = DataTable(source=source, columns=columns, width=800, height=400)

    return data_table

# Function to create a tab for the statistics table
def create_statistics_table_tab():
    table = display_uranium_statistics_table()
    table_tab = Panel(child=table, title="Uranium Statistics")
    return table_tab
