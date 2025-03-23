import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import folium
from streamlit_folium import st_folium  # Change from folium_static to st_folium
import requests
from folium.plugins import HeatMap

# Check if statsmodels is installed for trendline functionality
try:
    import statsmodels.api as sm
    STATSMODELS_INSTALLED = True
except ImportError:
    STATSMODELS_INSTALLED = False

# Set page title and layout
st.set_page_config(layout="wide", page_title="US State Crime Data Analysis")


def change_chart_type():
    st.session_state.chart_type = st.session_state.chart_type_select
    
    
# Function to load data
@st.cache_data
def load_data():
    url = "https://corgis-edu.github.io/corgis/datasets/csv/state_crime/state_crime.csv"
    df = pd.read_csv(url)
    return df

# Load data
df = load_data()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Table", "Descriptive Statistics", "Charts"])

# Tab 1: Overview
with tab1:
    st.title("US State Crime Data Overview")
    st.write("""
    This application provides an interactive visualization of crime data across US states over time.
    The dataset contains information about various types of crimes, population, and crime rates.
    """)
    
    # Add data reference
    st.write("""
    **Data Source:** [CORGIS Dataset Project - State Crime Data](https://corgis-edu.github.io/corgis/csv/state_crime/)
    """)
    
    # Show Data Summary first
    st.subheader("Dataset Summary")
    st.write(f"- **Years covered:** {df['Year'].min()} to {df['Year'].max()}")
    st.write(f"- **Number of states:** {df['State'].nunique()}")
    st.write(f"- **Total records:** {len(df)}")
    
    # Now show data preview
    st.subheader("Data Preview")
    st.write("Below are the first 5 rows from the dataset to provide a glimpse of the actual data:")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Add Data Dictionary 
    st.subheader("Data Dictionary")
    st.write("""
    The data dictionary provides information about each column in the dataset, including its data type and description.
    """)
    
    # Create more concise column descriptions
    col_desc = {
        "State": "String - Name of the US state",
        "Year": "Integer - Year of the report",
        "Data.Population": "Integer - Total state population",
        
        "Data.Rates.Property.All": "Float - All property crimes per 100,000 people",
        "Data.Rates.Property.Burglary": "Float - Burglaries per 100,000 people",
        "Data.Rates.Property.Larceny": "Float - Larcenies per 100,000 people", 
        "Data.Rates.Property.Motor": "Float - Motor vehicle thefts per 100,000 people",
        
        "Data.Rates.Violent.All": "Float - All violent crimes per 100,000 people",
        "Data.Rates.Violent.Assault": "Float - Assaults per 100,000 people",
        "Data.Rates.Violent.Murder": "Float - Murders per 100,000 people",
        "Data.Rates.Violent.Rape": "Float - Rapes per 100,000 people",
        "Data.Rates.Violent.Robbery": "Float - Robberies per 100,000 people",
        
        "Data.Totals.Property.All": "Integer - Total number of property crimes",
        "Data.Totals.Property.Burglary": "Integer - Total number of burglaries",
        "Data.Totals.Property.Larceny": "Integer - Total number of larcenies",
        "Data.Totals.Property.Motor": "Integer - Total number of motor vehicle thefts",
        
        "Data.Totals.Violent.All": "Integer - Total number of violent crimes",
        "Data.Totals.Violent.Assault": "Integer - Total number of assaults",
        "Data.Totals.Violent.Murder": "Integer - Total number of murders",
        "Data.Totals.Violent.Rape": "Integer - Total number of rapes",
        "Data.Totals.Violent.Robbery": "Integer - Total number of robberies"
    }
    
    # Display as two columns
    col1, col2 = st.columns(2)
    
    # Split the dictionary into two parts for better display
    items = list(col_desc.items())
    half_point = len(items) // 2
    
    with col1:
        for col, desc in items[:half_point]:
            st.write(f"**{col}**: {desc}")
    
    with col2:
        for col, desc in items[half_point:]:
            st.write(f"**{col}**: {desc}")

# Tab 2: Data Table
with tab2:
    st.title("Data Table")
    
    # Add filtering options
    col1, col2 = st.columns(2)
    with col1:
        filter_state = st.multiselect(
            "Filter by State",
            options=sorted(df["State"].unique()),
            default=[],
            key="data_table_filter_state"  # Add unique key
        )
    
    with col2:
        filter_year = st.multiselect(
            "Filter by Year",
            options=sorted(df["Year"].unique()),
            default=[],
            key="data_table_filter_year"  # Add unique key
        )
    
    # Apply filters
    filtered_df = df.copy()
    if filter_state:
        filtered_df = filtered_df[filtered_df["State"].isin(filter_state)]
    if filter_year:
        filtered_df = filtered_df[filtered_df["Year"].isin(filter_year)]
    
    # Add sorting functionality and pagination in a single row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sort_col = st.selectbox("Sort by column", options=df.columns.tolist(), key="data_table_sort_col")
    with col2:
        sort_order = st.radio("Sort order", options=["Ascending", "Descending"], horizontal=True, key="data_table_sort_order")
    with col3:
        # Calculate pagination info
        rows_per_page = 15
        total_pages = max(1, len(filtered_df) // rows_per_page + (1 if len(filtered_df) % rows_per_page > 0 else 0))
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))
    
    # Calculate start and end indices for pagination
    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(sorted_df))
    
    st.dataframe(sorted_df.iloc[start_idx:end_idx], use_container_width=True)
    
    st.write(f"Showing {start_idx+1}-{end_idx} of {len(sorted_df)} records (Page {page} of {total_pages})")

# Tab 3: Descriptive Statistics
with tab3:
    st.title("Descriptive Statistics")
    
    st.write("""
    Descriptive statistics summarize and quantify the main features of the dataset, helping to understand the 
    distribution and characteristics of the data. Below are the key statistical measures for the numeric variables 
    in this dataset:
    
    - **Mean**: The average value, calculated by summing all values and dividing by the count
    - **Median**: The middle value when data is ordered from lowest to highest
    - **Mode**: The most frequently occurring value in the dataset
    - **Standard Deviation**: Measures the amount of variation or dispersion in the dataset
    - **Variance**: The square of standard deviation, indicating how far values are spread out from the mean
    - **Minimum**: The smallest value in the dataset
    - **Maximum**: The largest value in the dataset
    """)
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate statistics with limited decimal places
    stats_df = pd.DataFrame({
        'Mean': df[numeric_cols].mean().round(2),
        'Median': df[numeric_cols].median().round(2),
        'Mode': [round(df[col].mode()[0], 2) if not df[col].mode().empty else None for col in numeric_cols],
        'Standard Deviation': df[numeric_cols].std().round(2),
        'Variance': df[numeric_cols].var().round(2),
        'Min': df[numeric_cols].min().round(2),
        'Max': df[numeric_cols].max().round(2)
    })
    
    st.dataframe(stats_df, use_container_width=True)

# Tab 4: Charts
with tab4:
    st.title("Interactive Data Visualization")
    
    # Create a layout with two columns
    col1, col2 = st.columns([1, 3])
    
    # Left column for controls
    with col1:
        

        # Move chart descriptions to the top of the left column
        
        chart_descriptions = {
            "Bar Chart": "Bar charts are useful for comparing values across categories. They work well with a smaller number of discrete categories.",
            "Stacked Bar Chart": "Stacked bar charts show the composition of categories and how different components contribute to the total.",
            "Line Chart": "Line charts show trends over time and are ideal for displaying continuous data over a time period.",
            "Histogram": "Histograms display the distribution of a dataset, showing how many data points fall within certain ranges.",
            "Pie Chart": "Pie charts show proportional distribution and work best with a small number of categories that add up to a meaningful whole.",
            "Scatter Plot": "Scatter plots show the relationship between two variables and are useful for identifying correlations and patterns.",
            "Choropleth Map": "Choropleth maps use color intensity to represent data values across geographic areas, making spatial patterns more visible.",
            "Heat Map": "Heat maps visualize data through color variation in a matrix format, showing patterns and trends across two categorical dimensions (states and years)."
        }
        
        # show an infobox based on the chart type selected
        #

        
        # Implement session state to persist selections across reruns
        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = "Bar Chart"
        st.info(f"{chart_descriptions.get(st.session_state.chart_type, 'Select a chart type to see its description')}")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Stacked Bar Chart", "Line Chart", "Histogram", 
             "Pie Chart", "Scatter Plot", "Choropleth Map", "Heat Map"],
            key="chart_type_select", on_change=change_chart_type
        )
        
        # Update session state when chart type changes
        st.session_state.chart_type = chart_type
        
        # Common filters
        states = sorted(df["State"].unique().tolist())
        # Remove "United States" from the states list if present
        if "United States" in states:
            states.remove("United States")
            
        select_all_states = st.checkbox("Select All States", value=True, key="chart_select_all_states")
        if select_all_states:
            selected_states = states
        else:
            selected_states = st.multiselect("Select States", states, default=states[:5])
        
        # Replace year range slider with a single year selection
        years = sorted(df["Year"].unique().tolist())
        min_year = min(years)
        max_year = max(years)
        
        # Move the interactive year slider to the left column
        if chart_type == "Heat Map":
            st.write("*Heat Map visualizes data across all years automatically*")
        elif chart_type in ["Line Chart", "Scatter Plot"]:
            st.write("*This chart type shows data across all years*")
        else:
            # Add the year slider directly in the left column
            selected_year = st.slider(
                "Drag to change year:",
                min_value=min_year,
                max_value=max_year,
                value=max_year,  # Default to most recent year
                step=1,
                format="%d",  # Format as integer
                key=f"main_year_slider_{chart_type}"
            )
        
        # For line chart, add option to aggregate all states, default to True
        if chart_type == "Line Chart":
            aggregate_states = st.checkbox("Aggregate all states (single trend line)", value=True, key="line_chart_aggregate_states")
            aggregation_method = st.radio(
                "Aggregation method:",
                ["Mean (Average)", "Sum (Total)"],
                horizontal=True,
                disabled=not aggregate_states,
                key="line_chart_aggregation_method"
            )
            
        # Metric selection based on chart type
        if chart_type in ["Bar Chart", "Stacked Bar Chart", "Line Chart", "Histogram", "Choropleth Map"]:
            metric_groups = {
                "Population": ["Data.Population"],
                "Violent Crime Rates": [col for col in df.columns if "Data.Rates.Violent" in col],
                "Property Crime Rates": [col for col in df.columns if "Data.Rates.Property" in col],
                "Violent Crime Totals": [col for col in df.columns if "Data.Totals.Violent" in col],
                "Property Crime Totals": [col for col in df.columns if "Data.Totals.Property" in col]
            }
            
            # Remove Population from options for Stacked Bar Chart
            if chart_type == "Stacked Bar Chart":
                metric_groups.pop("Population", None)
            
            metric_group = st.selectbox("Select Metric Group", list(metric_groups.keys()), key=f"metric_group_{chart_type}")
            
            # Only show the specific metric selection for chart types other than Stacked Bar Chart
            if chart_type != "Stacked Bar Chart":
                selected_metric = st.selectbox("Select Specific Metric", metric_groups[metric_group], key=f"specific_metric_{chart_type}")
            else:
                # For Stacked Bar Chart, we'll use the first metric in the group (All) by default
                selected_metric = [col for col in metric_groups[metric_group] if ".All" in col][0]
                # Display a message about how the stacked bar chart works

        # For scatter plot only (removed bubble chart)
        if chart_type in ["Scatter Plot"]:
            x_metric = st.selectbox("Select X Metric", df.select_dtypes(include=[np.number]).columns.tolist(), index=0, key="scatter_x_metric")
            y_metric = st.selectbox("Select Y Metric", df.select_dtypes(include=[np.number]).columns.tolist(), index=1, key="scatter_y_metric")
        
        # For pie chart
        if chart_type == "Pie Chart":
            pie_metric_groups = {
                "Population": ["Data.Population"],
                "Violent Crime Totals": [col for col in df.columns if "Data.Totals.Violent" in col],
                "Property Crime Totals": [col for col in df.columns if "Data.Totals.Property" in col]
            }
            pie_metric_group = st.selectbox("Select Metric Group for Pie Chart", list(pie_metric_groups.keys()), key="pie_metric_group")
            pie_metric = st.selectbox("Select Specific Metric for Pie Chart", pie_metric_groups[pie_metric_group], key="pie_specific_metric")
            
            # Replace the year selection with top N states selector
            top_n = st.slider(
                "Number of states to show individually",
                min_value=3,
                max_value=15,
                value=6,
                step=1,
                key="pie_chart_top_n"
            )
            
            # Other options for pie chart
            show_percentage = st.checkbox("Show percentages on pie chart", value=True, key="pie_show_percentage")

        # For choropleth map (renamed from heatmap)
        if chart_type == "Choropleth Map":
            map_year = st.selectbox("Select Year for Map", years, index=len(years)-1, key="map_year")
            # Remove the metric group selection and reuse the selected_metric from above
        
        # For heat map
        if chart_type == "Heat Map":
            # Default to violent crime rates
            heatmap_default_idx = next((i for i, x in enumerate(df.columns) if x == "Data.Rates.Violent.All"), 0)
            heatmap_metric = st.selectbox(
                "Select Metric for Heat Map", 
                df.select_dtypes(include=[np.number]).columns.tolist(),
                index=heatmap_default_idx,
                key="heatmap_metric"
            )

    # Right column for chart visualization
    with col2:
        
        # Filter data based on selections
        if not selected_states:
            st.warning("Please select at least one state")
            filtered_df = pd.DataFrame()
        else:
            # Apply state filter
            filtered_df = df[df["State"].isin(selected_states)]
            
            # For charts that use a specific year, apply that filter
            if chart_type not in ["Heat Map", "Line Chart", "Scatter Plot"]:
                filtered_df = filtered_df[filtered_df["Year"] == selected_year]
        
        # Display different charts based on selection
        if not filtered_df.empty:
            if chart_type == "Bar Chart":
                # Layout for chart and data table side by side
                chart_col, data_col = st.columns([3, 1])
                
                with chart_col:
                    # Filter to specific year and just group by state
                    chart_data = filtered_df.groupby("State")[selected_metric].mean().reset_index()
                    chart_title = f"{selected_metric} by State ({selected_year})"
                    chart_data = chart_data.sort_values(selected_metric, ascending=False)
                    
                    fig = px.bar(chart_data, x="State", y=selected_metric, 
                                title=chart_title,
                                color="State")
                    
                    fig.update_layout(xaxis_title="State", yaxis_title=selected_metric)
                    st.plotly_chart(fig, use_container_width=True)
                
                with data_col:
                    st.subheader("Data Table")
                    # Format the data for display
                    display_data = chart_data.copy()
                    
                    # Round numeric columns to 2 decimal places
                    numeric_cols = display_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        display_data[col] = display_data[col].round(2)
                    
                    st.dataframe(display_data, use_container_width=True)
            
            elif chart_type == "Stacked Bar Chart":
                # For stacked bar chart, we always need to work with component metrics
                # Get the base category (Property/Violent and Rates/Totals)
                crime_category = ""
                data_type = ""
                
                if "Property" in selected_metric:
                    crime_category = "Property"
                    if "Rates" in selected_metric:
                        data_type = "Rates"
                    else:
                        data_type = "Totals"
                elif "Violent" in selected_metric:
                    crime_category = "Violent"
                    if "Rates" in selected_metric:
                        data_type = "Rates"
                    else:
                        data_type = "Totals"
                
                # Now, find the component metrics regardless of which specific metric was selected
                metrics = [col for col in df.columns if f"Data.{data_type}.{crime_category}" in col 
                          and "All" not in col]
                
                # Create a more readable title based on the selected metric
                if "All" in selected_metric:
                    chart_subtitle = f"All {crime_category} Crime {data_type}"
                else:
                    # Extract the specific crime type from the selected metric
                    specific_crime = selected_metric.split(".")[-1].capitalize()
                    chart_subtitle = f"{crime_category} Crime {data_type} - {specific_crime}"
                
                # Filter to specific year
                chart_title = f"Breakdown of {chart_subtitle} by State ({selected_year})"
                chart_data = filtered_df.groupby("State")[metrics].mean().reset_index()
                
                # Clean up metric names for the legend
                melted_data = pd.melt(chart_data, id_vars=["State"], value_vars=metrics, 
                                    var_name="Crime Type", value_name="Value")
                
                # Extract cleaner crime type names for display in the legend
                melted_data["Crime Type"] = melted_data["Crime Type"].apply(
                    lambda x: x.split(".")[-1].capitalize() if "." in x else x
                )
                
                # Ensure consistent colors for crime types
                crime_colors = {
                    "Assault": "#1f77b4",  # blue
                    "Murder": "#d62728",   # red
                    "Rape": "#ff7f0e",     # orange
                    "Robbery": "#2ca02c",  # green
                    "Burglary": "#9467bd", # purple
                    "Larceny": "#8c564b",  # brown
                    "Motor": "#e377c2"     # pink
                }
                
                # Layout for chart and data table side by side
                chart_col, data_col = st.columns([3, 1])
                
                with chart_col:
                    # For stacked bar chart, add y-axis selection
                    stacked_y_option = st.radio(
                        "Y-Axis Display",
                        ["Absolute Values", "Percentage (100% Stacked)"],
                        horizontal=True,
                        key=f"stacked_y_option_{selected_year}"  # Make key unique with selected_year
                    )
                    
                    # Determine whether to use absolute values or percentages
                    if 'stacked_y_option' in locals() and stacked_y_option == "Percentage (100% Stacked)":
                        fig = px.bar(melted_data, x="State", y="Value", color="Crime Type",
                                    title=chart_title,
                                    barmode='relative',  # 100% stacked bar
                                    color_discrete_map=crime_colors)
                    else:
                        fig = px.bar(melted_data, x="State", y="Value", color="Crime Type",
                                    title=chart_title,
                                    color_discrete_map=crime_colors)
                    
                    y_axis_title = "Percentage (%)" if 'stacked_y_option' in locals() and stacked_y_option == "Percentage (100% Stacked)" else "Value"
                    fig.update_layout(
                        xaxis_title="State", 
                        yaxis_title=y_axis_title,
                        legend_title="Crime Type"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"stacked_bar_chart_{selected_year}")
                
                with data_col:
                    st.subheader("Data Table")
                    
                    # Create a summary table showing totals for each type per state
                    pivot_data = chart_data.copy()
                    
                    # Round values for better display
                    for col in pivot_data.columns:
                        if col != "State":
                            pivot_data[col] = pivot_data[col].round(2)
                    
                    # Display the data table
                    st.dataframe(pivot_data, use_container_width=True, key=f"stacked_bar_data_{selected_year}")
                    
                    # Add explanation based on y-axis selection
                    if stacked_y_option == "Percentage (100% Stacked)":
                        st.info("Showing relative proportions of each crime type as percentages.")
                    else:
                        st.info("Showing absolute values for each crime type.")
            
            elif chart_type == "Line Chart":
                # Layout for chart and data table side by side
                chart_col, data_col = st.columns([3, 1])
                
                with chart_col:
                    if aggregate_states:
                        # Aggregate all states based on selected method
                        if aggregation_method == "Sum (Total)":
                            chart_data = filtered_df.groupby("Year")[selected_metric].sum().reset_index()
                            title_prefix = "Total"
                        else:  # Mean (Average)
                            chart_data = filtered_df.groupby("Year")[selected_metric].mean().reset_index()
                            title_prefix = "Average"
                        
                        fig = px.line(chart_data, x="Year", y=selected_metric,
                                    title=f"{title_prefix} {selected_metric} Over Time (All Selected States)")
                        
                        # Make the single trend line more prominent
                        fig.update_traces(line=dict(width=3, color='#0072B2'))
                        
                    else:
                        # Show individual trend line for each state
                        chart_data = filtered_df.groupby(["Year", "State"])[selected_metric].mean().reset_index()
                        
                        fig = px.line(chart_data, x="Year", y=selected_metric, color="State",
                                    title=f"Trend of {selected_metric} Over Time by State")
                    
                    fig.update_layout(xaxis_title="Year", yaxis_title=selected_metric)
                    st.plotly_chart(fig, use_container_width=True, key="line_chart")
                    
                    # Add explanation for aggregated line chart
                    if aggregate_states:
                        if aggregation_method == "Sum (Total)":
                            st.info("""
                            This chart shows the total sum across all selected states combined.
                            Each data point represents the total value across states for that year.
                            """)
                        else:
                            st.info("""
                            This chart shows the average trend across all selected states combined.
                            Each data point represents the mean value across states for that year.
                            """)
                
                with data_col:
                    st.subheader("Data Table")
                    
                    # Create a display version of the data
                    if aggregate_states:
                        # For aggregated view, use the already created chart_data
                        display_data = chart_data.copy()
                        display_data = display_data.sort_values(by="Year")
                    else:
                        # For individual state view, create a pivot table for better display
                        # First take mean by Year and State
                        temp_data = filtered_df.groupby(["Year", "State"])[selected_metric].mean().reset_index()
                        # Then pivot to show years as rows and states as columns
                        display_data = temp_data.pivot(index="Year", columns="State", values=selected_metric).reset_index()
                    
                    # Round values for better display
                    for col in display_data.columns:
                        if col != "Year" and col != "State":
                            display_data[col] = display_data[col].round(2)
                    
                    # Display the data table
                    st.dataframe(display_data, use_container_width=True, key="line_chart_data")
                
            elif chart_type == "Histogram":
                # Add bin size control
                bin_options = [5, 10, 15, 20, 25, 30, "auto"]
                bin_size = st.select_slider(
                    "Adjust bin size:",
                    options=bin_options,
                    value="auto",
                    key=f"histogram_bin_size_{selected_year}"
                )
                
                # Convert bin_size to int for plotly if not "auto"
                nbins = None if bin_size == "auto" else int(bin_size)
                
                # Create columns for histogram and stats side by side
                hist_col, stat_col = st.columns([3, 1])
                
                with hist_col:
                    # Create histogram without color breakdown by state
                    fig = px.histogram(
                        filtered_df, 
                        x=selected_metric,
                        title=f"Distribution of {selected_metric} ({selected_year})",
                        nbins=nbins  # Apply bin size
                    )
                    
                    # Add borders to histogram bars
                    fig.update_traces(
                        marker=dict(
                            line=dict(width=1, color='black')
                        )
                    )
                    
                    fig.update_layout(
                        xaxis_title=selected_metric,
                        yaxis_title="Count"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"histogram_{selected_year}")
                
                with stat_col:
                    # Add statistical summary for the selected metric
                    st.subheader("Statistical Summary")
                    
                    # Calculate statistics
                    mean_val = filtered_df[selected_metric].mean()
                    median_val = filtered_df[selected_metric].median()
                    std_dev = filtered_df[selected_metric].std()
                    min_val = filtered_df[selected_metric].min()
                    max_val = filtered_df[selected_metric].max()
                    count = len(filtered_df)
                    
                    # Display as formatted text instead of styled boxes for better clarity
                    st.write(f"**Mean:** {mean_val:.2f}")
                    st.write(f"**Median:** {median_val:.2f}")
                    st.write(f"**Standard Deviation:** {std_dev:.2f}")
                    st.write(f"**Min:** {min_val:.2f}")
                    st.write(f"**Max:** {max_val:.2f}")
                    st.write(f"**Count:** {count}")
                    
                    # Calculate additional statistics that might be useful
                    q1 = filtered_df[selected_metric].quantile(0.25)
                    q3 = filtered_df[selected_metric].quantile(0.75)
                    iqr = q3 - q1
                    
                    st.write(f"**1st Quartile (Q1):** {q1:.2f}")
                    st.write(f"**3rd Quartile (Q3):** {q3:.2f}")
                    st.write(f"**IQR:** {iqr:.2f}")
                
            elif chart_type == "Pie Chart":
                # Layout for chart and data table side by side
                chart_col, data_col = st.columns([3, 1])
                
                with chart_col:
                    # Get top N states by the selected metric, where N is user-defined
                    top_states = filtered_df.nlargest(top_n, pie_metric)["State"].tolist()
                    
                    # Create a new DataFrame with "Others" category
                    pie_chart_data = filtered_df.copy()
                    pie_chart_data.loc[~pie_chart_data["State"].isin(top_states), "State"] = "Others"
                    
                    # Aggregate the data
                    pie_chart_data = pie_chart_data.groupby("State")[pie_metric].sum().reset_index()
                    
                    # Set up and display pie chart with customizations
                    fig = px.pie(
                        pie_chart_data, 
                        values=pie_metric, 
                        names="State",
                        title=f"Distribution of {pie_metric} by State ({selected_year})"
                    )
                    
                    # Customize the pie chart based on user preferences
                    if show_percentage:
                        fig.update_traces(textinfo='percent+label')
                    else:
                        fig.update_traces(textinfo='label')
                    
                    # Add hover data with values
                    fig.update_traces(hovertemplate='<b>%{label}</b><br>Value: %{value:,.2f}<br>Percentage: %{percent:.1%}')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with data_col:
                    st.subheader("Data Table")
                    
                    # Sort the data by value for better display
                    display_data = pie_chart_data.copy()
                    display_data = display_data.sort_values(by=pie_metric, ascending=False)
                    
                    # Round numeric values for better display
                    display_data[pie_metric] = display_data[pie_metric].round(2)
                    
                    # Add a percentage column
                    total_value = display_data[pie_metric].sum()
                    display_data['Percentage'] = ((display_data[pie_metric] / total_value) * 100).round(2)
                    display_data['Percentage'] = display_data['Percentage'].apply(lambda x: f"{x}%")
                    
                    # Display the data table
                    st.dataframe(display_data, use_container_width=True, key=f"pie_chart_data_{selected_year}")
                    
                    # Add an explanation
                    if "Others" in display_data["State"].values:
                        st.info(f"""
                        **Note**: The pie chart shows the top {top_n} states individually.
                        The "Others" category aggregates data from the remaining states 
                        to improve chart readability.
                        """)
                    
                    # Add summary statistics
                    st.subheader("Summary")
                    st.write(f"**Total {pie_metric}:** {total_value:,.2f}")
                    st.write(f"**Number of states shown individually:** {len(display_data) - ('Others' in display_data['State'].values)}")
                    if "Others" in display_data["State"].values:
                        others_value = display_data.loc[display_data["State"] == "Others", pie_metric].values[0]
                        others_pct = (others_value / total_value * 100)
                        st.write(f"**'Others' category represents:** {others_value:,.2f} ({others_pct:.1f}%)")

            elif chart_type == "Scatter Plot":
                # Configure the scatter plot
                options_col, chart_col = st.columns([1, 3])
                
                with options_col:
                    # Add options for marker size, transparency, and trendline
                    marker_size = st.slider("Marker Size", min_value=5, max_value=20, value=10, key="scatter_marker_size")
                    marker_opacity = st.slider("Marker Opacity", min_value=0.1, max_value=1.0, value=0.7, key="scatter_opacity")
                    
                    # Only show trendline option if statsmodels is installed
                    if STATSMODELS_INSTALLED:
                        show_trendline = st.checkbox("Show Trendline", value=True, key="scatter_trendline")
                        trendline_type = "ols" if show_trendline else None
                    else:
                        st.warning("Statsmodels not installed. Install with:\n```pip install statsmodels```")
                        show_trendline = False
                        trendline_type = None
                
                with chart_col:
                    try:
                        # Attempt to create scatter plot with or without trendline
                        if select_all_states or len(selected_states) == len(states):
                            scatter_kwargs = {
                                'x': x_metric,
                                'y': y_metric,
                                'hover_name': "State",
                                'hover_data': ["Year"],
                                'title': f"Relationship between {x_metric} and {y_metric}",
                                'opacity': marker_opacity
                            }
                            # Only add trendline if statsmodels is installed and trendline is requested
                            if STATSMODELS_INSTALLED and show_trendline:
                                scatter_kwargs['trendline'] = trendline_type
                                
                            fig = px.scatter(filtered_df, **scatter_kwargs)
                            # Use a consistent color for all points
                            fig.update_traces(marker=dict(color='#1f77b4', size=marker_size))  # A nice blue color
                        else:
                            # Use different colors for different states
                            scatter_kwargs = {
                                'x': x_metric,
                                'y': y_metric,
                                'color': "State",
                                'hover_name': "State",
                                'hover_data': ["Year"],
                                'title': f"Relationship between {x_metric} and {y_metric}",
                                'opacity': marker_opacity
                            }
                            # Only add trendline if statsmodels is installed and trendline is requested
                            if STATSMODELS_INSTALLED and show_trendline:
                                scatter_kwargs['trendline'] = trendline_type
                                
                            fig = px.scatter(filtered_df, **scatter_kwargs)
                            fig.update_traces(marker=dict(size=marker_size))
                        
                        fig.update_layout(xaxis_title=x_metric, yaxis_title=y_metric)
                        st.plotly_chart(fig, use_container_width=True, key="scatter_plot")
                        
                        # Display basic correlation statistics without statsmodels
                        corr = filtered_df[[x_metric, y_metric]].corr().iloc[0,1]
                        st.write(f"**Correlation Coefficient:** {corr:.3f}")
                        
                        # Interpret the correlation
                        if abs(corr) < 0.3:
                            correlation_strength = "weak"
                        elif abs(corr) < 0.7:
                            correlation_strength = "moderate"
                        else:
                            correlation_strength = "strong"
                            
                        correlation_direction = "positive" if corr > 0 else "negative"
                        st.write(f"The data shows a {correlation_strength} {correlation_direction} correlation.")
                        
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")
                        st.write("Try selecting different metrics or states to visualize the data.")
            
            elif chart_type == "Choropleth Map":
                # Get data for the selected year without filtering by state
                map_data = df[df["Year"] == selected_year].copy()
                
                # Remove "United States" if present for more accurate visualization
                if "United States" in map_data["State"].values:
                    map_data = map_data[map_data["State"] != "United States"]
                
                # Create a display version of the metric for population values
                display_metric = selected_metric
                display_unit = ""
                
                # If the selected metric is population, convert to thousands for better display
                if "Population" in selected_metric:
                    map_data["display_value"] = map_data[selected_metric] / 1000
                    display_metric = "display_value"
                    display_unit = " (thousands)"
                
                # Create side-by-side columns for map and data
                map_col, data_col = st.columns([3, 2])
                
                with map_col:
                    metric_label = f"{selected_metric}{display_unit}"
                    st.subheader(f"Choropleth Map for {metric_label} ({selected_year})")
                    
                    # Load US states geojson
                    @st.cache_data
                    def load_geojson():
                        url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
                        response = requests.get(url)
                        return response.json()
                    
                    geojson_data = load_geojson()
                    
                    # Calculate min and max values for proper color scaling
                    vmin = map_data[display_metric].min()
                    vmax = map_data[display_metric].max()
                    
                    # Create a folium map
                    m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
                    
                    # Create choropleth map with proper color scaling
                    choropleth = folium.Choropleth(
                        geo_data=geojson_data,
                        name='choropleth',
                        data=map_data,
                        columns=['State', display_metric],
                        key_on='feature.properties.name',
                        fill_color='YlOrRd',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name=metric_label,
                        threshold_scale=np.linspace(vmin, vmax, 8).tolist(),  # Create 8 color bins between min and max
                    ).add_to(m)
                    
                    # Add tooltips to the choropleth with both state name and metric value
                    choropleth.geojson.add_child(
                        folium.features.GeoJsonTooltip(
                            fields=['name'],
                            aliases=['State:'],
                            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                        )
                    )
                    
                    # Add layer control to toggle layers
                    folium.LayerControl().add_to(m)
                    
                    # Display the map in Streamlit - replace folium_static with st_folium
                    st_folium(m, width=800, height=500, key=f"choropleth_map_{selected_year}")
                
                with data_col:
                    st.subheader(f"Data for {selected_year}")
                    
                    # Format the value column to 2 decimal places
                    display_data = map_data[['State', selected_metric]].copy()
                    
                    # Handle population metrics for display
                    if "Population" in selected_metric:
                        display_data["Population (thousands)"] = (display_data[selected_metric] / 1000).round(2)
                        display_col = "Population (thousands)"
                        # Drop the original column to avoid confusion
                        display_data.drop(columns=[selected_metric], inplace=True)
                    else:
                        display_data[selected_metric] = display_data[selected_metric].round(2)
                        display_col = selected_metric
                    
                    # Sort the data by value
                    sorted_data = display_data.sort_values(by=display_col, ascending=False)
                    st.dataframe(sorted_data, use_container_width=True, key=f"choropleth_data_{selected_year}")
                    
                    # Add some statistics about the data
                    st.subheader("Statistical Summary")
                    
                    # Use the appropriate column for statistics
                    stats_col = display_metric if "Population" in selected_metric else selected_metric
                    
                    stats = {
                        "Maximum Value": f"{map_data[stats_col].max():.2f} ({map_data.loc[map_data[stats_col].idxmax(), 'State']})",
                        "Minimum Value": f"{map_data[stats_col].min():.2f} ({map_data.loc[map_data[stats_col].idxmin(), 'State']})",
                        "Average": f"{map_data[stats_col].mean():.2f}",
                        "Median": f"{map_data[stats_col].median():.2f}",
                    }
                    for stat, value in stats.items():
                        st.write(f"**{stat}:** {value}")
                    
                    # Add a histogram showing the distribution with borders
                    st.subheader("Distribution")
                    hist_title = f"Distribution of {selected_metric}{display_unit}"
                    hist_fig = px.histogram(map_data, x=display_metric,
                                          title=hist_title)
                    # Add borders to histogram bars
                    hist_fig.update_traces(
                        marker=dict(
                            line=dict(width=1, color='black')
                        )
                    )
                    hist_fig.update_layout(height=250)
                    st.plotly_chart(hist_fig, use_container_width=True, key=f"choropleth_hist_{selected_year}")
            
            elif chart_type == "Heat Map":
                # Heat Map already shows all years, so we don't need a year slider here
                # Filter for all states and all years
                heatmap_data = df.copy()
                
                # Remove "United States" if present for more accurate visualization
                if "United States" in heatmap_data["State"].values:
                    heatmap_data = heatmap_data[heatmap_data["State"] != "United States"]
                
                # Filter to selected states only
                heatmap_data = heatmap_data[heatmap_data["State"].isin(selected_states)]
                
                # Use a try-except block to handle potential errors with the pivot operation
                try:
                    # Pivot the data to create a matrix suitable for a heatmap
                    pivot_data = heatmap_data.pivot_table(
                        values=heatmap_metric,
                        index="State",
                        columns="Year"
                    )
                    
                    # Sort the states by their average value to group similar states together
                    state_avg = heatmap_data.groupby("State")[heatmap_metric].mean()
                    pivot_data = pivot_data.reindex(state_avg.sort_values(ascending=False).index)
                    
                    # Calculate min and max values for better color range
                    vmin = pivot_data.values.min()
                    vmax = pivot_data.values.max()
                    
                    # Create a custom color scale with more contrast
                    # Use a diverging color scale with higher contrast
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x="Year", y="State", color=heatmap_metric),
                        title=f"Heat Map of {heatmap_metric} Across States and Years",
                        # Use a higher contrast color scale
                        color_continuous_scale='RdBu_r',
                        aspect="auto",  # Adjust aspect ratio to fit the display
                        zmin=vmin,  # Set explicit min value
                        zmax=vmax,  # Set explicit max value
                        color_continuous_midpoint=np.median(pivot_data.values.flatten())  # Set midpoint at median
                    )
                    
                    # Add hover text showing the exact values and improve layout
                    fig.update_layout(
                        height=600,
                        xaxis=dict(tickmode='linear', dtick=5),  # Show year ticks every 5 years
                        coloraxis_colorbar=dict(
                            title=heatmap_metric,
                            thicknessmode="pixels", thickness=20,
                            lenmode="pixels", len=400,
                            title_side="right",
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="heat_map")
                    
                    # Let the user know about the color scale change
                    st.info(f"""
                    **Color Scale Interpretation:**
                    The heat map uses a diverging color scale where:
                    - Dark red indicates values higher than the median ({np.median(pivot_data.values.flatten()):.2f})
                    - Dark blue indicates values lower than the median
                    - States are sorted by their average {heatmap_metric} value (high to low)
                    
                    This makes it easier to see differences between states and identify trends over time.
                    """)
                    
                except Exception as e:
                    st.error(f"Error creating heat map: {str(e)}")
                    st.write("This may occur if there's insufficient data for the selected states.")




