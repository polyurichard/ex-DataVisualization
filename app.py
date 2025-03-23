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

# Set page title and layout
st.set_page_config(layout="wide", page_title="US State Crime Data Analysis")

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
            default=[]
        )
    
    with col2:
        filter_year = st.multiselect(
            "Filter by Year",
            options=sorted(df["Year"].unique()),
            default=[]
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
        sort_col = st.selectbox("Sort by column", options=df.columns.tolist())
    with col2:
        sort_order = st.radio("Sort order", options=["Ascending", "Descending"], horizontal=True)
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
        st.subheader("Chart Controls")
        
        # Implement session state to persist selections across reruns
        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = "Bar Chart"
        
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Stacked Bar Chart", "Line Chart", "Histogram", 
             "Pie Chart", "Scatter Plot", "Choropleth Map", "Heat Map"],
            key="chart_type_select"
        )
        
        # Update session state when chart type changes
        st.session_state.chart_type = chart_type
        
        # Common filters
        states = sorted(df["State"].unique().tolist())
        # Remove "United States" from the states list if present
        if "United States" in states:
            states.remove("United States")
            
        select_all_states = st.checkbox("Select All States", value=True)  # Default to checked
        if select_all_states:
            selected_states = states
            # Hide the multiselect when all are selected
        else:
            selected_states = st.multiselect("Select States", states, default=states[:5])
        
        # Replace year range slider with year selection
        years = sorted(df["Year"].unique().tolist())
        # Only show year selection if appropriate for the chart type
        if chart_type in ["Bar Chart", "Stacked Bar Chart", "Pie Chart", "Choropleth Map"]:
            selected_year = st.selectbox(
                "Select Year",
                options=["All Years"] + years,
                index=0,  # Default to "All Years"
            )
        elif chart_type == "Heat Map":
            # For Heat Map, set the year to "All Years" implicitly but don't show the widget
            selected_year = "All Years"
            st.write("*Heat Map visualizes data across all years automatically*")
        else:
            # For other chart types
            selected_year = "All Years"
        
        # For line chart, add option to aggregate all states, default to True
        if chart_type == "Line Chart":
            aggregate_states = st.checkbox("Aggregate all states (single trend line)", value=True)
            aggregation_method = st.radio(
                "Aggregation method:",
                ["Mean (Average)", "Sum (Total)"],
                horizontal=True,
                disabled=not aggregate_states
            )
            
        # Metric selection based on chart type
        if chart_type in ["Bar Chart", "Stacked Bar Chart", "Line Chart", "Histogram", "Choropleth Map"]:
            # For these charts, we need to select a metric to visualize
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
            
            metric_group = st.selectbox("Select Metric Group", list(metric_groups.keys()))
            
            # Only show the specific metric selection for chart types other than Stacked Bar Chart
            if chart_type != "Stacked Bar Chart":
                selected_metric = st.selectbox("Select Specific Metric", metric_groups[metric_group])
            else:
                # For Stacked Bar Chart, we'll use the first metric in the group (All) by default
                selected_metric = [col for col in metric_groups[metric_group] if ".All" in col][0]
                # Display a message about how the stacked bar chart works
                st.info("Stacked Bar Chart shows the breakdown of components for the selected crime category.")

        # For scatter plot only (removed bubble chart)
        if chart_type in ["Scatter Plot"]:
            x_metric = st.selectbox("Select X Metric", df.select_dtypes(include=[np.number]).columns.tolist(), index=0)
            y_metric = st.selectbox("Select Y Metric", df.select_dtypes(include=[np.number]).columns.tolist(), index=1)
        
        # For pie chart
        if chart_type == "Pie Chart":
            pie_metric_groups = {
                "Population": ["Data.Population"],
                "Violent Crime Totals": [col for col in df.columns if "Data.Totals.Violent" in col],
                "Property Crime Totals": [col for col in df.columns if "Data.Totals.Property" in col]
            }
            pie_metric_group = st.selectbox("Select Metric Group for Pie Chart", list(pie_metric_groups.keys()))
            pie_metric = st.selectbox("Select Specific Metric for Pie Chart", pie_metric_groups[pie_metric_group])
            pie_year = st.selectbox("Select Year for Pie Chart", years, index=len(years)-1)
        
        # For choropleth map (renamed from heatmap)
        if chart_type == "Choropleth Map":
            map_year = st.selectbox("Select Year for Map", years, index=len(years)-1)
            # Remove the metric group selection and reuse the selected_metric from above
        
        # For stacked bar chart, add y-axis selection
        if chart_type == "Stacked Bar Chart":
            stacked_y_option = st.radio(
                "Y-Axis Display",
                ["Absolute Values", "Percentage (100% Stacked)"],
                horizontal=True
            )
        
        # For heat map
        if chart_type == "Heat Map":
            # Default to violent crime rates
            heatmap_default_idx = next((i for i, x in enumerate(df.columns) if x == "Data.Rates.Violent.All"), 0)
            heatmap_metric = st.selectbox(
                "Select Metric for Heat Map", 
                df.select_dtypes(include=[np.number]).columns.tolist(),
                index=heatmap_default_idx
            )

        # For bar and stacked bar chart, add year aggregation explanation if All Years selected
        if chart_type in ["Bar Chart", "Stacked Bar Chart"] and selected_year == "All Years":
            st.info("Showing average values across all years")

    # Right column for chart description and visualization
    with col2:
        st.subheader("Chart Visualization")
        
        # Display chart description based on selection
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
        
        st.info(chart_descriptions[chart_type])
        
        # Filter data based on selections
        if not selected_states:
            st.warning("Please select at least one state")
            filtered_df = pd.DataFrame()
        else:
            # Apply state filter
            filtered_df = df[df["State"].isin(selected_states)]
            
            # Apply year filter if a specific year is selected and not viewing Heat Map
            if selected_year != "All Years":
                filtered_df = filtered_df[filtered_df["Year"] == selected_year]
        
        # Display different charts based on selection
        if not filtered_df.empty:
            if chart_type == "Bar Chart":
                if selected_year == "All Years":
                    # Group by state and take the mean of the selected metric across all years
                    chart_title = f"Average {selected_metric} by State (All Years)"
                    chart_data = filtered_df.groupby("State")[selected_metric].mean().reset_index()
                else:
                    # Filter to specific year and just group by state
                    year_data = filtered_df[filtered_df["Year"] == int(selected_year)]
                    chart_title = f"{selected_metric} by State ({selected_year})"
                    chart_data = year_data.groupby("State")[selected_metric].mean().reset_index()
                
                chart_data = chart_data.sort_values(selected_metric, ascending=False)
                
                fig = px.bar(chart_data, x="State", y=selected_metric, 
                            title=chart_title,
                            color="State")
                
                fig.update_layout(xaxis_title="State", yaxis_title=selected_metric)
                st.plotly_chart(fig, use_container_width=True)
            
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
                
                if selected_year == "All Years":
                    # Average across all years
                    chart_title = f"Breakdown of {chart_subtitle} by State (All Years)"
                    chart_data = filtered_df.groupby("State")[metrics].mean().reset_index()
                else:
                    # Filter to specific year
                    year_data = filtered_df[filtered_df["Year"] == int(selected_year)]
                    chart_title = f"Breakdown of {chart_subtitle} by State ({selected_year})"
                    chart_data = year_data.groupby("State")[metrics].mean().reset_index()
                
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation based on y-axis selection
                if 'stacked_y_option' in locals() and stacked_y_option == "Percentage (100% Stacked)":
                    st.info("""
                    This 100% stacked bar chart shows the relative proportion of each crime type as a percentage.
                    Each bar represents 100% of the crimes for that state, allowing for easier comparison of proportions across states.
                    """)
                
            elif chart_type == "Line Chart":
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
                st.plotly_chart(fig, use_container_width=True)
                
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
                
            elif chart_type == "Histogram":
                # Create columns for histogram and stats side by side
                hist_col, stat_col = st.columns([3, 1])
                
                with hist_col:
                    # Create histogram without color breakdown by state
                    fig = px.histogram(filtered_df, x=selected_metric,
                                      title=f"Distribution of {selected_metric}")
                    
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with stat_col:
                    # Add statistical summary for the selected metric
                    st.subheader("Statistical Summary")
                    hist_stats = {
                        "Mean": filtered_df[selected_metric].mean().round(2),
                        "Median": filtered_df[selected_metric].median().round(2),
                        "Std Dev": filtered_df[selected_metric].std().round(2),
                        "Min": filtered_df[selected_metric].min().round(2),
                        "Max": filtered_df[selected_metric].max().round(2),
                        "Count": len(filtered_df)
                    }
                    
                    # Create a styled display for the statistics
                    st.markdown("""
                    <style>
                    .stat-box {
                        background-color: #f0f2f6;
                        border-radius: 5px;
                        padding: 10px;
                        margin-bottom: 10px;
                    }
                    .stat-label {
                        font-weight: bold;
                        color: #0e1117;
                    }
                    .stat-value {
                        font-size: 18px;
                        color: #0078ff;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    for stat, value in hist_stats.items():
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-label">{stat}</div>
                            <div class="stat-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
            elif chart_type == "Pie Chart":
                # Filter to specific year for pie chart
                pie_data = filtered_df[filtered_df["Year"] == pie_year]
                
                # Get top 6 states by the selected metric
                top_states = pie_data.nlargest(6, pie_metric)["State"].tolist()
                
                # Create a new DataFrame with "Others" category
                pie_chart_data = pie_data.copy()
                pie_chart_data.loc[~pie_chart_data["State"].isin(top_states), "State"] = "Others"
                
                # Aggregate the data
                pie_chart_data = pie_chart_data.groupby("State")[pie_metric].sum().reset_index()
                
                fig = px.pie(pie_chart_data, values=pie_metric, names="State",
                            title=f"Distribution of {pie_metric} by State in {pie_year}")
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Scatter Plot":
                # If all states are selected and the "Select All States" option is checked,
                # use a single color for all points
                if select_all_states or len(selected_states) == len(states):
                    fig = px.scatter(filtered_df, x=x_metric, y=y_metric,
                                    hover_name="State", hover_data=["Year"],
                                    title=f"Relationship between {x_metric} and {y_metric}")
                    # Use a consistent color for all points
                    fig.update_traces(marker=dict(color='#1f77b4'))  # A nice blue color
                else:
                    # Use different colors for different states
                    fig = px.scatter(filtered_df, x=x_metric, y=y_metric, color="State",
                                    hover_name="State", hover_data=["Year"],
                                    title=f"Relationship between {x_metric} and {y_metric}")
                
                fig.update_layout(xaxis_title=x_metric, yaxis_title=y_metric)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Choropleth Map":
                # Get data for the selected year without filtering by state
                map_data = df[df["Year"] == map_year].copy()
                
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
                    st.subheader(f"Choropleth Map for {metric_label} ({map_year})")
                    
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
                    st_folium(m, width=800, height=500)
                
                with data_col:
                    st.subheader(f"Data for {map_year}")
                    
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
                    st.dataframe(sorted_data, use_container_width=True)
                    
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
                    st.plotly_chart(hist_fig, use_container_width=True)
            
            elif chart_type == "Heat Map":
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
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




