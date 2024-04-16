#========================================LIBRARIES========================================
from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
from dash import dcc
import dash_leaflet as dl
import plotly.express as px
import plotly.graph_objects as go
from dash import callback_context
from dash import dcc
from dash import html
from google.cloud import storage
from io import BytesIO


#========================================LOAD DATA========================================

def load_data_from_gcs(bucket_name, blob_name):
    """Load data from a Google Cloud Storage bucket into a pandas DataFrame."""
    # Initialize a client
    client = storage.Client()
    # Access the bucket
    bucket = client.bucket(bucket_name)
    # Access the blob (file) within the bucket
    blob = bucket.blob(blob_name)
    # Download the contents of the blob as bytes
    data = blob.download_as_bytes()
    # Convert the bytes data to a pandas DataFrame
    df = pd.read_csv(BytesIO(data), encoding='latin-1', low_memory=False)
    return df


# Load the equipment data
equipment_df = load_data_from_gcs('dinex_bucket', 'merged_equipment1.csv')
equipment_df['year'] = pd.to_datetime(equipment_df['Model Yr'], format='%Y', errors='coerce').dt.year  # Converts 'Model Yr' column to datetime object keeping only the year

# Load the sales data
sales_df = load_data_from_gcs('dinex_bucket', 'clean_sales1.csv')


#========================================INITIALIZE APP========================================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


#========================================LAYOUTS========================================

def layout():
    return html.Div([
            html.Br(),
            html.H3('U.S. Truck Database: 2010 - 2023'),
            # "View Sales" button added here
            html.Button('View Sales', id='view_sales_button', n_clicks=0, style={'margin': '10px'}),
            dl.Map(
                center=[37.0902, -95.7129],  # Center of the U.S.
                zoom=4,  # Initial zoom level
                children=[
                    dl.TileLayer(),  # Default OpenStreetMap tile layer
                    dl.LayerGroup(id='sales_layer'),  # LayerGroup for sales data
                    dl.LayerGroup(id='truck_layer'),  # LayerGroup for truck data
                ],
                style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"},
                id='map_country'
    ),

        # Legend for the map
        html.Div([
            html.H4('Legend:'),
            html.Div([
                html.Span(style={'background-color': 'red', 'padding': '5px 10px', 'margin-right': '10px', 'border-radius': '50%'}),
                'Truck Data'
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
            html.Div([
                html.Span(style={'background-color': 'blue', 'padding': '5px 10px', 'margin-right': '10px', 'border-radius': '50%'}),
                'Sales Data'
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'position': 'absolute', 'top': '100px', 'right': '20px', 'padding': '10px', 'background': 'white', 'border-radius': '5px', 'box-shadow': '0 0 10px rgba(0,0,0,0.5)', 'z-index': '1000'}),

    html.Div([
        dcc.RangeSlider(
            id='years',
            min=2010,
            max=2023,
            dots=True,
            value=[2010, 2023],
            marks={str(yr): "'" + str(yr)[2:] for yr in range(2010, 2023)}
        ),
        html.Br(), html.Br(),
    ], style={'width': '75%', 'margin-left': '12%', 'background-color': '#eeeeee'}),

    html.Div([
        dcc.Dropdown(
            id='states',
            multi=True,
            value=[''],
            placeholder='Select State',
            # Assume 'equipment_df' is defined elsewhere in your Dash app
            options=[{'label': c, 'value': c} for c in sorted(equipment_df['State'].unique())]
        )
    ], style={'width': '50%', 'margin-left': '25%', 'background-color': '#eeeeee'}),
    

    # Inserting the truck_model_counts_graph right below the map
    dcc.Graph(id='truck_brand_counts_graph'),

    dcc.Graph(id='gvwr_class_counts_graph'),

    dcc.Graph(id='engine_brand_counts_graph'),
    
    dcc.Graph(id='top_truck_model_graph'),

    dcc.Graph(id='model_year_distro_graph'),

    dcc.Graph(id='gvwr_class_distro_graph'),
    
    dcc.Graph(id='engine_brand_distro_graph'),
    
    dcc.Graph(id='truck_brand_distro_graph'),


   
], style={'background-color': '#eeeeee', 'font-family': 'Palatino'})
    return layout

# Callback to update page content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return layout()  # Return the layout function you defined
    # You can add more conditions here to handle different URLs
    else:
        return html.Div([
            html.H3('404 Page not found'),
            html.P('The page you are looking for does not exist.')
        ])

# Make sure to set the app's layout with the 'app.layout' at the top
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

#========================================CALLBACKS========================================

# callback for updating truck and sales data on map
@app.callback(
    [Output('sales_layer', 'children'),
     Output('truck_layer', 'children')],
    [Input('states', 'value'),
     Input('years', 'value'),
     Input('view_sales_button', 'n_clicks')])
def update_map_layers(states, year_range, n_clicks):
    ctx = callback_context

    # Initialize empty lists for sales and truck markers
    sales_markers = []
    truck_markers = []

    # Check which input triggered the callback
    if not ctx.triggered:
        # If no input was triggered, return empty layers
        return sales_markers, truck_markers
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'view_sales_button':
        # Logic for plotting sales data, excluding sales with '0' value
        sales_markers = [
            dl.CircleMarker(center=[row['latitude'], row['longitude']],  # No offset added
                            radius=5,
                            color='blue',
                            fillColor='blue',
                            fillOpacity=0.5,
                            children=[dl.Tooltip(f"Sales: {row['Sales']}")])
            for idx, row in sales_df.iterrows() if row['Sales'] != 0
        ]

    # Update truck data if states or years inputs were triggered
    if trigger_id in ['states', 'years']:
        # Logic for plotting truck data based on states and year range
        start_year, end_year = max(2010, year_range[0]), min(2023, year_range[1]) if year_range else (2010, 2023)
        df_filtered = equipment_df[
            (equipment_df['State'].isin(states)) &
            (equipment_df['Model Yr'].between(start_year, end_year))
        ]
        truck_markers = [
            dl.CircleMarker(center=[row['latitude'], row['longitude']],
                            radius=5,
                            color='red',
                            fillColor='red',
                            fillOpacity=0.5,
                            children=[dl.Tooltip(f"{row['County']}, {row['State']}: {row['Model Yr']}")])
            for idx, row in df_filtered.iterrows()
        ]

    return sales_markers, truck_markers


# generates bar graph of truck brand counts in selected states
@app.callback(
    Output('truck_brand_counts_graph', 'figure'),
    [Input('states', 'value')])
def update_truck_brand_counts(selected_states):
    # Filter the DataFrame based on selected states, if any
    if selected_states:
        filtered_df = equipment_df[equipment_df['State'].isin(selected_states)]
    else:
        filtered_df = equipment_df

    # Group by 'Eqt Brand', sum the 'VIN' counts, and get the top 10
    truck_counts = filtered_df.groupby('Eqt Brand')['VIN'].sum().sort_values(ascending=False).head(10)

    # Convert Series to DataFrame for Plotly
    truck_counts_df = truck_counts.reset_index()
    truck_counts_df.columns = ['Eqt Brand', 'Count']  # Rename columns for clarity

    # Generate a bar plot using Plotly Express
    fig = px.bar(truck_counts_df, 
                 x='Eqt Brand', 
                 y='Count', 
                 labels={'Eqt Brand': 'Truck Manufacturer', 'Count': 'Count'}, 
                 color_discrete_sequence=['steelblue'] )

    fig.update_layout(title_text='Truck Manufacturer Counts', 
                      xaxis_title='Truck Manufacturer', 
                      yaxis_title='Count', 
                      xaxis_tickangle=-45)  # Rotate labels

    return fig


# generates bar graph of gvwr counts in each state
@app.callback(
    Output('gvwr_class_counts_graph', 'figure'),
    [Input('states', 'value')])
def update_gvwr_class_counts(selected_states):
    # Filter the DataFrame based on selected states, if any
    if selected_states:
        filtered_df = equipment_df[equipment_df['State'].isin(selected_states)]
    else:
        filtered_df = equipment_df

    # Group by 'GVWR Class', sum the 'VIN' counts, and get the top 10
    gvwr_class_counts = filtered_df.groupby('GVWR Class')['VIN'].sum().sort_values(ascending=False).head(10)

    # Convert Series to DataFrame for Plotly
    gvwr_counts_df = gvwr_class_counts.reset_index()
    gvwr_counts_df.columns = ['GVWR Class', 'Count']  # Rename columns for clarity

    # Generate a bar plot using Plotly Express with the corrected DataFrame
    fig = px.bar(gvwr_counts_df, 
                 x='GVWR Class', 
                 y='Count', 
                 labels={'GVWR Class': 'GVWR Class', 'Count': 'Count'}, 
                 color_discrete_sequence=['crimson'] ) 

    fig.update_layout(title_text='GVWR Class Counts', 
                      xaxis_title='GVWR Class', 
                      yaxis_title='Count', 
                      xaxis_tickangle=-45)  # Rotate labels

    return fig


#generates a bar graph for engine brand counts by state
@app.callback(
    Output('engine_brand_counts_graph', 'figure'),
    [Input('states', 'value')])
def engine_brand_count(selected_states):
    # Filter the DataFrame based on selected states, if any
    if selected_states:
        filtered_df = equipment_df[equipment_df['State'].isin(selected_states)]
    else:
        filtered_df = equipment_df

    # Group by 'engine brand', sum the 'VIN' counts, and get the top 10
    engine_brand_counts = filtered_df.groupby('Engine Brand')['VIN'].sum().sort_values(ascending=False).head(10)

    # Convert Series to DataFrame for Plotly
    engine_brand_counts_df = engine_brand_counts.reset_index()
    engine_brand_counts_df.columns = ['Engine Brand', 'Count'] # Rename columns for clarity

    # Generate a bar plot using Plotly Express
    fig = px.bar(engine_brand_counts_df, 
                 x='Engine Brand', 
                 y='Count', 
                 labels={'Engine Brand': 'Engine Brand', 'Count': 'Count'}, 
                 color_discrete_sequence=['forestgreen'] )

    fig.update_layout(title_text='Engine Brand Counts', 
                      xaxis_title='Engine Brand', 
                      yaxis_title='Count', 
                      xaxis_tickangle=-45)  # Rotate labels

    return fig


#generates a bar graph for top truck models by state
@app.callback(
    Output('top_truck_model_graph', 'figure'),
    [Input('states', 'value')])
def top_truck_model(selected_states):
    # Filter the DataFrame based on selected states, if any
    if selected_states:
        filtered_df = equipment_df[equipment_df['State'].isin(selected_states)]
    else:
        filtered_df = equipment_df

    # Group by 'truck models', sum the 'VIN' counts, and get the top 10
    top_truck_models = filtered_df.groupby('Model')['VIN'].sum().sort_values(ascending=False).head(20)

    # Convert Series to DataFrame for Plotly
    truck_model_counts_df = top_truck_models.reset_index() 
    truck_model_counts_df.columns = ['Eqt Brand', 'Count'] # Rename columns for clarity

    # Generate a bar plot using Plotly Express
    fig = px.bar(truck_model_counts_df, 
                 x='Eqt Brand', 
                 y='Count', 
                 color_discrete_sequence=['blueviolet'] )

    fig.update_layout(title_text='Top Truck Models', 
                      xaxis_title='Truck Models', 
                      yaxis_title='Count', 
                      xaxis_tickangle=-45)  # Rotate labels

    return fig


#generates a bar graph for truck age distro by state
@app.callback(
    Output('model_year_distro_graph', 'figure'),  # Ensure this matches the id of your dcc.Graph component
    [Input('states', 'value'),
     Input('years', 'value')])  # Adding Input for the year range slider
def model_year_distros(selected_states, selected_years):
    # Filter the DataFrame based on selected states, if any
    if selected_states:
        filtered_df = equipment_df[equipment_df['State'].isin(selected_states)]
    else:
        filtered_df = equipment_df

    # Further filter the DataFrame based on the selected year range from the range slider
    start_year, end_year = selected_years  # Unpack the start and end year from the selected_years range slider value
    filtered_df = filtered_df[(filtered_df['Model Yr'] >= start_year) & (filtered_df['Model Yr'] <= end_year)]

    # Group by 'Model Yr', count the 'VIN', and get the counts for each model year
    model_year_distro = filtered_df.groupby('Model Yr')['VIN'].count().sort_values(ascending=False)

    # Convert Series to DataFrame for Plotly
    model_year_distro_df = model_year_distro.reset_index()
    model_year_distro_df.columns = ['Model Yr', 'Count'] # Rename columns for clarity

    # Generate a bar plot using Plotly Express
    fig = px.bar(model_year_distro_df, 
                 x='Model Yr', 
                 y='Count', 
                 color_discrete_sequence=['coral'])

    fig.update_layout(title_text='Model Year Distribution', 
                      xaxis_title='Year', 
                      yaxis_title='Count', 
                      xaxis_tickangle=-45)  # Rotate labels

    return fig


#generate stacked bar graph of gvwr class and model year
@app.callback(
    Output('gvwr_class_distro_graph', 'figure'),  # Ensure this id matches your dcc.Graph component for the GVWR Class Distribution
    [Input('years', 'value')])  # Assuming you want to filter by years selected from a range slider
def update_gvwr_graph(selected_years):
    # Filter the DataFrame based on the selected year range
    start_year, end_year = selected_years
    filtered_df = equipment_df[(equipment_df['Model Yr'] >= start_year) & (equipment_df['Model Yr'] <= end_year)]
    
    # Group the data by 'Year' and 'GVWR Class' and sum the 'VIN' values
    gvwr_data = filtered_df.groupby(['Model Yr', 'GVWR Class'])['VIN'].sum().unstack(fill_value=0)

    # Creating the stacked bar chart using Plotly
    fig = go.Figure()
    for gvwr_class in gvwr_data.columns:
        fig.add_trace(go.Bar(
            x=gvwr_data.index,
            y=gvwr_data[gvwr_class],
            name=str(gvwr_class)
        ))

    # Update the layout for a stacked bar chart
    fig.update_layout(
        barmode='stack',
        title_text='GVWR Class Distribution by Year',
        xaxis_title='Year',
        yaxis_title='Total Number of Trucks',
        xaxis_tickangle=-45,
        legend_title='GVWR Class',
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


#generate stacked bar graph of truck brand distro and model year
@app.callback(
    Output('truck_brand_distro_graph', 'figure'),  # Ensure this id matches your dcc.Graph component for the GVWR Class Distribution
    [Input('years', 'value')])  # Assuming you want to filter by years selected from a range slider
def update_truck_brand_graph(selected_years):
    # Filter the DataFrame based on the selected year range
    start_year, end_year = selected_years
    filtered_df = equipment_df[(equipment_df['Model Yr'] >= start_year) & (equipment_df['Model Yr'] <= end_year)]
    
    # Group the data by 'Year' and 'GVWR Class' and sum the 'VIN' values
    truck_brand_data = filtered_df.groupby(['Model Yr', 'Eqt Brand'])['VIN'].sum().unstack(fill_value=0)

    # Creating the stacked bar chart using Plotly
    fig = go.Figure()
    for truck_brand in truck_brand_data.columns:
        fig.add_trace(go.Bar(
            x=truck_brand_data.index,
            y=truck_brand_data[truck_brand],
            name=str(truck_brand)
        ))

    # Update the layout for a stacked bar chart
    fig.update_layout(
        barmode='stack',
        title_text='Truck Manufacturer Distribution by Year',
        xaxis_title='Year',
        yaxis_title='Total Number of Trucks',
        xaxis_tickangle=-45,
        legend_title='Truck Brands',
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig


#generate stacked bar graph of engine brand distro and model year
@app.callback(
    Output('engine_brand_distro_graph', 'figure'),  # Ensure this id matches your dcc.Graph component for the GVWR Class Distribution
    [Input('years', 'value')])  # Assuming you want to filter by years selected from a range slider
def update_engine_brand_graph(selected_years):
    # Filter the DataFrame based on the selected year range
    start_year, end_year = selected_years
    filtered_df = equipment_df[(equipment_df['Model Yr'] >= start_year) & (equipment_df['Model Yr'] <= end_year)]
    
    # Group the data by 'Year' and 'GVWR Class' and sum the 'VIN' values
    engine_brand_data = filtered_df.groupby(['Model Yr', 'Engine Brand'])['VIN'].sum().unstack(fill_value=0)

    # Creating the stacked bar chart using Plotly
    fig = go.Figure()
    for engine_brand in engine_brand_data.columns:
        fig.add_trace(go.Bar(
            x=engine_brand_data.index,
            y=engine_brand_data[engine_brand],
            name=str(engine_brand)
        ))

    # Update the layout for a stacked bar chart
    fig.update_layout(
        barmode='stack',
        title_text='Engine Brand Distribution by Year',
        xaxis_title='Year',
        yaxis_title='Total Number of Trucks',
        xaxis_tickangle=-45,
        legend_title='Engine Brands',
        margin=dict(l=20, r=20, t=30, b=20),
    )

    return fig



# Main entry point
if __name__ == '__main__':
    app.run_server(debug=True)
