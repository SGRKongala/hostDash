# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer
from sqlalchemy import create_engine, text
import os
from sqlalchemy import create_engine

# Fetch the database URL, and ensure it is valid
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check Render environment variables.")

pg_engine = create_engine(DATABASE_URL)

# Data Loading and Preprocessing
def load_data():
    # Load main datasets
    df = pd.read_sql('SELECT * FROM main_data', pg_engine)
    df_rpm = pd.read_sql('SELECT * FROM rpm', pg_engine)
    metric = 'std_dev'
    df1 = pd.read_sql(f'SELECT * FROM {metric}', pg_engine)
    
    # Merge dataframes
    merged_df1 = pd.merge(df, df1, on='id', how='inner')
    merged_df2 = pd.merge(df, df_rpm, on='id', how='inner')
    
    # Convert time columns
    merged_df1['time'] = pd.to_datetime(merged_df1['time'])
    merged_df2['time'] = pd.to_datetime(merged_df2['time'])
    
    return merged_df1, merged_df2, metric

# Load data
merged_df1, merged_df2, metric = load_data()

# Constants
SENSORS = ['s1', 's2', 's3', 's4', 's5', 's6']
BINS = np.arange(0, 18, 0.5)
CHANNELS = ['ch1', 'ch2', 'ch3']
COLORS = {'ch1': 'blue', 'ch2': 'red', 'ch3': 'green'}

# Calculate default y-limits
def calculate_y_limits():
    all_values = []
    for ch in CHANNELS:
        for s in SENSORS:
            col = f'{ch}{s}'
            all_values.extend(merged_df1[col].dropna().values)
    return np.percentile(all_values, [2.5, 97.5])

y_min, y_max = calculate_y_limits()

# Initialize Dash app
app = dash.Dash(__name__)

# App Layout
app.layout = html.Div([
    html.H1("Sensor Data Analysis Dashboard"),
    
    # Control Panel
    html.Div([
        # Sensor Selection
        html.Div([
            html.H3("Select Sensor"),
            dcc.Dropdown(
                id='sensor-dropdown',
                options=[{'label': f'Sensor {s}', 'value': s} for s in SENSORS],
                value=SENSORS[0],
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        # RPM Selection
        html.Div([
            html.H3("Select RPM Bin"),
            dcc.Dropdown(
                id='rpm-dropdown',
                options=[{'label': f'{b}-{b+0.5} RPM', 'value': b} for b in BINS[:-1]],
                value=10.0,
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        # Moving Average Control
        html.Div([
            html.H3("Moving Average Window (Days)"),
            dcc.Slider(
                id='ma-slider',
                min=1, max=30, step=1, value=1,
                marks={i: str(i) for i in [1,7,14,21,30]},
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ]),
    
    # Y-Axis Controls
    html.Div([
        html.H3("Y-Axis Limits"),
        html.Div([
            dcc.Input(id='y-min-input', type='number', value=y_min, step=0.1,
                     style={'width': '100px', 'marginRight': '10px'}),
            dcc.Input(id='y-max-input', type='number', value=y_max, step=0.1,
                     style={'width': '100px'})
        ])
    ], style={'marginTop': '20px'}),
    
    # Date Range Selection
    html.Div([
        html.H3("Select Date Range"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=merged_df1['time'].min().date(),
            end_date=merged_df1['time'].max().date(),
            display_format='YYYY-MM-DD'
        )
    ], style={'marginTop': '20px'}),
    
    # Graph and Download Section
    dcc.Graph(id='sensor-graph'),
    html.Button("Download Graph", id="btn-download"),
    dcc.Download(id="download-graph")
])

# Callbacks
@app.callback(
    Output('sensor-graph', 'figure'),
    [Input('sensor-dropdown', 'value'),
     Input('rpm-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('y-min-input', 'value'),
     Input('y-max-input', 'value'),
     Input('ma-slider', 'value')]
)
def update_graph(selected_sensor, rpm_bin, start_date, end_date, y_min, y_max, ma_days):
    # Date filtering
    mask = (merged_df1['time'].dt.date >= pd.to_datetime(start_date).date()) & \
           (merged_df1['time'].dt.date <= pd.to_datetime(end_date).date())
    df_filtered = merged_df1[mask].sort_values('time')
    
    # RPM filtering
    rpm_mask = (merged_df2['ch1s1'] >= rpm_bin) & (merged_df2['ch1s1'] < (rpm_bin + 0.5))
    rpm_filtered = merged_df2[rpm_mask].sort_values('time')
    
    # Combine filtered data
    final_df = pd.merge(df_filtered, rpm_filtered[['id', 'time']], on=['id', 'time'])
    final_df = final_df.sort_values('time')
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each channel
    for ch in CHANNELS:
        col_name = f'{ch}{selected_sensor}'
        ma_window = f'{ma_days}D'
        ma_data = final_df.set_index('time')[col_name].resample('D').mean().rolling(
            window=ma_days, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=ma_data.index,
            y=ma_data.values,
            mode='lines+markers',
            name=f'Channel {ch} ({ma_days}-day MA)',
            line=dict(color=COLORS[ch], width=1.5, shape='linear'),
            marker=dict(color=COLORS[ch], size=5),
            connectgaps=True,
            opacity=0.6
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{metric} - Sensor {selected_sensor} Data for RPM {rpm_bin}-{rpm_bin+0.5} ({ma_days}-day Moving Average)',
        xaxis_title='Time',
        yaxis_title='Value',
        yaxis=dict(range=[y_min, y_max]),
        showlegend=True,
        height=600,
        legend_title='Channel'
    )
    
    return fig

@app.callback(
    Output("download-graph", "data"),
    Input("btn-download", "n_clicks"),
    [State('sensor-dropdown', 'value'),
     State('rpm-dropdown', 'value'),
     State('ma-slider', 'value'),
     State('sensor-graph', 'figure')],
    prevent_initial_call=True
)
def download_graph(n_clicks, selected_sensor, rpm_bin, ma_days, figure):
    if n_clicks:
        filename = f'{metric}_Sensor_{selected_sensor}_RPM_{rpm_bin}-{rpm_bin+0.5}_MA_{ma_days}days.png'
        img_bytes = go.Figure(figure).to_image(
            format='png',
            width=1920,
            height=1080,
            scale=2.0,
            engine='kaleido'
        )
        return dcc.send_bytes(img_bytes, filename)

# Run server
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
