import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL = "postgresql://user:password@host:port/dbname"  # Paste Render DB URL

engine = create_engine(DATABASE_URL)
conn = engine.connect()

df = pd.read_sql('SELECT * FROM main_data', conn)
df_rpm = pd.read_sql('SELECT * FROM rpm', conn)
metric = 'std_Dev'
df1 = pd.read_sql(f'SELECT * FROM {metric}', conn)
conn.close()

# Merge DataFrames
merged_df1 = pd.merge(df, df1, on='id', how='inner')
merged_df2 = pd.merge(df, df_rpm, on='id', how='inner')

# Convert time column to datetime
merged_df1['time'] = pd.to_datetime(merged_df1['time'])
merged_df2['time'] = pd.to_datetime(merged_df2['time'])

# Define Sensors and RPM Bins
sensors = ['s1', 's2', 's3', 's4', 's5', 's6']
bins = np.arange(0, 18, 0.5)

# Calculate y-limits based on 95% of data
all_values = []
for ch in ['ch1', 'ch2', 'ch3']:
    for s in sensors:
        col = f'{ch}{s}'
        all_values.extend(merged_df1[col].dropna().values)
y_min, y_max = np.percentile(all_values, [2.5, 97.5])

# Create Dash App
app = dash.Dash(__name__)
server = app.server  # Required for Gunicorn

# App Layout
app.layout = html.Div([
    html.H1("Sensor Data Analysis Dashboard"),
    
    html.Div([
        html.Div([
            html.H3("Select Sensor"),
            dcc.Dropdown(
                id='sensor-dropdown',
                options=[{'label': f'Sensor {s}', 'value': s} for s in sensors],
                value=sensors[0],
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H3("Select RPM Bin"),
            dcc.Dropdown(
                id='rpm-dropdown',
                options=[{'label': f'{b}-{b+0.5} RPM', 'value': b} for b in bins[:-1]],
                value=10.0,  
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H3("Moving Average Window (Days)"),
            dcc.Slider(
                id='ma-slider',
                min=1, max=30, step=1, value=1,
                marks={i: str(i) for i in [1,7,14,21,30]},
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.H3("Y-Axis Limits"),
        html.Div([
            dcc.Input(id='y-min-input', type='number', value=y_min, step=0.1, style={'width': '100px', 'marginRight': '10px'}),
            dcc.Input(id='y-max-input', type='number', value=y_max, step=0.1, style={'width': '100px'})
        ])
    ], style={'marginTop': '20px'}),
    
    html.Div([
        html.H3("Select Date Range"),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=merged_df1['time'].min().date(),
            end_date=merged_df1['time'].max().date(),
            display_format='YYYY-MM-DD'
        )
    ], style={'marginTop': '20px'}),
    
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
    mask = (merged_df1['time'].dt.date >= pd.to_datetime(start_date).date()) & \
           (merged_df1['time'].dt.date <= pd.to_datetime(end_date).date())
    df_filtered = merged_df1[mask].sort_values('time')

    rpm_mask = (merged_df2['ch1s1'] >= rpm_bin) & (merged_df2['ch1s1'] < (rpm_bin + 0.5))
    rpm_filtered = merged_df2[rpm_mask].sort_values('time')

    final_df = pd.merge(df_filtered, rpm_filtered[['id', 'time']], on=['id', 'time'])
    final_df = final_df.sort_values('time')

    fig = go.Figure()
    colors = {'ch1': 'blue', 'ch2': 'red', 'ch3': 'green'}

    for ch in ['ch1', 'ch2', 'ch3']:
        col_name = f'{ch}{selected_sensor}'
        ma_data = final_df.set_index('time')[col_name].resample('D').mean().rolling(window=ma_days, min_periods=1).mean()
        
        fig.add_trace(go.Scatter(
            x=ma_data.index, y=ma_data.values, mode='lines+markers',
            name=f'Channel {ch} ({ma_days}-day MA)',
            line=dict(color=colors[ch], width=1.5, shape='linear'),
            marker=dict(color=colors[ch], size=5),
            connectgaps=True, opacity=0.6
        ))

    fig.update_layout(
        title=f'{metric} - Sensor {selected_sensor} Data for RPM {rpm_bin}-{rpm_bin+0.5} ({ma_days}-day Moving Average)',
        xaxis_title='Time', yaxis_title='Value',
        yaxis=dict(range=[y_min, y_max]), showlegend=True, height=600,
        legend_title='Channel'
    )
    
    return fig

# Deploy with Gunicorn
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=5000)
