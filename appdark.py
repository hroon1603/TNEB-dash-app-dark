import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
import calendar


file_path = 'modified_electricity_demand_dataset.csv'
df = pd.read_csv(file_path)

#formatting time data
df['Day'] = pd.to_datetime(df['Day'])
df['Hour_of_Day'] = pd.to_numeric(df['Hour_of_Day'])


df['Week'] = df['Day'].dt.to_period('W').apply(lambda r: r.start_time)

# week naming function
def month_week_label(date):
    month_name = date.strftime('%B')
    _, days_in_month = calendar.monthrange(date.year, date.month)
    
    day_of_month = date.day
    if day_of_month <= days_in_month // 4:
        week_of_month = 1
    elif day_of_month <= 2 * (days_in_month // 4):
        week_of_month = 2
    elif day_of_month <= 3 * (days_in_month // 4):
        week_of_month = 3
    else:
        week_of_month = 4
    
    return f"{month_name}: Week {week_of_month}"


df['Week_Label'] = df['Day'].apply(month_week_label)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# prediction model defs
def rolling_average(data, window_size=3):
    return data.rolling(window=window_size, min_periods=1).mean()

def exponential_smoothing(data, smoothing_level=0.2):
    model = SimpleExpSmoothing(data).fit(smoothing_level=smoothing_level)
    return model.fittedvalues

def arima_prediction(data, steps=7):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


def create_layout():
    return dbc.Container(
        fluid=True,
        style={'backgroundColor': '#34495e'},
        children=[
            dbc.Row(
                dbc.Col(html.H1("TNEB Utility Dashboard", className='text-center', style={'color': '#ecf0f1'}), className='mt-4 mb-4'),
                justify='center'
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id='week-selector',
                                    options=[{'label': label, 'value': str(week)} for label, week in zip(df['Week_Label'].unique(), df['Week'].unique())],
                                    value=str(df['Week'].unique()[0]),
                                    placeholder="Week",
                                    style={'width': '300px', 'color': '#2c3e50'},
                                ),
                                width="auto",
                                style={'padding-right': '10px'},
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='algorithm-selector',
                                    options=[
                                        {'label': 'Past - Simple', 'value': 'rolling'},
                                        {'label': 'Past - Exponential Smoothing', 'value': 'exponential'},
                                        {'label': 'Future Prediction - ARIMA', 'value': 'arima'}
                                    ],
                                    value='rolling',
                                    placeholder="Prediction Algorithm",
                                    style={'width': '300px', 'color': '#2c3e50'},
                                ),
                                width="auto",
                            ),
                        ],
                        justify='center',
                    ),
                    width=12
                ),
                style={'margin-bottom': '20px', 'display': 'flex', 'justify-content': 'center'}
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id='consumption-line-chart'),
                    width=9,
                    style={
                        # 'border': '1px solid #ccc',
                        'box-shadow': '2px 2px 8px rgba(0, 0, 0, 0.1)',
                        'padding': '15px',
                        'margin-bottom': '10px',
                        'backgroundColor': '#2c3e50'
                    }
                ),
                justify='center'
            ),
            dbc.Row(
                dbc.Col(
                    id='graph-info',
                    className='text-center',
                    style={
                        'fontSize': '20px',
                        'padding': '15px',
                        'margin-bottom': '20px',
                        'color': '#ecf0f1'
                    }
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id='histogram-weekend-weekday'),
                        width=6,
                        style={
                            # 'border': '1px solid #ccc',
                            'box-shadow': '2px 2px 8px rgba(0, 0, 0, 0.1)',
                            'padding': '5px',
                            'margin-top': '10px',
                            'margin-right':'50px',
                            'height': '480px',
                            'width': '600px',
                            'backgroundColor': '#2c3e50'
                        }
                    ),
                    dbc.Col(
                        dcc.Graph(id='histogram-weeks'),
                        width=6,
                        style={
                            # 'border': '1px solid #ccc',
                            'box-shadow': '2px 2px 8px rgba(0, 0, 0, 0.1)',
                            'padding': '5px',
                            'margin-top': '10px',
                            'height': '480px',
                            'width': '600px',
                            'backgroundColor': '#2c3e50'
                        }
                    ),
                ],
                justify='center',
                style={'margin-bottom': '20px'}
            ),
            dbc.Row(
                dbc.Col(
                    id='percentage-difference',
                    className='text-center',
                    style={
                        'fontSize': '24px',
                        'marginTop': '20px',
                        'border': '1px solid #ccc',
                        'box-shadow': '2px 2px 8px rgba(0, 0, 0, 0.1)',
                        'padding': '15px',
                        'color': '#ecf0f1',
                        'backgroundColor': '#34495e'
                    }
                ),
            )
        ]
    )


app.layout = create_layout()

# Callbacks
@app.callback(
    [Output('consumption-line-chart', 'figure'),
     Output('graph-info', 'children'),
     Output('percentage-difference', 'children'),
     Output('histogram-weekend-weekday', 'figure'),
     Output('histogram-weeks', 'figure')],
    [Input('week-selector', 'value'),
     Input('algorithm-selector', 'value')]
)
def update_line_chart(selected_week, selected_algorithm):
    if not selected_week:
        raise dash.exceptions.PreventUpdate

    selected_week = pd.to_datetime(selected_week)

    week_data = df[df['Week'] == selected_week]

    if week_data.empty:
        return {}, 'No data available for the selected week.', '', {}, {}

    weekly_data = week_data.groupby('Day').agg({'Electricity_Demand_MW': 'sum'}).reset_index()

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_data['Day'],
        y=weekly_data['Electricity_Demand_MW'],
        mode='lines+markers',
        marker=dict(size=8, color='#3498db'),  
        line=dict(color='#3498db', width=3),
        name='Electricity Demand',
        hovertemplate='Date: %{x}<br>Total Consumption: %{y:.2f} MW'
    ))

    
    max_value = weekly_data['Electricity_Demand_MW'].max()
    min_value = weekly_data['Electricity_Demand_MW'].min()
    y_min = min_value - (max_value * 0.1)
    y_max = max_value + (max_value * 0.1)
    fig.update_layout(
        title=dict(text="Weekly Electricity Consumption", x=0.5, font=dict(size=20, color='#ecf0f1')),
        xaxis_title="Date",
        yaxis_title="Power Usage (MW)",
        yaxis=dict(range=[y_min, y_max]),
        plot_bgcolor='#34495e',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ecf0f1'),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    
    if selected_algorithm == 'rolling':
        prediction = rolling_average(weekly_data['Electricity_Demand_MW'])
        fig.add_trace(go.Scatter(
            x=weekly_data['Day'],
            y=prediction,
            mode='lines+markers',
            marker=dict(size=8, color='#f39c12'),  # Adding markers
            line=dict(dash='dash', color='#f39c12', width=3),  # Thicker line
            name='Rolling Average',
            hovertemplate='Date: %{x}<br>Rolling Average: %{y:.2f} MW'
        ))
    elif selected_algorithm == 'exponential':
        prediction = exponential_smoothing(weekly_data['Electricity_Demand_MW'])
        fig.add_trace(go.Scatter(
            x=weekly_data['Day'],
            y=prediction,
            mode='lines+markers',
            marker=dict(size=8, color='#27ae60'),  # Adding markers
            line=dict(dash='dash', color='#27ae60', width=3),  # Thicker line
            name='Exponential Smoothing',
            hovertemplate='Date: %{x}<br>Exponential Smoothing: %{y:.2f} MW'
        ))
    elif selected_algorithm == 'arima':
        prediction = arima_prediction(weekly_data['Electricity_Demand_MW'], steps=7)
        future_dates = pd.date_range(start=weekly_data['Day'].iloc[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=prediction,
            mode='lines+markers',
            marker=dict(size=8, color='#e74c3c'),  # Adding markers
            line=dict(dash='dash', color='#e74c3c', width=3),  # Thicker line
            name='Future Prediction - ARIMA',
            hovertemplate='Date: %{x}<br>ARIMA Prediction: %{y:.2f} MW'
        ))

    # weekly consump.
    total_consumption = weekly_data['Electricity_Demand_MW'].sum()
    total_days = weekly_data['Day'].nunique()
    average_daily_consumption = total_consumption / total_days if total_days > 0 else 0
    graph_info = f'Total Weekly Consumption: {total_consumption:.2f} MW. Average Daily Consumption: {average_daily_consumption:.2f} MW'

    #weekend vs weekday stat
    weekend_data = df[df['Day'].dt.dayofweek >= 5]
    weekday_data = df[df['Day'].dt.dayofweek < 5]

    weekend_avg = weekend_data['Electricity_Demand_MW'].mean()
    weekday_avg = weekday_data['Electricity_Demand_MW'].mean()

    weekend_weekday_diff = abs(((weekend_avg - weekday_avg) / weekday_avg)) * 100 if weekday_avg != 0 else float('inf')

    
    hist_fig_weekend_weekday = go.Figure(data=[go.Bar(
        x=['Weekdays', 'Weekends'],
        y=[weekday_avg, weekend_avg],
        text=[f"{weekday_avg:.2f}", f"{weekend_avg:.2f}"],
        textposition='auto',
        marker_color=['#0bb4ff', '#0bb4ff']
    )])
    hist_fig_weekend_weekday.update_layout(
        title=dict(text="Average Power Usage - Weekdays vs Weekends", x=0.5, font=dict(color='#ecf0f1')),
        yaxis_title="Power Usage (MW)",
        plot_bgcolor='#34495e',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ecf0f1')
    )

    # summer vs winter (jan vs feb in dataset)
    first_two_weeks_data = df[df['Week'].isin(df['Week'].unique()[:2])]
    last_two_weeks_data = df[df['Week'].isin(df['Week'].unique()[-2:])]

    first_two_weeks_avg = first_two_weeks_data['Electricity_Demand_MW'].mean()
    last_two_weeks_avg = last_two_weeks_data['Electricity_Demand_MW'].mean()

    summer_difference = abs(((last_two_weeks_avg - first_two_weeks_avg) / first_two_weeks_avg)) * 100 if first_two_weeks_avg != 0 else float('inf')
    
    
    hist_fig_weeks = go.Figure(data=[go.Bar(
        x=['Hotter Climate', 'Colder Climate'],
        y=[first_two_weeks_avg, last_two_weeks_avg],
        text=[f"{first_two_weeks_avg:.2f}", f"{last_two_weeks_avg:.2f}"],
        textposition='auto',
        marker_color=['#0bb4ff', '#0bb4ff']
    )])
    hist_fig_weeks.update_layout(
        title=dict(text="Average Power Usage - Hotter vs Colder Climate", x=0.5, font=dict(color='#ecf0f1')),
        yaxis_title="Power Usage (MW)",
        plot_bgcolor='#34495e',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ecf0f1')
    )

    percentage_diff = f'Average Power Usage Increased by {summer_difference:.2f}% during the Summer and {weekend_weekday_diff:.2f}% during the weekends.' if summer_difference != float('inf') else 'Data unavailable for percentage calculation.'

    return fig, graph_info, percentage_diff, hist_fig_weekend_weekday, hist_fig_weeks


if __name__ == '__main__':
    app.run_server(debug=True)
