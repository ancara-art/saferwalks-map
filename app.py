import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import os #handling paths and files in python
import plotly.io as pio
import pandas as pd

from scripts.read_data import read_firebase, read_schools, merge_data
from scripts.clustering_algorithms import filter_school, kmeans, agglomerative, spectral, evaluation_metrics

my_path = os.path.abspath(os.path.dirname(__file__)) #capturing the current path of this file.
pio.templates.default = "plotly_white" #Theme we used for bar-graph. 

df_parents = read_firebase() # read data from firebase real time database of parents location. 
df_schools = read_schools(os.path.join('data', 'schools_data.csv'), my_path) 
df = merge_data(df_parents, df_schools)

#Token for getting access to mapbox services.
token = open(".mapbox_token").read() #.mapbox_token file must not go in the commit in git.
px.set_mapbox_access_token(token)

#Initiliazing the app, with some additionals parameters: style-sheets from bootstrap for the layout,
#and fontawesome for icons. (such as the icon of Clear-clustering button).
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      'https://use.fontawesome.com/releases/v5.12.0/css/all.css'],
                # these meta_tags ensure content is scaled correctly on different devices
                # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

#This is default for dash setting the server. 
server = app.server

###Layout creation:

#Radio items
inline_radioitems = dbc.FormGroup(
    [
        dbc.Label("1. Select an algorithm to find children groups:"),
        dbc.RadioItems(
            options=[
                {"label": "K-means", "value": 'kmeans_clustering'},
                {"label": "Agglomerative", "value": 'agglomerative_clustering'},
                {"label": "Spectral", "value": 'spectral_clustering'}
            ],
            value='kmeans_clustering',
            id="selected-algorithm",
            inline=True,         
        ),
        dbc.Row([
            dbc.Col(dbc.Row([dbc.Col(dbc.Label("2. Select the number of desired groups:"), width='auto'), 
                    dbc.Col(dbc.Input(placeholder="#", type="number", step=1, min=1, max=5, 
                    id='number-of-clusters', style=dict(marginLeft='-18px', width='75px')), width='auto')]), width='auto'),
            dbc.Col(dbc.Button([html.I(className='fa fa-redo-alt'), ' Clear clustering'], 
                               id='clear-clustering', outline=True, color="secondary", 
                               className="mr-1", disabled=True), width='auto')
                ],
                style=dict(marginTop='10px'),
                justify='between'
                ),
        html.Div([dbc.Label('3. Select a point in the map:'), 
                  html.I(className='fa fa-graduation-cap', style=dict(color='#9E8C26',  marginLeft='10px', marginRight='5px')),
                  dbc.Label('Schools'),
                  html.I(className='fa fa-circle', style=dict(color='gray',  marginLeft='10px', marginRight='5px')),
                  dbc.Label('Parents')
                  ]),
    ],
    style=dict(marginBottom='10px')
)

#Layout
app.layout = html.Div([dcc.Store(id='selected-school'),
                       html.H1("Safer Walks to Schools Application"),
                       inline_radioitems,
                       dbc.Row(
                           [dbc.Col(dcc.Graph(id="map-graph", style=dict(height="350px")),
                           width=7),
                           dbc.Col(dcc.Graph(id='bar-graph', style=dict(height='350px')), width=5)])], 
                       style=dict(margin='30px'))


@app.callback(
    [Output('map-graph', 'figure'),
    Output('bar-graph', 'figure')], 
    [Input('selected-school', 'modified_timestamp')],
    [State('selected-school', 'data'),
     State('selected-algorithm', 'value'),
     State('number-of-clusters', 'value'),
     State('map-graph', 'figure')]
)
def map_update(ts, selected, algorithm, clusters, position): 
    if not selected:
        fig = px.scatter_mapbox(df, lon='school_long', lat='school_lat', #color='schoolId',
                                hover_data=['school_name'], custom_data=['schoolId'])
        fig.update_traces(marker_symbol='college')                            
        fig.add_traces(px.scatter_mapbox(df, lon='user_long', lat='user_lat', color='school_name',
                                         hover_data=['parentName', 'timeRegistration'], 
                                         custom_data=['schoolId']).data)
        fig.update_traces(marker_size=12, marker_opacity=0.8)
        fig.update_layout(clickmode='select+event', showlegend=False) 
        #Initializing graph bar in 0.
        df_empty = pd.DataFrame({'Algorithm': ['K-means', 'Agglomerative', 'Spectral'], 
                                 'Value': [0, 0, 0]})
        bar_fig = px.bar(df_empty, x='Algorithm', y='Value', color='Algorithm', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)  
    else:
        dff = filter_school(df, selected)
        name_school = df.loc[df['schoolId']==selected, 'school_name'].unique()[0]

        #Cluster algorithms
        y_kmeans = kmeans(dff[['user_long', 'user_lat']], clusters)
        y_agglomerative = agglomerative(dff[['user_long', 'user_lat']], clusters)  
        y_spectral = spectral(dff[['user_long', 'user_lat']], clusters) 

        #Store the labels in a new column for plotting by category depending on the 
        #radio button(algorithm) selected. 
        if algorithm=='kmeans_clustering':
            dff['cluster'] = y_kmeans.astype(str) 
        elif algorithm=='agglomerative_clustering':
            dff['cluster'] = y_agglomerative.astype(str) 
        elif algorithm=='spectral_clustering':
            dff['cluster'] = y_spectral.astype(str) 
            
        #Plot the map according to the cluster algorithm choosen.
        fig = px.scatter_mapbox(dff.head(1), lon='school_long', lat='school_lat', #color='schoolId',
                                hover_data=['school_name'], custom_data=['schoolId'],
                                title='Recommendation for school ' + name_school)
        fig.update_traces(marker_symbol='college') 
    
        fig.add_traces(px.scatter_mapbox(dff, lon='user_long', lat='user_lat', color='cluster',
                                    hover_data=['parentName'], custom_data=['schoolId']).data)
        fig.update_traces(marker_size=12, marker_opacity=0.8)

        #Evaluate the metrics:
        df_accuracy = evaluation_metrics(dff[['user_long', 'user_lat']], y_kmeans, 'euclidean', 'K-means')
        df_accuracy = df_accuracy.append(evaluation_metrics(dff[['user_long', 'user_lat']], y_agglomerative, 'euclidean', 'Agglomerative'))
        df_accuracy = df_accuracy.append(evaluation_metrics(dff[['user_long', 'user_lat']], y_spectral, 'euclidean', 'Spectral'))
        #Bar plot
        bar_fig = px.bar(df_accuracy, x='Algorithm', y='Value', color='Algorithm', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)

    #Layout bar graph
    bar_fig.update_layout( 
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,
        yaxis=dict(title='Silhouette Score'),
        xaxis=dict(title='Clustering algorithm')
    )

    #Layout map graph
    fig.update_layout(
        mapbox = {
            'style': "outdoors", 'zoom': 8},
        legend=dict(font=dict(size=10), orientation='h', title='Recommended groups:'), 
        margin=dict(l=0, r=0, b=0, t=30)
    )

    #For keeping the zoom and position when selecting a point and when clearing clustering.
    if position:
        zoom = position['layout']['mapbox']['zoom']
        center = position['layout']['mapbox']['center']
        fig.layout['mapbox']['zoom'] = zoom
        fig.layout['mapbox']['center'] = center

    return fig, bar_fig

#Call back where we are storing in the hidden div, the school Id of the point selected
#In addition, we are enabling and disabling the clear button in this call back.
@app.callback(
    [Output('selected-school', 'data'), 
    Output('clear-clustering', 'disabled')], 
    [Input('map-graph', 'selectedData')]
)
def update_school(selected):
    if not selected:
        selected = None
        disabled = True
    else:
        selected = selected['points'][0]['customdata'][0]
        disabled = False
    return selected, disabled

#Call back 'Clear Clustering' button.
@app.callback(
    Output("map-graph", "selectedData"), 
    [Input("clear-clustering", "n_clicks")]
)
def clear_selected(n):
    return None

#For initializing the application
if __name__ == "__main__":
    app.run_server(debug=True)