import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.express as px
import os #handling paths and files in python


from scripts.read_data import read_firebase, read_schools, merge_data
from scripts.clustering_algorithms import filter_school, kmeans, hdbscan_algorithm, evaluation_metrics, agglomerative, spectral

my_path = os.path.abspath(os.path.dirname(__file__)) #capturing the current path of this file. 

df_parents = read_firebase(my_path)
df_schools = read_schools(os.path.join('data', 'schools_data.csv'), my_path)
df = merge_data(df_parents, df_schools)

token = open(".mapbox_token").read() #This file must not go in the commit in git.
px.set_mapbox_access_token(token)

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      'https://use.fontawesome.com/releases/v5.12.0/css/all.css'],
                # these meta_tags ensure content is scaled correctly on different devices
                # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

server = app.server

#Radio items
inline_radioitems = dbc.FormGroup(
    [
        dbc.Label("Choose an algorithm to clusterize children, the number of clusters, and then select a point in the map"),
        dbc.RadioItems(
            options=[
                {"label": "K-means", "value": 'kmeans'},
                {"label": "HDBSCAN", "value": 'hdbscan'},
                {"label": "Agglomerative", "value": 'agglomerativeclustering'},
                {"label": "Spectral", "value": 'spectralclustering'}
            ],
            value='kmeans',
            id="selected-algorithm",
            inline=True,
        ),
        dbc.Row([dbc.Input(placeholder="Number of clusters", type="number",
                  step=1, min=1, max=5, id='number-of-clusters', style=dict(marginTop='5px', width='200px')),
                  dbc.Button('Clear clustering', id='clear-clustering', outline=True, color="secondary",
                  className="mr-1", style=dict(marginLeft='50px'), disabled=True)],
                  )
    ],
    style=dict(marginBottom='20px')
)

#Layout
app.layout = html.Div([dcc.Store(id='selected-school'),
                       html.H1("Safer Walks to Schools Application"),
                       inline_radioitems,
                       dcc.Graph(id="map-graph", style=dict(width="750px", height="350px"))], 
                       style=dict(margin='30px'))

@app.callback(
    Output("map-graph", "figure"), 
    [Input('selected-school', 'modified_timestamp')],
    [State('selected-school', 'data'),
     State('selected-algorithm', 'value'),
     State('number-of-clusters', 'value')]
)
def map_update(ts, selected, algorithm, clusters): 
    if not selected:
        fig = px.scatter_mapbox(df, lon='school_long', lat='school_lat', #color='schoolId',
                                hover_data=['school_name'], custom_data=['schoolId'])
        fig.update_traces(marker_symbol='college')                            
        fig.add_traces(px.scatter_mapbox(df, lon='user_long', lat='user_lat', color='school_name',
                                         hover_data=['parentName', 'timeRegistration'], 
                                         custom_data=['schoolId']).data)
        fig.update_traces(marker_size=12, marker_opacity=0.8)
        fig.update_layout(clickmode='select+event', showlegend=False)    
    else:
        dff = filter_school(df, selected)
        name_school = df.loc[df['schoolId']==selected, 'school_name'].unique()[0]
        if algorithm=='kmeans':
            y = kmeans(dff[['user_long', 'user_lat']], clusters)
        elif algorithm=='hdbscan':
            y = hdbscan_algorithm(dff[['user_long', 'user_lat']], clusters)
        elif algorithm=='agglomerativeclustering':
            y = agglomerative(dff[['user_long', 'user_lat']], clusters)  
        elif algorithm=='spectralclustering':
            y = spectral(dff[['user_long', 'user_lat']], clusters) 

        dff['cluster'] = y.astype(str) #Creating a new column with the labels for plotting by category.
        fig = px.scatter_mapbox(dff.head(1), lon='school_long', lat='school_lat', #color='schoolId',
                                hover_data=['school_name'], custom_data=['schoolId'])
        fig.update_traces(marker_symbol='college') 
    
        fig.add_traces(px.scatter_mapbox(dff, lon='user_long', lat='user_lat', color='cluster',
                                    hover_data=['parentName'], custom_data=['schoolId'],
                                    title='Recommendation for school ' + name_school + ' children walking groups').data)
        fig.update_traces(marker_size=12, marker_opacity=0.8)

    fig.update_layout(
        mapbox = {
            'style': "outdoors"},
        legend=dict(font=dict(size=10), title='Groups recommended:'), 
        margin=dict(l=0, r=0, b=0, t=0)
        )
    return fig

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

@app.callback(
    Output("map-graph", "selectedData"), 
    [Input("clear-clustering", "n_clicks")]
)
def clear_selected(n):
    return None


#For initializing the application
if __name__ == "__main__":
    app.run_server(debug=True)




