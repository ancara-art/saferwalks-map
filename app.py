import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import json
from firebase import firebase
import os #handling paths and files in python

mypath = os.path.abspath(os.path.dirname(__file__)) #capturing the current path of this file. 

#Tutorial we followed for reading data in firebase from python: https://morioh.com/p/4dca3ded4cea
firebase = firebase.FirebaseApplication(r'https://safer-walks-default-rtdb.firebaseio.com/', None)
parents_data = firebase.get(r'/RegisteredParents', '')

#reading schoolsdata file which contains all schools in Sweden.
schools_file = 'schoolsdata.csv'
df_schools = pd.read_csv(schools_file)

#Creating a dataframe from parents_data. 
df_parents = pd.DataFrame()
for i, parent in enumerate(parents_data.values()):
    df_parents = df_parents.append(pd.DataFrame(parent, index=[i]), ignore_index=True)

#Splitting userLocation from df_parents, into latitude: user_lat and longitude: user_long.
df_parents['user_lat'] = [float(s.split(',')[0]) for s in df_parents['userLocation']]
df_parents['user_long'] = [float(s.split(',')[1]) for s in df_parents['userLocation']]

#Merging df_schools into df_parents by schoolId(df_parents) and osm_id(df_schools)
df = pd.merge(df_parents, df_schools, left_on='schoolId', right_on='osm_id')

token = open(".mapbox_token").read() #This file must not go in the commit in git.
px.set_mapbox_access_token(token)
df["schoolId"] = df["schoolId"].astype(str)


app = dash.Dash(__name__)
server = app.server

dropdown = dcc.Dropdown(id="schools-parents-dropdown", 
                        options=[{"label": "Schools", "value": "schools"},
                                 {"label": "Parents", "value": "parents"}],
                        value="schools",
                        style=dict(width="1000px"))

#Layout
app.layout = html.Div([html.H1("Safer Walks to Schools Application"),
                       dropdown,
                       dcc.Graph(id="map-graph", style=dict(width="1000px", height="600px"))])

@app.callback(
    Output("map-graph", "figure"), 
    [Input("schools-parents-dropdown", "value")]
)
def map_update(value):
    if value == "schools":
        fig = px.scatter_mapbox(df, lon='school_long', lat='school_lat', #color='osm_id',
                                hover_data=['school_name'])
    else:
        fig = px.scatter_mapbox(df, lon='user_long', lat='user_lat', color='school_name',
                                hover_data=['parentName', 'timeRegistration'])
    fig.update_layout(
        mapbox = {
            'style': "satellite"},
        showlegend = True)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)




