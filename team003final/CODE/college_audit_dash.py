#
# - Structure
# [1] Load the packages [2] Load the needed dataframe [3] Code for filtering [4] Code for operation [5] Code for app [6] Code for callbacks

# - for [4] you can add any code for operating. Just please leave the comments about the contents,your name, and numbering!
# - for [5] if you search your name, you will find the place for your code!
# - for [6] because our dash is an interactive one, we use callback functions.
#           it will be edited after everyone finish their code.

## key Variable (2023/11/09 current)
# ranked_list : df from the final rank file
# sample_filtered_list : filtering result but not using filtering process/ just sample for UI development
# most_recent_cohorts_FoS_final : df that has all majors(FoS) for each school; default
# historical_data: df that has AVG adm_rate, net_price, debt for each school, annually (2014-2022); default

## key CSS rule
# 'fontFamily': 'Noto Serif, serif', 'fontWeight': choose 300/400/600



# [1] Load the packages
import dash
from dash import Dash, dcc, html,  Input, Output, dash_table, callback, State
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go


external_stylesheets = ["https://fonts.googleapis.com/css2?family=Noto+Serif:wght@300;400;600&family=Oswald:wght@300&display=swap"]

# [2] Load the needed dataframe

## 1. Load the final rank file ('ranked_list_final.csv')
ranked_list = pd.read_csv('CODE/ranked_list_final.csv')

## 2. Load the most recent cohort fos file
most_recent_cohorts_FoS_final = pd.read_csv('CODE/most_recent_cohorts_fos_final.csv')

## 3. Load the historical data
historical_data = pd.read_csv('CODE/historical_final.csv')

## 4. Load the location data
location_data = pd.read_csv('CODE/location_info.csv')

# [3] Code for filtering - Christina
# You can add codes for filtering process here
#get a dictionary with the FOS and the corresponding CIPCODEs
cipCodes = most_recent_cohorts_FoS_final['CIPCODE'].unique()
cipDescs = []
cipuniques = []
for c in cipCodes:
    cipDescs.append(most_recent_cohorts_FoS_final[most_recent_cohorts_FoS_final['CIPCODE'] == c]['CIPDESC'].unique()[0])
    cipuniques.append([c, cipDescs[-1].replace(".", "")])
cipOptions = []
for u in cipuniques:
    val = u[1]
    if len(val) > 30:
        val = val[:27] + "..."
    else: val = val[:30]
    cipOptions.append({'label': val, 'value': u[0], 'title': u[1]})

#get dictionary with the states and the corresponding UNITIDs
stabbrOptions = list(set(ranked_list.STABBR))
stabbrOptions.sort()

#get list of size bucket values
sizeBins = pd.cut(ranked_list.Size, bins=3, precision=-2)
first_I = sizeBins.cat.categories[0]
new_I = pd.Interval(0, first_I.right)
sizeBins = sizeBins.cat.rename_categories({first_I: new_I})
sizeList = []
[sizeList.append([int(sizeBins.cat.categories[x].left), int(sizeBins.cat.categories[x].right)]) for x in range(3)]


# [4] Code for operating
# (Please leave the comment about the content and who wrote it.)
# (Nicolas & Satish ; Because the code for filtering process isn't finished,
#  I made the 'sample_result' variable which contains the information of 10 universities.
#  You can use it to develop 'detail information' part and 'historical trends' part.)

## 1. Sample filtering result just for UI development - from Clara


## 2. Code for presenting top 10 universitiy - from Clara

    # Left table


    # Right table


right_table = dash_table.DataTable(id='table-right',columns=[{"name": "Rank", "id": "Rank"}, {"name": "University", "id": "INSTNM"}],
     style_cell={'textAlign': 'center','height': '25px', 'width':'50px','fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
        'overflow': 'hidden', 'textOverflow': 'ellipsis','whiteSpace': 'normal'},
    style_cell_conditional=[{'if': {'column_id': 'INSTNM'}, 'minWidth': '650px', 'width': '650px', 'maxWidth': '650px', 'textAlign':'left'}],
    style_header={'fontWeight': 'bold', 'textAlign':'center', 'display': 'hide'})



#
#

## 6. code for 'empty figure' return when any university wasn't selected.
empty_figure = go.Figure()

# [5] Code for app
# 1. Initialize the Dash app - from Clara
app = Dash(__name__, external_stylesheets=external_stylesheets)

# 2. Drawing the layout for app - from Clara
# (Please check the comment to find the place of each component)

app.layout = html.Div(
    style={'width': '1500px', 'height': '1500px'},  # Set the size of the outermost container
    children=[

        # Row container

        html.Div(

            className='row',
            style={'height': '1500px', 'display': 'flex', 'flexDirection': 'row'},
            children=[

                # Left column for Filtering Options
                html.Div(

                    className='div-user-controls',
                    style={'width': '250px', 'height': '1500px', 'backgroundColor': '#D0F1B2'},
                    children=[


                        #name
                        html.H2("College Audit",
                                style = {'fontFamily': 'Noto Serif, serif', 'fontSize': '50px', 'fontWeight': '600',
                                         'color': '#3F6121', 'textAlign': 'center'}),
                        #Filtering options

                        html.P("Filtering Options",
                               style = {'fontFamily': 'Noto Serif, serif', 'fontSize': '28px','fontWeight' : '400',
                                        'color' :'#1C2F0A', 'textAlign': 'center'}),

                        html.H2("Please enter the filtering options and click submit button below",
                                style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                       'color': '#3F6121', 'fontWeight' : '300','marginLeft': '10px'}),
                        html.P("Please make sure you enter the required options", id='alert-message', style={'display':'none'}),


#### Filtering Option UI (Jacob) - Start
                        # (I added some code for sample text, dropbox, slider just for reference. You can adjust style freely.)

                        dcc.Store(id="top-ten-dict"),
                        #submit button?
                        html.Button("Submit", id="submit-button", n_clicks=0, style={'width':'200px', 'height':'40px',
                        'fontFamily': 'Noto Serif, serif', 'fontSize': '20px','fontWeight': '600', 'marginLeft': '20px',
                        'backgroundColor':'#1C2F0A','color':'white'}),


                        #First option Academic Fields
                        html.P("Academic Fields [Required]",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px', 'color': 'black',
                                       'fontWeight' : '300','marginLeft': '10px'}, title='Enter the name of academic fields.'
                               ),

                        #academic fields drop down
                        dcc.Dropdown(id='academic-fields', # style= {'marginLeft': '2px', 'width': '230px', 'overflow-y': 'auto','max-height': '280px' },
                                     options=cipOptions,value=[1101], multi=True),


                        #Second option Location
                        html.P(f"Location (State) [Required]",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},title='Enter the name of States'),

                        #Location drop down
                        dcc.Dropdown(id='location-state',
                                     # style = {'marginLeft': '2px','width': '240px'},
                                     options=stabbrOptions,value=["GA"], multi=True),

                        #Test score
                        html.P("Test Scores [Required]",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},
                               title='Select test type and move slider to enter your test score and display schools you qualify'),
                        #selector for ACT or SAT score
                        dcc.RadioItems(id='test-score-selector',
                                       options=[
                                           {'label': 'ACT', 'value': True},
                                           {'label': 'SAT', 'value': False},
                                       ],
                                       value=True,
                                       inline=True,
                                       style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'}
                                       ),

                        #Slider for ACT score
                        dcc.Slider(id='ACT-score', min=1, max=36, disabled=True, value=30, step=1, marks={x: str(x) for x in range(1,37, 6)},  tooltip={"placement": "top", "always_visible": False}),

                        #Slider for SAT score
                        dcc.Slider(id='SAT-score', min=400, max=1600, disabled=True,value=None, tooltip={"placement": "top", "always_visible": False}),

                        # Family Income
                        html.Div(
                        html.P("Family Income",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'}),
                            title="Move slider to enter your family income and display schools you qualify"),

                        #Slider for Family income
                        dcc.Slider(id='family-income', min=math.floor((ranked_list.AVG_FAMINC.min()/1000))*1000, max=math.ceil((ranked_list.AVG_FAMINC.max()/1000))*1000, value=None, tooltip={"placement": "top", "always_visible": False}),

                        # Average net price
                        html.P("Average Net Price",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},
                               title='Move slider to enter expected average net price. Average net price includes tuition and fees, books and supplies, and living expenses, minus the average grant/scholarship aid.'),
                        #
                        #Slider for Average net price
                        dcc.Slider(id='avg-net-price', min=math.floor((ranked_list.NetPrice.min()/1000))*1000, max=math.ceil((ranked_list.NetPrice.max()/1000))*1000, value=None, tooltip={"placement": "top", "always_visible": False}),

                        # Debt
                        html.P("Debt",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},
                               title='Move slider to enter expected median debt'),
                        #
                        #Slider for Debt
                        dcc.Slider(id='debt', min=math.floor((ranked_list.Debt.min()/1000))*1000, max=math.ceil((ranked_list.Debt.max()/1000))*1000, value=None, tooltip={"placement": "top", "always_visible": False}),

                        # Acceptance Rate
                        html.P("Acceptance Rate",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},
                               title='Move slider to enter expected acceptance rates'),
                        #
                        #Slider for Acceptance Rate
                        dcc.Slider(id='acceptance-rate', min=math.floor((ranked_list.AdmRate.min()/100))*100, max=math.ceil((ranked_list.AdmRate.max()/100))*100, value=100, marks={x: str(x)+"%" for x in range(0,101, 20)}, tooltip={"placement": "top", "always_visible": False}),

                        # Type of School
                        html.P("Type of School",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight' : '300','marginLeft': '10px'},
                               title='Select type of school'),
                        #
                        #Checklist for type of school
                        dcc.Checklist([x for x in ranked_list.CONTROL.unique().tolist() if str(x) != 'nan'], value=[ranked_list.CONTROL.unique().tolist()[0]] ,id='type-of-school', inline=False,
                                      style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                             'color': 'black', 'fontWeight': '300', 'marginLeft': '10px'}
                                      ),

                        # Size (change to 0-5000, 5000-15000, 15000+)
                        html.P("Size",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight': '300', 'marginLeft': '10px'},
                               title='Select the size of university. Small: 0-4,999 Students / Medium : 5,000-14,999 Students / Large : 15,000+ Students'),
                        #
                        # Checklist for size
                        dcc.Checklist(options = [{"label" : "Small", 'value':"0-4999", "title": "0-4,999 Students"},
                                                 {"label" : "Medium", 'value':"5000-14999", "title": "5,000-14,999 Students"},
                                                 {"label" : "Large", 'value':"15000-100000", "title": "15,000+ Students"}],
                                      id='size-of-school',value=["0-4999", "5000-14999"], inline=False,
                                      style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                             'color': 'black', 'fontWeight': '300', 'marginLeft': '10px'}
                                      ),

                        # other filters
                        html.P("Other Filters",
                               style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                      'color': 'black', 'fontWeight': '300', 'marginLeft': '10px'},title='Select other special options'),

                        # Checklist for other filters
                        dcc.Checklist(["Women Only", "Men Only"],
                                      id='sex-constraint',value=[None], inline=False,
                                      style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                                             'color': 'black', 'fontWeight': '300', 'marginLeft': '10px'}
                                      )
         ]
        ),

                        # Please add more options here


#### Filtering Option UI (Jacob) - End

                # Right column for displaying content
                html.Div(
                    className='div-for-charts bg-grey',
                    style={'width': '1250px'},
                    children=[

#### Result List (Clara) - Start
                        html.Div(className="List", style={'height': '50px','backgroundColor':'#B6E08F', 'margin':'0'},
                                     children = [
                             html.H2("Top 10 Universities for you",
                                     style={'fontFamily': 'Noto Serif, serif', 'fontSize': '30px',
                                            'fontWeight': '600',
                                            'color': '#1C2F0A', 'marginLeft': '20px', 'marginTop': '0px',
                                            'marginBottom': '10px'})
                                 ]),

                        html.Div(className="List contents", style={'height' : '400px', 'margin':'0'},
                        children =[
                        html.H2("The best top 10 universities we recommend considering your filtering options.",
                                style={'fontFamily': 'Noto Serif, serif', 'fontSize': '15px','fontWeight': '400',
                                       'marginLeft': '5px', 'marginTop': '1px'}),
                        html.Div(
                                className='list_row',
                                style={'height':'350px','display': 'flex', 'flexDirection': 'row', 'marginLeft':'20px'},
                                children=[right_table])
                        ]),
#### Result List (Clara) - end


                        html.Div(
                            className='middle_row',
                            style={'height': '700px', 'display': 'flex', 'flexDirection': 'row'},
                            children=[


#### Detail information (Satish) - Start
                        html.Div(className="Detail", style={'height': '400px', 'width':'50%'},
                            children = [
                                # div for color box
                                html.Div(id='detail-information', style={'height': '50px','backgroundColor':'#B6E08F'},
                                    children=[
                                        # text for title

                                        html.H3("Detail information of top 10",
                                                style={'fontFamily': 'Noto Serif, serif', 'fontSize': '20px',
                                                'fontWeight': '600', 'color': '#1C2F0A', 'marginLeft': '10px', 'marginTop': '0px','marginBottom': '0px'}),

                                         ]),
                                            html.H3(['Our system used the following variables to rank.', html.Br(),'The charts below show they ranked in each metric.',html.Br(),'Click on the node to see the exact value.'],
                                                    id='detail_info_text',
                                         style={'fontFamily': 'Noto Serif, serif','fontSize': '15px','fontWeight': '400','marginLeft': '5px', 'marginTop': '0px', 'display':'none'}),

                                dcc.Graph(id='detail_info_graph', style={'height': '570px', 'display':'none'})
                                ### Please add graphs here

                            ]),

#### Detail information (Satish) - end


#### Historical Trends (Nicolas) - start
                                html.Div(id='historical-trends', style={'height': '700px', 'width': '50%'},
                                         children=[
                                             # Div for color box
                                             html.Div(id='Trends-text',
                                                      style={'height': '50px', 'backgroundColor': '#B6E08F'},
                                                      children=[
                                                          # text for title
                                                          html.H3("Historical Trends (2015 ~ 2022)",
                                                                  style={'fontFamily': 'Noto Serif, serif',
                                                                         'fontSize': '20px',
                                                                         'fontWeight': '600', 'color': '#1C2F0A',
                                                                         'marginLeft': '10px', 'marginTop': '0px',
                                                                         'marginBottom': '0px'})]),

                                             # Div for the graph view box
                                             html.Div(id='Trends-graph', style={'height': '650px'},
                                                      children=[
                                                        #   html.H2('Click on a university to see the historical trend.\n To check the percentage change, click the nodes.',
                                                        #           style={'fontFamily': 'Noto Serif, serif',
                                                        #                  'fontSize': '15px', 'fontWeight': '400',
                                                        #                  'marginLeft': '5px', 'marginTop': '1px'}),
                                                          # Please add graphs here
                                                        #   html.H2("",
                                                        #           style={'fontFamily': 'Noto Serif, serif', 'fontSize': '20px','color': '#3F6121',
                                                        #           'fontWeight' : '600','textAlign':'center'}),
                                                        #   dcc.Graph(id='hist-trend-plots',
                                                        #             figure=empty_figure,
                                                        #             style={'height': '570px'})

                                                      ])

                                         ])

                            ]),

#### Historical Trends (Nicolas) - end

#### Location (Clara) - start
                    html.Div(id='location', style={'height': '350px'},
                        children=[
                        # Div for color box
                        html.Div(id='location_box', style={'height': '50px', 'backgroundColor': '#B6E08F'},
                            children=[
                                html.H3("Location", style={'fontFamily': 'Noto Serif, serif', 'fontSize': '20px',
                                'fontWeight': '600', 'color': '#1C2F0A', 'marginLeft': '10px', 'marginTop': '0px','marginBottom': '0px'})]),
                        html.Div(className='location_row',style={'height': '350px', 'display': 'flex', 'flexDirection': 'row'},
                                    children=[
                        html.H3(["This map shows the locations of universities.", html.Br(), "You can zoom the map, and hover your mouse over the nodes to see the school's ranking and name."],
                                id='location_map_text',style={'fontFamily': 'Noto Serif, serif','fontSize': '15px','display':'none', 'fontWeight': '400','marginLeft': '5px', 'marginTop': '1px', 'width':'200px'}),
                        dcc.Graph(id = 'location_map',  style={'height':'350px', 'display':'none', 'width':'1050px'})])

                    ]),




                    ]
                ),
            ]
        )
    ]
)

def basic_less_than(ranked_list, input, filter_val):
    if input == 0:
        input = None
    if input is not None:
        list_filt = ranked_list[ranked_list[filter_val] <= input]
    else: list_filt = ranked_list
    return list_filt, input
# [6] Code for callbacks (interactive functions)

## 1. Callback functions for filtering options - from Jacob
@callback([
    Output(component_id='table-right', component_property='data'),
    Output(component_id='top-ten-dict', component_property='data')],
    Input('submit-button', 'n_clicks'),
    [State(component_id='location-state', component_property='value'),
     State(component_id='academic-fields', component_property='value'),
     State(component_id='ACT-score', component_property='value'),
     State(component_id='SAT-score', component_property='value'),
     State(component_id='family-income', component_property='value'),
     State(component_id='avg-net-price', component_property='value'),
     State(component_id='debt', component_property='value'),
     State(component_id='acceptance-rate', component_property='value'),
     State(component_id='type-of-school', component_property='value'),
     State(component_id='size-of-school', component_property='value'),
     State(component_id='sex-constraint', component_property='value')]
)
def update_table(n_clicks, location_state, academic_fields, act_score, sat_score, family_income, avg_net_price, debt, acceptance_rate, type_of_school, size_of_school, sex_constraint):
    #get only undergraduate values
    filtered_list = ranked_list[ranked_list['CREDDESC'] == 'Bachelor\'s Degree']
    #wait until submit button is clicked
    if n_clicks > 0:
        #melt lists in case of nesting
        if len([location_state]) > 1:
            location_state = [item for sublist in location_state for item in sublist]
        if len([academic_fields]) > 1:
            academic_fields = [item for sublist in academic_fields for item in sublist]

        type_of_school.append(None)
        if len([type_of_school]) > 1:
            type_of_school = [item for sublist in type_of_school for item in sublist]

        #handle scores and filter
        if act_score == None:
            filtered_list = filtered_list[filtered_list.SAT_AVG <= sat_score]
        elif sat_score == None:
            filtered_list = filtered_list[filtered_list.ACTCMMID <= act_score]



        #filter by family income if given, set to none if 0
        filtered_list, family_income = basic_less_than(filtered_list, family_income, "AVG_FAMINC")

        #filter by average net price, set to None if 0
        filtered_list, avg_net_price = basic_less_than(filtered_list, avg_net_price, "NetPrice")

        #filter by debt, set to None if 0
        filtered_list, debt = basic_less_than(filtered_list, debt, "Debt")

        #filter by acceptance rate, set to None if 0
        if acceptance_rate == 0:
            acceptance_rate = None
        if acceptance_rate is not None:
            filtered_list = filtered_list[filtered_list["AdmRate"] <= acceptance_rate/100]

        #filter by type of school
        filtered_list = filtered_list[filtered_list["CONTROL"].isin(type_of_school)]

        #filter by size
        if len(size_of_school) == 1:
            size_of_school = size_of_school[0].split("-")
            filtered_list = filtered_list[(filtered_list["Size"] >= int(size_of_school[0])) & (filtered_list["Size"] <= int(size_of_school[-1]))]
        elif len(size_of_school) > 1:
            size_of_school = [x.split("-") for x in size_of_school]
            size_of_school = [item for sublist in size_of_school for item in sublist]
            filtered_list = filtered_list[
            (filtered_list["Size"] >= int(size_of_school[0])) & (filtered_list["Size"] <= int(size_of_school[-1]))]


        #filter by sex constraint
        if len(sex_constraint) > 1:
            if "Women Only" in sex_constraint:
                filtered_list = filtered_list[filtered_list.WOMENONLY == 1]
            elif "Men Only" in sex_constraint:
                filtered_list = filtered_list[filtered_list.MENONLY == 1]


        #get unit id list for schools that have given FOSs
        fosUnitIds = most_recent_cohorts_FoS_final[most_recent_cohorts_FoS_final["CIPCODE"].isin(academic_fields)]["UNITID"].dropna().unique().tolist()
        #filter by state and FOS
        filtered_list = filtered_list[(filtered_list['STABBR'].isin(location_state)) & (filtered_list["UNITID"].isin(fosUnitIds))][:10]
        right_table_data = filtered_list.reset_index()
        right_table_data['Rank'] = right_table_data.index + 1
        right_table_data = right_table_data[['Rank', 'INSTNM', 'UNITID']]
        right_table_data_dicts = right_table_data.to_dict('records')
        return right_table_data_dicts, right_table_data_dicts
    else: return dash.no_update, dash.no_update

#
@callback(
    [Output(component_id='ACT-score', component_property='disabled'),
     Output(component_id='ACT-score', component_property='value')],
    Input(component_id='test-score-selector', component_property='value')
)
def show_hide_element(selector):
    if selector:
        return False, None
    else:
        return True, None
@callback(
    [Output(component_id='SAT-score', component_property='disabled'),
     Output(component_id='SAT-score', component_property='value')],
    Input(component_id='test-score-selector', component_property='value')
)
def show_hide_element(selector):
    if selector:
        return True, None
    else:
        return False, None

## 3. code for presenting the maps - from Clara
@callback(
    Output(component_id='location_map_text', component_property='style'),
    Output(component_id='location_map', component_property='figure'),
    Output(component_id='location_map', component_property='style'),
    Input(component_id='top-ten-dict', component_property='data')
)
def generate_map(unitid_dict):

    filtered_location_df = location_data[location_data['INSTNM'].isin([x['INSTNM'] for x in unitid_dict])].copy()
    Rank_list = list(i+1 for i in range(filtered_location_df.shape[0]))
    filtered_location_df.insert(loc=1, column='Rank', value=Rank_list)

    filtered_location_df['hover_text'] =  filtered_location_df['Rank'].astype(str) + '. '+filtered_location_df['INSTNM']


    university_map = go.Figure(data=go.Scattergeo(
                lon = filtered_location_df['Longitude'],
                lat = filtered_location_df['Latitude'],
                text = filtered_location_df['hover_text'],
        mode='markers'
    ))
    #
    university_map.update_layout(
        geo_scope='usa',
        width=850, height=300,
        margin=dict(l=0, r=0, b=0, t=0, autoexpand=True),
    )


    return {'fontFamily': 'Noto Serif, serif','fontSize': '15px','display':'block', 'fontWeight': '400','marginLeft': '5px', 'marginTop': '1px', 'width':'400px'},\
    university_map, {'height':'350px','display':'block', 'width':'850px'}

# ## 4. code for detailed information - from Satish
@callback(Output(component_id='detail_info_text', component_property='style'),
          Output(component_id='detail_info_graph', component_property='figure'),
          Output(component_id='detail_info_graph', component_property='style'),
          Input(component_id='top-ten-dict', component_property='data'))
def update_detail_graph(unitid_dict):


    df_detail = ranked_list[ranked_list['UNITID'].isin([x['UNITID'] for x in unitid_dict])]
    cols=['GradRate', 'NetPrice', 'AdmRate', 'Salary', 'Debt', 'Size', 'Predicted Score']
    df_detail = df_detail[cols]
    df_detail['Rank'] = df_detail['Predicted Score'].rank(ascending=False).astype(int)
    #
    detail_info_fig = make_subplots(rows=6, cols=1, subplot_titles=("Graduation Rate (%)", "Admission Rate (%)", "Net Price ($)", "Salary ($)", "Debt ($)", "Size"))
    detail_info_mkr_sz = 10

    detail_info_fig.append_trace(go.Scatter(x=df_detail.GradRate*100, y=[0]*10, mode='markers+text',
                                            marker_size=detail_info_mkr_sz, hoverinfo='text', hovertext=round(df_detail.GradRate*100,2), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=1, col=1)

    detail_info_fig.append_trace(go.Scatter(x=df_detail.AdmRate*100, y=[0]*10, mode='markers+text', marker_size=detail_info_mkr_sz, hoverinfo='text',
                                            hovertext=round(df_detail.AdmRate*100,2), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=2, col=1)

    detail_info_fig.append_trace(go.Scatter(x=df_detail.NetPrice, y=[0]*10, mode='markers+text', marker_size=detail_info_mkr_sz, hoverinfo='text',
                                            hovertext=round(df_detail.NetPrice), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=3, col=1)

    detail_info_fig.append_trace(go.Scatter(x=df_detail.Salary, y=[0]*10, mode='markers+text', marker_size=detail_info_mkr_sz, hoverinfo='text',
                                            hovertext=round(df_detail.Salary), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=4, col=1)

    detail_info_fig.append_trace(go.Scatter(x=df_detail.Debt, y=[0]*10, mode='markers+text', marker_size=detail_info_mkr_sz, hoverinfo='text',
                                            hovertext=round(df_detail.Debt), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=5, col=1)

    detail_info_fig.append_trace(go.Scatter(x=df_detail.Size, y=[0]*10, mode='markers+text', marker_size=detail_info_mkr_sz, hoverinfo='text',
                                            hovertext=round(df_detail.Size), text=df_detail.Rank,
                                            textposition='top center'),
                                 row=6, col=1)

    detail_info_fig.update_xaxes(showgrid=False, autorange=True)
    detail_info_fig.update_yaxes(showgrid=False,
                                 zeroline=True, zerolinecolor='black', zerolinewidth=2,
                                 showticklabels=False)
    detail_info_fig.update_layout(plot_bgcolor='beige', font=dict(size=15),
                                  autosize=False, width=625, height=570,
                                  margin=dict(l=0, r=0, t=20, b=20),
                                  showlegend=False
                                  )
    return {'fontFamily': 'Noto Serif, serif', 'fontSize': '15px', 'fontWeight': '400', 'marginLeft': '5px',\
             'marginTop': '0px', 'display': 'block'}, detail_info_fig,{'height': '570px', 'display':'block'}



## 2. Callbacks for historical trends - from Nicolas
## 5. historical trends continued
active_state = 0


# update historical trend plots figure based on user clicking one of the top 10 colleges
@app.callback(
    Output('Trends-graph', 'children'),
    Input('table-right', 'active_cell'),
    Input('top-ten-dict', 'data')
)
def update_figure(chosen_cell, right_table_data):
    
    right_table_data = pd.DataFrame(right_table_data)
    # ## 5. code for historical trends - from Nick
    #
    if not right_table_data.empty: 
        # historical trends for AdmRate, NetPrice, and Debt (instead of expenses)
        # the default graph will be for the top univeristy

        df_top10 = ranked_list[ranked_list['UNITID'].isin(right_table_data['UNITID'])]
        top_univ = df_top10.iloc[0]
        chosen_univ_hist_df = historical_data[historical_data['UNITID'] == top_univ['UNITID']].sort_values(by='YEAR', ascending=True)

        # adding a new set of columns for delta of each year
        chosen_univ_hist_df['delta_AdmRate'] = chosen_univ_hist_df['AdmRate'].pct_change()*100
        chosen_univ_hist_df['delta_AdmRate'] = chosen_univ_hist_df['delta_AdmRate'].fillna(0)

        chosen_univ_hist_df['delta_NetPrice'] = chosen_univ_hist_df['NetPrice'].pct_change()*100
        chosen_univ_hist_df['delta_NetPrice'] = chosen_univ_hist_df['delta_NetPrice'].fillna(0)

        chosen_univ_hist_df['delta_Debt'] = chosen_univ_hist_df['Debt'].pct_change()*100
        chosen_univ_hist_df['delta_Debt'] = chosen_univ_hist_df['delta_Debt'].fillna(0)

        # setting up initial historical trend figure
        hist_initial_figure = make_subplots(rows=3, cols=1, subplot_titles=("Admission Rate", "Net Price", "Debt"))

        hist_initial_figure.append_trace(go.Scatter(x=chosen_univ_hist_df['YEAR'], y=chosen_univ_hist_df['AdmRate']*100, customdata= chosen_univ_hist_df['delta_AdmRate'], mode='markers+lines+text',
                                                hovertemplate =
                                                'Percent Change: %{customdata:.2f}%<extra></extra>',
                                                texttemplate='%{y:.2f}%', textposition='top center', textfont=dict(size=8), cliponaxis=False), row=1, col=1)
        hist_initial_figure.append_trace(go.Scatter(x=chosen_univ_hist_df['YEAR'], y=chosen_univ_hist_df['NetPrice'], customdata= chosen_univ_hist_df['delta_NetPrice'], mode='markers+lines+text',
                                                hovertemplate =
                                                'Percent Change: %{customdata:.2f}%<extra></extra>',
                                                texttemplate='$%{y:.2f}', textposition='top center', textfont=dict(size=8), cliponaxis=False), row=2, col=1)
        hist_initial_figure.append_trace(go.Scatter(x=chosen_univ_hist_df['YEAR'], y=chosen_univ_hist_df['Debt'], customdata= chosen_univ_hist_df['delta_Debt'], mode='markers+lines+text',
                                                hovertemplate =
                                                'Percent Change: %{customdata:.2f}%<extra></extra>',
                                                texttemplate='$%{y:.2f}', textposition='top center', textfont=dict(size=8), cliponaxis=False), row=3, col=1)
        hist_initial_figure.update_annotations(font_size=14, yshift=10)
        hist_initial_figure.update_layout(
            plot_bgcolor='white',
            font=dict(size=10),
            autosize=False, width=625, height=550,
            margin=dict(l=30, r=30, t=30, b=30),
            showlegend=False
        )
        hist_initial_figure.update_yaxes(
            showticklabels=False,
            gridcolor='lightgrey'
        )

        if chosen_cell is None:
            initial_college_name =html.H2(top_univ['INSTNM'],style={'fontFamily': 'Noto Serif, serif', 'fontSize': '20px','color': '#3F6121',
                                                    'fontWeight' : '600','textAlign':'center'})
            return html.H2(['Click on a university in the top list to see the historical trend.',html.Br(),'To check the percentage change, hover the nodes.'], \
                        style={'fontFamily': 'Noto Serif, serif','fontSize': '15px', 'fontWeight': '400','marginLeft': '5px', 'marginTop': '0px'}),\
                initial_college_name, dcc.Graph(id='hist-trend-plots',figure=hist_initial_figure,style={'height': '550px'})

        
        chosen_row = chosen_cell['row']
        # You may need to access the data for the chosen table based on 'chosen_table'

        chosen_college_name = right_table_data.iloc[chosen_row]['INSTNM']
        chosen_college_df = right_table_data[right_table_data['INSTNM'] == chosen_college_name]
        chosen_college_id = chosen_college_df.iloc[0]['UNITID']

        # update df
        updated_univ_hist_df = historical_data[historical_data['UNITID'] == chosen_college_id].copy()

        updated_univ_hist_df['delta_AdmRate'] = updated_univ_hist_df['AdmRate'].pct_change() * 100
        updated_univ_hist_df['delta_AdmRate'] = updated_univ_hist_df['delta_AdmRate'].fillna(0)

        updated_univ_hist_df['delta_NetPrice'] = updated_univ_hist_df['NetPrice'].pct_change() * 100
        updated_univ_hist_df['delta_NetPrice'] = updated_univ_hist_df['delta_NetPrice'].fillna(0)

        updated_univ_hist_df['delta_Debt'] = updated_univ_hist_df['Debt'].pct_change() * 100
        updated_univ_hist_df['delta_Debt'] = updated_univ_hist_df['delta_Debt'].fillna(0)

        updated_figure = make_subplots(rows=3, cols=1, subplot_titles=("Admission Rate", "Net Price", "Debt"))

        updated_figure.append_trace(go.Scatter(x=updated_univ_hist_df['YEAR'], y=updated_univ_hist_df['AdmRate'] * 100,
                                            customdata=updated_univ_hist_df['delta_AdmRate'], mode='markers+lines+text',
                                            hovertemplate=
                                            'Percent Change: %{customdata:.2f}%<extra></extra>',
                                            texttemplate='%{y:.2f}%', textposition='top center', textfont=dict(size=8),
                                            cliponaxis=False), row=1, col=1)
        updated_figure.append_trace(go.Scatter(x=updated_univ_hist_df['YEAR'], y=updated_univ_hist_df['NetPrice'],
                                            customdata=updated_univ_hist_df['delta_NetPrice'], mode='markers+lines+text',
                                            hovertemplate=
                                            'Percent Change: %{customdata:.2f}%<extra></extra>',
                                            texttemplate='$%{y:.2f}', textposition='top center', textfont=dict(size=8),
                                            cliponaxis=False), row=2, col=1)
        updated_figure.append_trace(go.Scatter(x=updated_univ_hist_df['YEAR'], y=updated_univ_hist_df['Debt'],
                                            customdata=updated_univ_hist_df['delta_Debt'], mode='markers+lines+text',
                                            hovertemplate=
                                            'Percent Change: %{customdata:.2f}%<extra></extra>',
                                            texttemplate='$%{y:.2f}', textposition='top center', textfont=dict(size=8),
                                            cliponaxis=False), row=3, col=1)
        updated_figure.update_annotations(font_size=14, yshift=10)
        updated_figure.update_layout(
            plot_bgcolor='white',
            font=dict(size=10),
            autosize=False, width=625, height=550,
            margin=dict(l=30, r=30, t=30, b=30),
            showlegend=False
        )
        updated_figure.update_yaxes(
            showticklabels=False,
            gridcolor='lightgrey'
        )

        updated_university_name = html.H2(chosen_college_name, style={'fontFamily': 'Noto Serif, serif', 'fontSize': '20px','color': '#3F6121',
                                        'fontWeight' : '600','textAlign':'center'})

        return html.H2(['Click on a university in the top list to see the historical trend.',html.Br(),'To check the percentage change, hover the nodes.'], \
                        style={'fontFamily': 'Noto Serif, serif','fontSize': '15px', 'fontWeight': '400','marginLeft': '5px', 'marginTop': '0px'}),\
            updated_university_name, dcc.Graph(id='hist-trend-plots', figure=updated_figure, style={'height': '550px'})


@callback(
    Output('alert-message', 'style'),  # 'alert-message' is the ID of the alert paragraph
    [Input('submit-button', 'n_clicks')],
    [State('academic-fields', 'value'),
     State('location-state', 'value'),
     State('test-score-selector', 'value'),  # True for ACT, False for SAT
     State('ACT-score', 'value'),
     State('SAT-score', 'value')]
)
def show_alert(n_clicks, academic_fields, location_state, test_score_selector, act_score, sat_score):
    # Check if the button is clicked and if any required field is empty
    if n_clicks > 0:
        if not academic_fields or not location_state:
            return {'display': 'block', 'color': 'red', 'fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                    'fontWeight' : '300','marginLeft': '10px'}  # Make alert visible

        # Check for test scores based on the selector
        if test_score_selector and (act_score == 1 or act_score is None):
            return {'display': 'block', 'color': 'red','fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                    'fontWeight' : '300','marginLeft': '10px'}  # Make alert visible for ACT
        elif not test_score_selector and (sat_score == 400 or sat_score is None):
            return {'display': 'block', 'color': 'red','fontFamily': 'Noto Serif, serif', 'fontSize': '15px',
                    'fontWeight' : '300','marginLeft': '10px'} # Make alert visible for SAT

    return {'display': 'none'}  # Keep alert hidden




if __name__ == '__main__':
    app.run_server(debug=True)
