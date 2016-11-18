from os.path import dirname, join

import numpy as np
import pandas.io.sql as psql
import sqlite3 as sql

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput, PreText
from bokeh.io import curdoc
from bokeh.sampledata.movies_data import movie_path

from os.path import join, dirname
import datetime

import pandas as pd
from scipy.signal import savgol_filter

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DataRange1d, Select
from bokeh.palettes import Blues4
from bokeh.plotting import figure

import collections
import pandas

#conn = sql.connect(movie_path)
#query = open(join(dirname(__file__), 'query.sql')).read()
#movies = psql.read_sql(query, conn)
mypath = 'C:\\Users\\InnoblativeResearch\\Desktop\\AllCodes\\'
results_filename = mypath + 'results.csv'
ablation = pd.read_csv(results_filename)

ablation["color"] = np.where(ablation["face"] == 'N', "red", 
                    np.where(ablation["face"] == 'E', "green",
                    np.where(ablation["face"] == 'S', "orange",
                    np.where(ablation["face"] == 'W', "blue", "purple"))))
                    #np.where(ablation["face"] == 'B', "purple")
ablation["alpha"] = np.where(ablation["face"] == "HI", 0.25, 0.01) # Should only be 0.75
#movies.fillna(0, inplace=True)  # just replace missing values with zero
#movies["revenue"] = movies.BoxOffice.apply(lambda x: '{:,d}'.format(int(x)))

## For specific color and alpha levels --- don't need though
#with open(join(dirname(__file__), "razzies-clean.csv")) as f:
#    razzies = f.read().splitlines()
#movies.loc[movies.imdbID.isin(razzies), "color"] = "purple"
#movies.loc[movies.imdbID.isin(razzies), "alpha"] = 0.9

## TODO : Figure out what the values is 
#axis_map = {
#    "Tomato Meter": "Meter",
#    "Numeric Rating": "numericRating",
#    "Number of Reviews": "Reviews",
#    "Box Office (dollars)": "BoxOffice",
#    "Length (minutes)": "Runtime",
#    "Year": "Year",
#}

axis_map = { 
    "Probe Distance (mm)": "position",
    "Model Difference (Predicted - Expected)": "difference",
    "Time (s)" : "time",
    "Predicted" : "predicted",
    "Expected" : "expected"
    #"Model Percentage": "Percentage",
    #"Number of Ablations": "NumAblations",
}

face_map = []
face_map.append('All')
face_map.extend(list(set(ablation['face'])))
#position_map = [str(x) for x in list(set(ablation['position']))]
position_map = []
position_map.append('All')
position_map.extend([str(x) for x in np.arange(0,15,0.5)]) # use this to keep the positions in order

# Description at the top
desc = Div(text = open(mypath + "description.html").read(), width=800)

# Create Input controls
#reviews = Slider(title="Minimum number of reviews", value=80, start=10, end=300, step=10)
#min_year = Slider(title="Year released", start=1940, end=2014, value=1970, step=1)
#max_year = Slider(title="End Year released", start=1940, end=2014, value=2014, step=1)
#oscars = Slider(title="Minimum number of Oscar wins", start=0, end=4, value=0, step=1)
#boxoffice = Slider(title="Dollars at Box Office (millions)", start=0, end=800, value=0, step=1)
#genre = Select(title="Genre", value="All",
#               options=open(join(dirname(__file__), 'genres.txt')).read().split())
#director = TextInput(title="Director name contains")
#cast = TextInput(title="Cast names contains")
x_axis = Select(title = "X Axis", options = sorted(axis_map.keys()), value = "Probe Distance (mm)")
y_axis = Select(title = "Y Axis", options = sorted(axis_map.keys()), value = "Model Difference (Predicted - Expected)")
face = Select(title = "Face", options = face_map, value = 'All')
position = Select(title = "Position", options = position_map, value = 'All')
stats1 = PreText(text = '', width = 400)
stats2 = PreText(text = '', width = 400)
stats3 = PreText(text = '', width = 400)
stats4 = PreText(text = '', width = 400)
stats5 = PreText(text = '', width = 400)
stats6 = PreText(text = '', width = 400)
stats7 = PreText(text = '', width = 400)
stats8 = PreText(text = '', width = 400)

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data = dict(x = [], y = [], color = [], alpha=[],
                                    face = [], position = []))  ## This row is what we're going to have hover
source2 = ColumnDataSource(data = dict(x = [], y = []))

# Don't need hover tools -- too many data points
#hover = HoverTool(tooltips=[
#    ("Face", "@face"),
#    ("Position", "@postion"),
#    ("Impedance", "@impedance")
#])

p1 = figure(plot_height = 500, plot_width = 600, title = "", toolbar_location = None, tools = []) # tools = [hover]
p1.circle(x = "x", y = "y", source = source, size = 7, color = "color", line_color = None, fill_alpha = "alpha")

p2 = figure(plot_height = 400, plot_width = 500, title = "", toolbar_location = None, tools = []) # tools = [hover]
p2.vbar(x = "x", width = 0.5, top = "y", source = source2, color = "#CAB2D6") # vbar in the second to latest update
#http://stackoverflow.com/questions/30696486/how-can-i-add-text-annotation-in-bokeh
#http://bokeh.pydata.org/en/latest/docs/user_guide/annotations.html#userguide-annotations

def select_ablations():
    face_val = face.value
    position_val = position.value

    selected = ablation
    #genre_val = genre.value
    #director_val = director.value.strip()
    #cast_val = cast.value.strip()
    #selected = movies[
    #    (movies.Reviews >= reviews.value) &
    #    (movies.BoxOffice >= (boxoffice.value * 1e6)) &
    #    (movies.Year >= min_year.value) &
    #    (movies.Year <= max_year.value) &
    #    (movies.Oscars >= oscars.value)
    #]

    if (face_val != "All"):
        selected = selected[selected.face.str.match(face_val) == True] # use match over time just cause its better
    if (position_val != "All"):
        selected = selected[selected.position.astype('str').str.match(position_val) == True]

    #if (genre_val != "All"):
    #    selected = selected[selected.Genre.str.contains(genre_val)==True]
    #if (director_val != ""):
    #    selected = selected[selected.Director.str.contains(director_val)==True]
    #if (cast_val != ""):
    #    selected = selected[selected.Cast.str.contains(cast_val)==True]

    return selected


def update():
    df = select_ablations() # when control changes -  calls update, which calls select ablations
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p1.xaxis.axis_label = x_axis.value
    p1.yaxis.axis_label = y_axis.value
    p1.title.text = "%d ablations selected - N:red, E:green, S:orange, W:blue, B:purple" % len(df)
    source.data = dict( # source is the data fed for plotting
        x = df[x_name],
        y = df[y_name],
        color = df["color"],
        alpha = df["alpha"],
        face = df["face"],
        position = df["position"]       
    )

    bottoms, tops = update_stats(df)
    source2.data = dict(
        x = bottoms,
        y = tops)

    return


def update_stats(df):
    #stats.text = str(df['face']) # This works -- TODO: change to specific measure

    # This should we where we calculate the stats... I think
    face_val = face.value
    position_val = position.value

    total = len(df[df['difference'] != 1337] == True)
    correct = len(df[df['difference'] == 0] == True)
    wrong = len(df[df['difference'] != 0] == True)

    above_one = len(df[df['difference'] == 1] == True)
    above_two = len(df[df['difference'] == 2] == True)
    above_three = len(df[df['difference'] == 3] == True)
 
    below_one = len(df[df['difference'] == -1] == True)
    below_two = len(df[df['difference'] == -2] == True)
    below_three = len(df[df['difference'] == -3] == True)

    # TODO: make this easier instead of so many texts lols
    stats1.text = 'Percent correct %2.2f %% =  %d // %d' % (correct/total, correct, total)
    stats2.text = 'Percent wrong %2.2f %% =  %d // %d' % (wrong/total, wrong, total)
    stats3.text = 'Number of ablations predicted one above: %d' % above_one
    stats4.text = 'Number of ablations predicted two above: %d' % above_two
    stats5.text = 'Number of ablations predicted three above: %d' % above_three
    stats6.text = 'Number of ablations predicted one below: %d' % below_one
    stats7.text = 'Number of ablations predicted two below: %d' % below_two
    stats8.text = 'Number of ablations predicted three below: %d' % below_three
    
    x = [-3, -2, -1, 0, 1, 2, 3]
    y = [below_three, below_two, below_one, correct, above_one, above_two, above_three]

    #x = [str(i) for i in bottoms],
    bottoms = x #pandas.core.frame.DataFrame(x).astype('str')
    tops = y #pandas.core.frame.DataFrame(y).astype('str')
    #y = [str(i) for i in tops])
    return bottoms, tops

controls = [y_axis, x_axis, face, position]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode = sizing_mode)
l = layout([
    [desc],
    [column(inputs, stats1, stats2, stats3, stats4, stats5, stats6, stats7, stats8), column(p1, p2)]
], sizing_mode = sizing_mode)

# TODO: Test this
#[desc],
#[inputs],
#[row(p1, p2)] # Title of P1 will have the correct and wrong %


#example: 
#widgets = column(ticker1, ticker2, stats)
#main_row = row(corr, widgets)
#series = column(ts1, ts2)
#layout = column(main_row, series)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Ablations"