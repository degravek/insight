from math import pi
import pandas as pd
import numpy as np
import seaborn as sns
from bokeh import mpl
np.random.seed(0)

from bokeh.io import curdoc
from bokeh.charts import Bar
from bokeh.models.widgets import Panel, Tabs, DataTable
from bokeh.models.widgets import TableColumn, CheckboxGroup, TextInput
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.charts.attributes import ColorAttr, CatAttr
from bokeh.models import Range1d
from bokeh.plotting import figure

data_ngram1 = pd.read_csv('/Users/degravek/Downloads/df_ngram1.csv')
data_ngram2 = pd.read_csv('/Users/degravek/Downloads/df_ngram2.csv')
data_ngram3 = pd.read_csv('/Users/degravek/Downloads/df_ngram3.csv')
data_chunk  = pd.read_csv('/Users/degravek/Downloads/df_chunk.csv')
data_rake   = pd.read_csv('/Users/degravek/Downloads/df_rake.csv')

# Define some helper functions
def SortData(input_df, n_expon, rfilter=None):
    if rfilter:
        input_df = input_df[input_df['rating'].isin(rfilter)].copy()

    input_df['counts'] = input_df.groupby(['aspects'])['sentiment'].transform('count')
    group1 = input_df.groupby(['aspects'])['counts'].mean()
    group2 = input_df.groupby(['aspects'])['sentiment'].sum()
    group3 = input_df.groupby(['aspects'])['sentiment'].mean()
    sorted_df = pd.DataFrame()
    sorted_df['counts']          = group1
    sorted_df['mean sentiment']  = np.round(group3, 2)
    sorted_df['importance']      = np.round(group2/(group1**n_expon), 2)
    sorted_df = sorted_df.sort_values('importance', ascending=False)
    sorted_df.reset_index(level=0, inplace=True)
    return sorted_df

def get_dataset(dataset, n_expon, n_stars):
    if dataset == 'n-gram 1':
        result = SortData(data_ngram1, n_expon, n_stars)
    elif dataset == 'n-gram 2':
        result = SortData(data_ngram2, n_expon, n_stars)
    elif dataset == 'n-gram 3':
        result = SortData(data_ngram3, n_expon, n_stars)
    elif dataset == 'chunk':
        result = SortData(data_chunk, n_expon, n_stars)
    elif dataset == 'rake':
        result = SortData(data_rake, n_expon, n_stars)
    return result

def get_ends(n_samples):
    df = pd.DataFrame(source_all.data)
    result = df.head(n_samples).append(df.tail(n_samples))
    return result


# Set up initial parameters
n_samples = 15
n_stars = [1,2,3,4,5]
n_expon = 0.1
dataset = 'n-gram 1'


# Set up dataset source
newdata = get_dataset(dataset, n_expon, n_stars)
source_all = ColumnDataSource(data=newdata.to_dict('list'))

newdata = get_ends(n_samples)
source_cut = ColumnDataSource(data=newdata.to_dict('list'))

# Set up plot (styling in theme.yaml)
plot = figure(toolbar_location=None, plot_width=700, plot_height=500, x_range=source_cut.data['aspects'],
                y_range=Range1d(min(source_cut.data['importance']), max(source_cut.data['importance'])))
plot.vbar(x='aspects', width=0.5, bottom=0, top='importance', source=source_cut)
plot.xaxis.major_label_orientation = pi/2


columns = [
    TableColumn(field="aspects",        title="Aspects"),
    TableColumn(field="counts",         title="Counts"),
    TableColumn(field="mean sentiment", title="Mean Sentiment"),
    TableColumn(field="importance",     title="Importance")
]

datasets_names = [
    'n-gram 1',
    'n-gram 2',
    'n-gram 3',
    'chunk',
    'rake'
]

dataset_select = Select(value='n-gram 1',
                        title='Select Dataset:',
                        width=200,
                        options=datasets_names)

exponent_slider = Slider(title="Weighted Sum Exponent",
                        value=0.1,
                        start=0,
                        end=1,
                        step=0.1,
                        width=200)

samples_slider = Slider(title="Number of Aspects",
                        value=15,
                        start=1,
                        end=30,
                        step=1,
                        width=200)

ratings_box = TextInput(value="1,2,3,4,5", title="Star Rating:", width=185)


def update_dataset(attrname, old, new):
    dataset = dataset_select.value
    n_expon = exponent_slider.value
    n_samples = int(samples_slider.value)
    n_stars = [int(val) for val in ratings_box.value.split(',')]

    newdata = get_dataset(dataset, n_expon, n_stars)
    source_all.data = dict(newdata.to_dict('list'))

    newdata = get_ends(n_samples)
    source_cut.data = dict(newdata.to_dict('list'))

    plot.x_range.factors = newdata['aspects'].tolist() # this was missing
    plot.y_range.start = min(source_cut.data['importance'])
    plot.y_range.end = max(source_cut.data['importance'])

def update_slider(attrname, old, new):
    n_samples = int(samples_slider.value)

    newdata = get_ends(n_samples)
    source_cut.data = dict(newdata.to_dict('list'))

    plot.x_range.factors = newdata['aspects'].tolist() # this was missing
    plot.y_range.start = min(source_cut.data['importance'])
    plot.y_range.end = max(source_cut.data['importance'])

data_table = DataTable(source=source_cut, columns=columns, width=700, height=500)

dataset_select.on_change('value', update_dataset)
exponent_slider.on_change('value', update_dataset)
ratings_box.on_change('value', update_dataset)
samples_slider.on_change('value', update_slider)

# Set up layout
selects = row(dataset_select)
inputs = column(selects, widgetbox(exponent_slider, ratings_box, samples_slider))
table = widgetbox(data_table)

tab1 = Panel(child=table, title="Data")
tab2 = Panel(child=plot, title="Bar Plot")
tabs = Tabs(tabs=[tab1, tab2])
lay = layout([[inputs,tabs],]) 

# Add to document
curdoc().add_root(lay)
curdoc().title = "Keyword Extraction"
