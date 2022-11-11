''' 	Author: Emmanuel Chukwuka Oraegbu
	Email: ecoraegbu@gmail.com
	Course: ALX-T Data analytics
 '''

# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")

#store different color in variables for later use
dark_blue = sb.color_palette()[0]
orange = sb.color_palette()[1]
green =  sb.color_palette()[2]
red = sb.color_palette()[3]
purple = sb.color_palette()[4]
brown = sb.color_palette()[5]
grey = sb.color_palette()[7]

def change_levels(df, column_name, dictionary):
    row = 0
    for x in df[column_name]:
        for key, value in dictionary.items():
            if x == 0.0:
                df.at[row, column_name] = key
            if x > value[0] and x <= value[1]:
                df.at[row, column_name] = key
        row += 1
    return

# a func to plot a univariate countplot
def plot_count(dataframe, x_axis= None, y_axis = None, plot_title = None, plot_order = None, plot_color = None, 
               tick_rotation = None, plot_fontsize = None, tick_fontsize = 12, x_label = None, y_label = None, 
               plot_hue = None, plot_title_fontsize = 18, plot_figsize = (15, 5), bartext_fontsize = 12, show_count = False):
    if plot_hue == None:
    #This is for plotting univariate count plots  
    
        plt.rcParams["figure.figsize"] = list(plot_figsize)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['font.size'] = bartext_fontsize
        plot = sb.countplot(data = dataframe, x = x_axis, y = y_axis, order = plot_order, color = plot_color)
        if show_count == True:
            for p in plot.patches:
            #plot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+5))
                plot.text(p.get_x(), p.get_height(), '{:.0f}'.format(p.get_height(), va = 'center', ha = 'center',  fontdict = None,))
        plt.title(plot_title) if plot_title == None else plt.title(plot_title.upper(), fontsize = plot_title_fontsize)
        plt.xticks(fontsize = tick_fontsize, rotation = tick_rotation)
        plt.yticks(fontsize = tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
        plt.plot()
    else:
        #This would plot bivariate count plots or bar charts
        plt.rcParams["figure.figsize"] = list(plot_figsize)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams['font.size'] = bartext_fontsize
        plot = sb.countplot(data = dataframe, hue = plot_hue, x = x_axis, y = y_axis, order = plot_order, color = plot_color)
        for p in plot.patches:
            plot.text(p.get_x(), p.get_height(), '{:.0f}'.format(p.get_height(), va = 'center', ha = 'center',  fontdict = None,))
            #plot.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
        plt.title(plot_title) if plot_title == None else plt.title(plot_title.upper(), fontsize = plot_title_fontsize)
        plt.xticks(rotation = tick_rotation)
        plt.xticks(fontsize = 12 if tick_fontsize == None else tick_fontsize, rotation = tick_rotation)
        plt.yticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
        plt.tight_layout()
        plt.plot()
plt.show();

# A function to plot a pie chart
def plot_pie(dataframe, column, plot_startangle = 0, plot_counterclock = False, plot_title = None, plot_fontsize = 12, 
             plot_title_fontsize = 18, plot_figsize = (8,8)):
    counts = dataframe[column].value_counts()
    label = counts.index
    plt.figure(figsize = plot_figsize);
    plt.pie(counts, startangle = plot_startangle, counterclock = plot_counterclock, autopct='%.3f%%' )
    plt.legend(labels = label, fontsize = plot_fontsize, loc= 'center right', bbox_to_anchor = (1.2, 0.5))
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_fontsize)
    plt.axis('square');
    
    # define a func to plot hist
def sb_plot_hist(dataframe, x_axis = None, y_axis = None, plot_title = None, plot_title_font_size = 18,
              x_label = None, y_label = None, plot_fontsize = 12, plot_bin_number = 500, plot_color = None, 
              tick_fontsize = 12, plot_figsize = (15,10)):
    plt.figure(figsize=plot_figsize)
    plot = sb.histplot(data = dataframe, x = x_axis, bins = plot_bin_number, 
                       color = plot_color)
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_fontsize)
    plt.xticks(fontsize = tick_fontsize)
    plt.yticks(fontsize =  tick_fontsize)
    plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
    plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize);
    
    
def plt_plot_hist(dataframe, x_axis = None, y_axis = None, bin_num = 1000, plot_title = None, plot_title_font_size = 18,
                 x_label = None, y_label = None, plot_fontsize = 12, plot_color = None, tick_fontsize = 12, 
                  plot_figsize = (15,10), bin_edge = 0, axis = [], axis_set = False):
    bin = np.arange(bin_edge, dataframe[y_axis if x_axis == None else x_axis].max()+bin_num, bin_num)
    plt.figure(figsize = plot_figsize)
    plt.hist(data = dataframe, x = x_axis, bins = bin);
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
    plt.xticks(fontsize = tick_fontsize)
    plt.yticks(fontsize =  tick_fontsize)
    plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
    plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize);
    plt.axis(axis) if axis_set == True else []
    
# define a func to plot kernel density estimate
def plot_kde(dataframe, x_axis = None, y_axis = None, plot_title = None, x_label = None, y_label = None, 
             should_fill = False, plot_fontsize = None, tick_fontsize = None, plot_figsize =(15,5) ):
    #dataset = sb.load_dataset('dataframe')
    plt.figure(figsize= plot_figsize)
    sb.kdeplot(data = dataframe, x = x_axis, y = y_axis, fill = should_fill)
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
    plt.xticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
    plt.yticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
    plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
    plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
    
def sb_scatter_plot(dataframe, x_axis = None, y_axis = None, plot_title = None, plot_title_font_size = 18, x_label = None, 
             y_label = None, plot_fontsize = 12, plot_color = None, tick_fontsize = 12, plot_figsize = (15,10),
#if you want to plot a regplot, you supply the regplot title and regplot = True
                    reg_plot = False, reg_plot_title = None, plot_hue = None, marker = None , alpha = 0.25, cmap = 'cmap'):
    #print(dataframe, x_axis, y_axis)
    #make arrangements to accomodate jitter and transparency
    if plot_hue == None:
        plt.figure(figsize=plot_figsize)
        plot = sb.scatterplot(data = dataframe, x = x_axis, y = y_axis)
        plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
        plt.xticks(fontsize = tick_fontsize)
        plt.yticks(fontsize =  tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
        plt.show();
        if reg_plot == True:
            plot = sb.regplot(data = dataframe, x = x_axis, y = y_axis)
            plt.title('' if reg_plot_title == None else reg_plot_title.upper(), fontsize = plot_title_font_size)
            plt.xticks(fontsize = tick_fontsize)
            plt.yticks(fontsize =  tick_fontsize)
            plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
            plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
            plt.show();
    else:
        plt.figure(figsize=plot_figsize)
        plot = sb.scatterplot(data = dataframe, x = x_axis, y = y_axis, hue = plot_hue, marker = marker, 
                              alpha = alpha, cmap = cmap)
        plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
        plt.xticks(fontsize = tick_fontsize)
        plt.yticks(fontsize =  tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
        plt.show();

def plot_heat_map(dataframe, columns = (), plot_title = None, plot_title_font_size = 18,x_label = None, 
                  y_label = None, plot_fontsize = 12, plot_color = None, tick_fontsize = 12, plot_figsize = (15,10)):
    plt.figure(figsize=plot_figsize)
    sb.set()
    dataset = sb.load_dataset(dataframe)
    dataset = dataset.map(pivot, columns)
    sb.heatmap(dataset)
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
    plt.xticks(fontsize = tick_fontsize)
    plt.yticks(fontsize =  tick_fontsize)
    plt.show();

#Violin plots are used to show the relationships between a quantitative and a qualitative variable
# x_axis must be the categorical variable

# sedan_classes = ['Minicompact Cars', 'Subcompact Cars', 'Compact Cars', 'Midsize Cars', 'Large Cars']
# Change the qualitative variable type using this example
# vclasses = pd.api.types.CategoricalDtype(ordered=True, categories=sedan_classes)
# fuel_econ['VClass'] = fuel_econ['VClass'].astype(vclasses);
'''
Make sure to fill out null values before proceeding to plot your visualizations with this function
Plot the categorical variable on the x_axis
'''
def plot_violin(dataframe, x_axis = None, y_axis = None, plot_title = None, plot_title_font_size = 18,
              x_label = None, y_label = None, plot_fontsize = 12, plot_color = None, tick_fontsize = 12, 
                     plot_figsize = (15,10), tick_rotation = None, univariate = False, bivariate = False):
#the categorical/qualitative variable must be the x_axis
# we are going to use occupation and estimatedreturn to test this function
#types of occupation 
#Prepare the categorical data for analysis by changing the datatype
    if univariate == True:
        plt.figure(figsize=plot_figsize)
        sb.violinplot(data= dataframe, x= x_axis,)
        plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
        plt.xticks(fontsize = tick_fontsize, rotation = tick_rotation)
        plt.yticks(fontsize =  tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
        plt.show();
    elif bivariate == True:
        variable_classes = dataframe[x_axis].unique()
        _classes = pd.api.types.CategoricalDtype(ordered = True, categories = variable_classes)
        dataframe[x_axis] = dataframe[x_axis].astype(_classes)
        plt.figure(figsize=plot_figsize)
        sb.violinplot(data= dataframe, x= x_axis, y= y_axis)
        plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
        plt.xticks(fontsize = tick_fontsize, rotation = tick_rotation)
        plt.yticks(fontsize =  tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
        plt.show();
    else:
        return('please set univariate = True if you are plotting a single variable or bivariate = True for two variables')

# box plots are used to show the relationship between a qualitative and a numerical variable
# x_axis must be the categorical variable 
'''
Make sure to fill out null values before proceeding to plot your visualizations with this function
The categorical variable must be assigned to the x_axis
horizontal is used to change the axis of the plot
we can plot univariate box plots by setting the univariate argument to True
'''
def plot_boxplot(dataframe, x_axis= None, y_axis = None, plot_title = None, plot_color = None, tick_rotation = None, 
                 plot_fontsize = None, tick_fontsize =12 , x_label = None, y_label = None, plot_figsize =(15,10), 
                 horizontal = False, univariate = False, bivariate = False):
    if univariate == True:
        plt.figure(figsize = plot_figsize)
        sb.boxplot(data = dataframe, x = x_axis, color = plot_color)
        plt.title(plot_title) if plot_title == None else plt.title(plot_title.upper(), fontsize = 18 if plot_fontsize == None else plot_fontsize)
        plt.xticks(rotation = tick_rotation)
        plt.xticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
        plt.yticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
        plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
        plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
        plt.tight_layout()
        plt.show();
    elif bivariate == True:
        #Prepare the categorical data for analysis by changing the datatype
        variable_classes = dataframe[x_axis].unique()
        _classes = pd.api.types.CategoricalDtype(ordered = True, categories = variable_classes)
        dataframe[x_axis] = dataframe[x_axis].astype(_classes)    
        if horizontal == True:
                plt.figure(figsize = plot_figsize)
                sb.boxplot(data = dataframe, y = x_axis, x = y_axis, color = plot_color)
                plt.title(plot_title) if plot_title == None else plt.title(plot_title.upper(), fontsize = 18 if plot_fontsize == None else plot_fontsize)
                plt.xticks(rotation = tick_rotation)
                plt.xticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
                plt.yticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
                plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
                plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
                plt.tight_layout()
                plt.show();
        else:
                plt.figure(figsize = plot_figsize)
                sb.boxplot(data = dataframe, x = x_axis, y = y_axis)
                plt.title(plot_title) if plot_title == None else plt.title(plot_title.upper(), fontsize = 18 if plot_fontsize == None else plot_fontsize)
                plt.xticks(rotation = tick_rotation)
                plt.xticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
                plt.yticks(fontsize = 12 if tick_fontsize == None else tick_fontsize)
                plt.xlabel('' if x_label == None else x_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize)
                plt.ylabel('' if y_label == None else y_label.upper(), fontsize = 12 if plot_fontsize == None else plot_fontsize);
                plt.tight_layout()
                plt.show();
    else:
        return('Warning! please set univariate = True if you are plotting a single variable or bivariate = True for two variables')

def plot_lineplot(dataframe, x_axis = None, y_axis = None, plot_title = None, plot_title_font_size = 18,
              x_label = None, y_label = None, plot_fontsize = 12, plot_color = None, tick_fontsize = 12, 
                     plot_figsize = (15,10)):
    plt.figure(figsize=plot_figsize)
    plt.title('' if plot_title == None else plot_title.upper(), fontsize = plot_title_font_size)
    plt.xticks(fontsize = tick_fontsize)
    plt.yticks(fontsize =  tick_fontsize)
    plt.xlabel('' if x_label == None else x_label.upper(), fontsize = plot_fontsize)
    plt.ylabel('' if y_label == None else y_label.upper(), fontsize = plot_fontsize)
    plt.show();
    
    
def faceting(df, xa = None, ya = None, title = None, title_font_size = 18,
              x_label = None, y_label = None, pfont = 12, color = None, tfont = 12, 
                     fig_size = (15,10), ordered = False):
    '''
    Always assign the categorical variable to the xa
    facets can be plotted in order of increasing means by using the argument ordered = True
    '''
    #Prepare the categorical data for analysis by changing the datatype
    variable_classes = df[xa].unique()
    _classes = pd.api.types.CategoricalDtype(ordered = True, categories = variable_classes)
    df[xa] = df[xa].astype(_classes)
    
    group_means = df[[xa, ya]].groupby([xa]).mean()
    group_order = group_means.sort_values([ya], ascending = False).index

    
    plt.figure(figsize=fig_size)
    g = sb.FacetGrid(data = df, col = xa, col_wrap = 4, col_order = group_order if ordered == True else None)
    g.map(plt.hist, ya);
    
    plt.title('' if title == None else title.upper(), fontsize = title_font_size)
    plt.xticks(fontsize = tfont)
    plt.yticks(fontsize =  tfont)
    plt.xlabel('' if x_label == None else x_label.upper(), fontsize = pfont)
    plt.ylabel('' if y_label == None else y_label.upper(), fontsize = pfont)
    plt.show();
    
def to_categorical_dtype(dataframe, column_names = []):
    for column_name in column_names:
        variable_classes = dataframe[column_name].unique()
        _classes = pd.api.types.CategoricalDtype(ordered = True, categories = variable_classes)
        dataframe[column_name] = dataframe[column_name].astype(_classes)

        
        


#function to change values to 1s and 0s
def to_binary(dataframe, column_names = []):
    for column in column_names:
        #print(column)
        i = 0
        for row in dataframe[column]:
            if dataframe[column][i] > 0:
                dataframe.at[i, column] = 1
            else:
                dataframe.at[i, column] = 0
            i+= 1



#funtion to detect outliers
def outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    factor = iqr*1.5
    low_b = q1-factor
    upper_b = q3+factor
    wl = df[column]>low_b
    wh = df[column]<upper_b

    return 
