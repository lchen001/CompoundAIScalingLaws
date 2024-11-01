import pandas as pd
from collections import Counter
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.optimize import minimize
import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger('simulator')

color_filter = px.colors.qualitative.Plotly[0]
color_vote = px.colors.qualitative.Plotly[1]
color_line = px.colors.qualitative.Plotly[2]
colors = [color_vote,color_filter]

## Load datasets

def load_raw_data(dataname = 'gpqa',
                  cate = 'gpqa_diamond',
                  model_name ='gpt-3.5-turbo-0125',
                 use_filter=False,
                 ):

    Kmin=0
    Kmax=0
    gen_max_id = 401
    enhance_layer_n_list =[0]
    folder_name = f"../results/{dataname}/"
    
    judge_type = 'single_step'
    dfs = []
    for n in enhance_layer_n_list:
        data1 = pd.read_csv(f"{folder_name}/{dataname}_{cate}_{model_name}_{Kmin}_{Kmax}_{gen_max_id}_{n}_{use_filter}_{judge_type}_all.csv")
        data1['en_n'] = n
        dfs.append(data1)
    df_concatenated = pd.concat(dfs, ignore_index=True)
    return df_concatenated

## Compute Metric

def compute_dVx(df):
    def calculate_dVx(row):
        # Extract the dictionary of probabilities and the true answer from the respective columns
        prob_dict = eval(row["full_possible_answer_count_before_filter"])
        true_answer = row["true_answer"]
        
        # Probability of the true answer
        #print(prob_dict)
        Pr_y_star = 0
        if(true_answer in prob_dict):
            Pr_y_star = prob_dict[true_answer]

        if (len(prob_dict) == 1 and true_answer in prob_dict):
            # If true_answer is the only item, there is no other probability to compare
            max_Pr_y_not_star = 0
        else:
            # Otherwise, find the maximum probability among all answers except the true answer
            max_Pr_y_not_star = max(value for key, value in prob_dict.items() if key != true_answer)
             
        # Compute dVx
        dVx = max_Pr_y_not_star - Pr_y_star
        return dVx
    
    # Apply the calculate_dVx function to each row in the DataFrame
    df['dVx'] = df.apply(calculate_dVx, axis=1)
    
    return df

def compute_dFx(df):
    def calculate_dVx(row):
        # Extract the dictionary of probabilities and the true answer from the respective columns
        prob_dict = eval(row["full_possible_answer_count"])
        true_answer = row["true_answer"]
        
        # Probability of the true answer
        #print(prob_dict)
        Pr_y_star = 0
        if(true_answer in prob_dict):
            Pr_y_star = prob_dict[true_answer]

        if (len(prob_dict) == 1 and true_answer in prob_dict):
            # If true_answer is the only item, there is no other probability to compare
            max_Pr_y_not_star = 0
        else:
            # Otherwise, find the maximum probability among all answers except the true answer
            max_Pr_y_not_star = max(value for key, value in prob_dict.items() if key != true_answer)
             
        # Compute dVx
        dVx = max_Pr_y_not_star - Pr_y_star
        return dVx
    
    # Apply the calculate_dVx function to each row in the DataFrame
    df['dFx'] = df.apply(calculate_dVx, axis=1)
    
    return df

def count_frequencies(items):
    # Count the frequency of each unique value in the list
    counter = Counter(items)
    
    # Total number of items in the list
    total_count = len(items)
    
    # Calculate the normalized frequency (probability) of each unique value
    frequency_dict = {item: count / total_count for item, count in counter.items()}
    print(total_count)
    return frequency_dict

'''
def get_simulate_perf(df, n, samples=1000):
    # Precompute probabilities and convert to arrays outside the inner function
    answers_and_probs = df['full_possible_answer_count'].apply(ast.literal_eval)
    keys_list = answers_and_probs.apply(lambda x: list(x.keys()))
    probs_list = answers_and_probs.apply(lambda x: np.array(list(x.values()), dtype=float))
    true_answers = df['true_answer'].values
    
    # Normalize probabilities
    probs_list = [probs / probs.sum() for probs in probs_list]

    def simulate(index):
        # Use numpy to simulate the sampling process more efficiently
        sampled_answers = np.random.choice(keys_list[index], size=n, p=probs_list[index], replace=True)
        most_common_answer = Counter(sampled_answers).most_common(1)[0][0]
        return most_common_answer == true_answers[index]

    # Run simulations and compute average accuracy
    results = np.mean([simulate(i) for i in range(len(df)) for _ in range(samples)])
    return results
'''

def get_simulate_perf_old(df, n, samples=1000,use_filter=False):
    # Process the answer counts into a usable form for numpy operations
    answer_counts = df['full_possible_answer_count'].apply(ast.literal_eval)
    
    # Create arrays for keys and probabilities, and normalize the probabilities
    keys = answer_counts.apply(lambda x: np.array(list(x.keys())))
    probabilities = answer_counts.apply(lambda x: np.array(list(x.values()), dtype=float))
    probabilities = probabilities.apply(lambda p: p / p.sum())

    results = np.zeros(len(df))

    for i, (k, p) in enumerate(zip(keys, probabilities)):
        # Simulate 'samples' number of trials at once
        sampled_matrix = np.random.choice(k, size=(samples, n), p=p)

        # Determine the most common answer for each simulation
        for j in range(samples):
            most_common_answer, _ = Counter(sampled_matrix[j]).most_common(1)[0]
            results[i] += (most_common_answer == df.iloc[i]['true_answer'])
    
    # Compute average accuracy per row, then mean over all rows
    results /= samples
    return results.mean()

def get_simulate_perf(df, n, samples=1000, use_filter=False):
    # Process the answer counts into a usable form for numpy operations
    answer_counts = df['full_possible_answer_count_before_filter'].apply(ast.literal_eval)
    
    # Create arrays for keys and probabilities, and normalize the probabilities
    keys = answer_counts.apply(lambda x: np.array(list(x.keys())))
    probabilities = answer_counts.apply(lambda x: np.array(list(x.values()), dtype=float))
    probabilities = probabilities.apply(lambda p: p / p.sum())

    results = np.zeros(len(df))

    for i, (k, p) in enumerate(zip(keys, probabilities)):
        #print(i,k,p)
        if use_filter:
            z = np.array(ast.literal_eval(df['z'].iloc[i]))
            w = np.array(ast.literal_eval(df['w'].iloc[i]))
            #print("z lens",len(z))
            for _ in range(samples):
                # Sample n indexes
                sampled_indexes = np.random.choice(len(z), size=int(n/2), replace=True)
                z_sampled = z[sampled_indexes]
                w_sampled = w[sampled_indexes]
                #print(type(w_sampled))

                #w_sampled = np.ones_like(z_sampled,dtype=float)  # Set each element in w_sampled to 1
                #print(w_sampled)
                #print(type(w_sampled))
                # Calculate common_after_filter
                unique_values = np.unique(z_sampled)
                weighted_counts = {val: np.sum(w_sampled[z_sampled == val]) for val in unique_values}
                #print("w counts::",weighted_counts)
                common_after_filter = max(weighted_counts, key=weighted_counts.get)

                # Check if the common_after_filter is the true answer
                results[i] += (common_after_filter == df.iloc[i]['true_answer'])
        else:
            # Simulate 'samples' number of trials at once
            sampled_matrix = np.random.choice(k, size=(samples, n), p=p)

            # Determine the most common answer for each simulation
            for j in range(samples):
                most_common_answer, _ = Counter(sampled_matrix[j]).most_common(1)[0]
                results[i] += (most_common_answer == df.iloc[i]['true_answer'])
    
    # Compute average accuracy per row, then mean over all rows
    results /= samples
    return results.mean()



def compute_performance_by_gen_en(df, gen_n_list, samples=1000,use_filter=False):
    # Initialize a list to hold the results
    results = []

    # Get unique en_n values from the dataframe
    unique_en_ns = df['en_n'].unique()

    for gen_n in tqdm(gen_n_list):
        for en_n in unique_en_ns:
            # Subset the dataframe once per en_n
            subset_df = df[df['en_n'] == en_n]
            
            # Compute performance once per subset
            performance = get_simulate_perf(subset_df, gen_n, samples,use_filter=use_filter)
            results.append({'gen_n': gen_n, 'en_n': en_n, 'performance': performance})
    
    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


## visualization

def plot_distributions_kde(data, font_size=14, line_width=2, fig_width=800, fig_height=600):
    """
    Plot the smooth distribution of dVx and dFx in the same figure using KDE with Plotly,
    formatted for a research paper.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'dVx' and 'dFx' columns.
    font_size (int): Font size for the labels and ticks.
    line_width (int): Line width for the KDE plots.
    fig_width (int): Width of the figure.
    fig_height (int): Height of the figure.

    Returns:
    fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    """
    # Extract dVx and dFx values
    dVx_values = data['dVx'].dropna()
    dFx_values = data['dFx'].dropna()
    
    # Create a KDE plot
    fig = ff.create_distplot(
        [dVx_values, dFx_values], 
        #group_labels=['dVx', 'dFx'],
        group_labels=['d<sub>V</sub>(x)', 'd<sub>F</sub>(x)'], 
        show_hist=False,
        show_rug=False,
        colors=['#1f77b4', '#ff7f0e']  # Distinguishable colors
    )

    # Set line width for the KDE plots
    for trace in fig.data:
        trace.line.width = line_width

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Density",
        template="plotly_white",
        xaxis=dict(
            title_font=dict(size=font_size, family='Times New Roman'),
            tickfont=dict(size=font_size, family='Times New Roman'),
            range=[-1, 1]
        ),
        yaxis=dict(
            title_font=dict(size=font_size, family='Times New Roman'),
            tickfont=dict(size=font_size, family='Times New Roman')
        ),
        legend=dict(
            font=dict(size=font_size, family='Times New Roman'),
            x=0.05,
            y=0.95
        ),
        margin=dict(l=60, r=20, t=20, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=fig_width,
        height=fig_height
    )

    return fig


def plot_performance(df_before, df_after, width=800, height=600, 
                     fontsize=14, marker_size=10, line_width=2,show_legend=False):
    # Create traces for each dataframe
    trace_before = go.Scatter(
        x=df_before['gen_n'],
        y=df_before['performance'],
        mode='lines+markers',
        name='Voting Systems',
        marker=dict(size=marker_size, symbol='circle', line=dict(width=1, color='black')),
        line=dict(width=line_width, color='blue')
    )
    
    trace_after = go.Scatter(
        x=df_after['gen_n'],
        y=df_after['performance'],
        mode='lines+markers',
        name='Filtering Systems',
        marker=dict(size=marker_size, symbol='square', line=dict(width=1, color='black')),
        line=dict(width=line_width, color='red')
    )
    
    # Create the figure and add traces
    fig = go.Figure()
    fig.add_trace(trace_before)
    fig.add_trace(trace_after)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Performance vs. Generator Number',
            font=dict(size=fontsize)
        ),
        xaxis=dict(
            title=dict(
                text='Number of LLM Calls',
                font=dict(size=fontsize)
            ),
            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            type='log',  # Optional: Set x-axis to logarithmic scale
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text='Performance',
                font=dict(size=fontsize)
            ),
            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        legend=dict(
            title=dict(
                #text='Data Filter Status',
                font=dict(size=fontsize)
            ),
            font=dict(size=fontsize),
            bordercolor='black',
            borderwidth=1,
            x=0.5,
            y=0.9,
             #show_legend=show_legend,

        ),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_layout(showlegend=show_legend)

    # Show the plot
    fig.show()
    return fig

def plot_conditioned_scatter(data, x_col, y_col, fig_size=(800, 600), font_size=14, marker_size=10):
    # Define conditions for coloring
    conditions = np.where(
        (data[x_col] < 0) & (data[y_col] > 0), 'easy Vote, hard Filter-Vote',
        np.where((data[x_col] > 0) & (data[y_col] < 0), 'hard Vote, easy Filter-Vote', 'Other')
    )
    
    # Create the scatter plot
    fig = px.scatter(data, x=x_col, y=y_col, color=conditions,
                     labels={'color': 'Condition', x_col: f'dV(x)', y_col: f'dF(x)'},
                     #title='Relationship between dVx and dFx with Highlighted Conditions',
                     color_discrete_map={
                         'easy Vote, hard Filter-Vote': color_vote,
                         'hard Vote, easy Filter-Vote': color_filter,
                         'Other': 'lightgrey'
                     },
                     template='plotly_white',
                     opacity=0.8,
                     width=fig_size[0],
                     height=fig_size[1])
    
    # Adjust marker size
    fig.update_traces(marker=dict(size=marker_size))
    
    # Customize font and layout
    fig.update_layout(
        font=dict(size=font_size),
        legend_title_text='',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title_font_size=font_size + 2
    )
    
    # Add zero lines for reference
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="grey")
    
    # Show the plot
    fig.show()

def plot_performance_vs_LLMCalls(df_before, df_after, width=800, height=600, 
                     fontsize=14, marker_size=10, line_width=2,show_legend=False, colors=['red','blue'],
                                dtick=0.1,
                                xaxis_range=[0.99,1005],
            h_legend=False,
                                ):
    # Create traces for each dataframe
    trace_before = go.Scatter(
        x=df_before['gen_n'],
        y=df_before['performance'],
        mode='lines+markers',
        name='Vote',
        marker=dict(size=marker_size, symbol='circle', line=dict(width=1, color='black')),
        line=dict(width=line_width, color=colors[0])
    )
 
    trace_after = go.Scatter(
        x=df_after['gen_n'],
        y=df_after['performance'],
        mode='lines+markers',
        name='Filter-Vote',
        marker=dict(size=marker_size, symbol='square', line=dict(width=1, color='black')),
        line=dict(width=line_width, color=colors[1])
    )
    
    # Create the figure and add traces
    fig = go.Figure()
    fig.add_trace(trace_before)
    fig.add_trace(trace_after)
    
    # Update layout
    fig.update_layout(
        title=dict(
            #text='Performance vs. Generator Number',
            font=dict(size=fontsize)
        ),
        xaxis=dict(
            title=dict(
                text='Number of LLM Calls',
                font=dict(size=fontsize)
            ),
            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            type='log',  # Optional: Set x-axis to logarithmic scale
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True,
            tickvals=[1, 10, 100, 1000],  # Custom tick values for clearer display
            ticktext=['1',  '10',  '100', '1k'],  # Custom tick text

            range = xaxis_range,

        ),
        yaxis=dict(
            title=dict(
                text='Performance',
                font=dict(size=fontsize)
            ),
            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True,
            dtick=dtick,
        ),
        legend=dict(
            title=dict(
                #text='Data Filter Status',
                font=dict(size=fontsize)
            ),
            font=dict(size=fontsize),
            bordercolor='black',
            borderwidth=1,
            x=0.5,
            y=0.97,
             #show_legend=show_legend,

        ),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white',
 
    )
    if(h_legend):
        fig.update_layout(legend=dict(
 orientation="h",
yanchor="bottom",
        x=0.5,
            y=2.0,
        ))
                      
    fig.update_layout(showlegend=show_legend)

    # Show the plot
    fig.show()
    return fig

def plot_performance_vs_LLMCalls_onecurve(df_before, width=800, height=600, 
                                 fontsize=14, marker_size=10, line_width=2, show_legend=False,color='red',
                                         xaxis_range=[0.99,1005],
                                          dtick=0.2,
                                         ):
    # Create trace for the dataframe
    trace_before = go.Scatter(
        x=df_before['gen_n'],
        y=df_before['performance'],
        mode='lines+markers',
        name='Vote',
        marker=dict(size=marker_size, symbol='circle', line=dict(width=1, color=color)),
        line=dict(width=line_width, color=color)
    )
 
    # Create the figure and add trace
    fig = go.Figure()
    fig.add_trace(trace_before)
    
    # Update layout
    fig.update_layout(
        title=dict(font=dict(size=fontsize)),
        xaxis=dict(
            title='# LLM Calls',
            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            type='log',  # Optional: Set x-axis to logarithmic scale
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            tickvals=[1, 10, 100, 1000],  # Custom tick values for clearer display
            ticktext=['1',  '10',  '100', '1k'],  # Custom tick text

            tickangle=0,  # Rotate tick labels

            mirror=True
        ),
        yaxis=dict(
            title='Perf',
                        #tickmode='auto',  # Enable automatic tick placement

            title_font=dict(size=fontsize),
            tickfont=dict(size=fontsize),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        legend=dict(
            font=dict(size=fontsize),
            bordercolor='black',
            borderwidth=1,
            x=0.5,
            y=0.97
        ),
        width=width,
        height=height,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_layout(showlegend=show_legend,
                    
#                      margin=dict(l=100, r=100, t=100, b=10),  # Add margin to ensure the axis lines are visible

                     
                     )

    fig.update_layout(
        #title=dict(font=dict(size=fontsize)),
        xaxis=dict(
            
            tickvals=[1, 10, 100, 1000],  # Custom tick values for clearer display
            ticktext=['1',  '10',  '100', '1k'],  # Custom tick text

            tickangle=0,  # Rotate tick labels
            range=xaxis_range,

        ),
        yaxis=dict(
                        #tickmode='auto',  # Enable automatic tick placement
            dtick=dtick,
        ),
)
    
    # Show the plot
    fig.show()
    return fig

def plot_scatter_plotly_combined(acc_simulate, acc_predicted, acc_simulate_filter, acc_predict_filter,
                                
                                figure_size=(800, 600), linewidth=2, markersize=10, fontsize=14,
                                
                                ):
    """
    Plots a scatter plot using Plotly to compare both original and filtered simulated and predicted accuracy values.
    All input arrays should be numpy arrays.

    Parameters:
    acc_simulate (numpy array): The original simulated accuracy values.
    acc_predicted (numpy array): The original predicted accuracy values.
    acc_simulate_filter (numpy array): The filtered simulated accuracy values.
    acc_predict_filter (numpy array): The filtered predicted accuracy values.
    """
    fig = go.Figure()

    # Add original data points
    fig.add_trace(go.Scatter(x=acc_simulate, y=acc_predicted, mode='markers',
                             marker=dict(color=color_vote, size=markersize, line=dict(width=linewidth, color='DarkSlateGrey')),
                             name='Vote'))

    # Add filtered data points
    fig.add_trace(go.Scatter(x=acc_simulate_filter, y=acc_predict_filter, mode='markers',
                             marker=dict(color=color_filter, size=markersize, line=dict(width=linewidth, color='DarkSlateGrey')),
                             name='Filter-Vote'))
    
    # Calculate the limits for the line of perfect prediction using numpy
    min_plot_val = np.min([np.min(acc_simulate), np.min(acc_predicted), np.min(acc_simulate_filter), np.min(acc_predict_filter)])
    max_plot_val = np.max([np.max(acc_simulate), np.max(acc_predicted), np.max(acc_simulate_filter), np.max(acc_predict_filter)])
    
    # Add a line of perfect prediction
    fig.add_trace(go.Scatter(x=[min_plot_val, max_plot_val], y=[min_plot_val, max_plot_val],
                             mode='lines', line=dict(color=color_line, dash='dash',width=linewidth),
                             name='Empirical==Predicted'))

    # Set plot layout
    fig.update_layout(#title='Comparison of Simulated vs. Predicted Accuracy',
                      xaxis_title='Empirical Performance',
                      yaxis_title='Predicted Performance',
                      legend_title="",
                    font=dict(size=fontsize),

                      width=figure_size[0], height=figure_size[1],
    
            legend=dict(
            font=dict(size=fontsize),
            bordercolor='black',
            borderwidth=1,
            x=0.1,
            y=0.95
        ),
        
    )



    fig.update_layout(xaxis_title='Empirical Performance',
                      yaxis_title='Predicted Performance',
                      font=dict(size=fontsize),
                      width=figure_size[0], height=figure_size[1],
                      legend=dict(font=dict(size=fontsize), bordercolor='black', borderwidth=1, x=0.02, y=0.97),
#                      legend=dict(font=dict(size=fontsize), bordercolor='black', borderwidth=1, orientation="h",
#    yanchor="bottom",),

                      
                      paper_bgcolor='white',  # Set background color to white
                      plot_bgcolor='white',  # Ensure plot background is also white
                      xaxis=dict(showline=True, linewidth=2, linecolor='black',mirror=True, showgrid=True, gridwidth=1, gridcolor='grey'),  # Show grid and black borders
                      yaxis=dict(showline=True, linewidth=2, linecolor='black', mirror=True,showgrid=True, gridwidth=1, gridcolor='grey'),  # Show grid and black borders
                      margin=dict(l=20, r=20, t=20, b=20),  # Add margin to ensure the axis lines are visible
                      )
    
    #fig.show()
    return fig






## Scale model

class ScaleLaws():
    def fitscalelaws(self,df,M=[1,10,100,1000,10000], trial=100,use_filter=False):
        params = fitscalelaws(df=df, M=M, trial=trial,use_filter=use_filter)
        self.params = params
        return params
    
    def predict(self,M=[1,10,100]):
        result = []
        params = self.params
        for param in params:
            G,(a1,a2,a3), scale = param
            G_predicted = [G(k, a1, a2, a3)/scale for k in M]
            result.append(np.array(G_predicted))
        stacked_arrays = np.stack(result, axis=0)

        # Compute the mean along the first axis (axis=0)
        average_array = np.mean(stacked_arrays, axis=0)
        return average_array
    
def fitscalelaws(df, M=[1,10,100,1000,10000], trial=100,use_filter=False):
    # Define a wrapper function to include M and trial in the call
    def wrapper(row):
        return fitscalelaws_onepoint(row, M=M, trial=trial,use_filter=use_filter)
    
    # Apply the wrapper function to each row
    # axis=1 specifies that the function should be applied to rows, not columns
    results = df.apply(wrapper, axis=1)
    return results

    
def fitscalelaws_onepoint(empirical_datapoint, M=[1,10,100,1000,10000],trial=100,use_filter=False):
    # each row of empirical_data is:
    '''
    x, y, counts
    '''
    # step 1: find the probability 
    y_probs = eval(empirical_datapoint['full_possible_answer_count'])
    label = empirical_datapoint['true_answer']
    # step 2: choose which function to fit
    max_key = max(y_probs, key=y_probs.get)
    
    max_value = max(y_probs.values())

    max_keys = [key for key, value in y_probs.items() if value == max_value]

    logger.debug(f"max keys are {max_keys}")
    
    scale = 1
    if(label in max_keys):
        if(len(max_keys)==1):
            G=G_1
            perf = [simulate_answer(empirical_datapoint,k=m,number_of_trials=trial,use_filter=use_filter) for m in M]

        else:
            G=G_1
            perf = [len(max_keys)*simulate_answer(empirical_datapoint,k=m,number_of_trials=trial,use_filter=use_filter) for m in M]
            scale = len(max_keys)
    else:
        G=G_2
        perf = [simulate_answer(empirical_datapoint,k=m,number_of_trials=trial,use_filter=use_filter) for m in M]

    # step 3: fit the curves
    logger.debug(f"training perf is {perf}")
    parms = fit_parameters(M,perf,G) 
    logger.debug(f"training param is {parms}")
    
    return G, parms, scale



# Define the new approximation function G_1(K) and G_2
def G_1(K, a1, a2,a3):
    return 1 - np.exp(-a1 * K - a2* np.sqrt(K) - a3)

def G_2(K, a1, a2,a3):
    exponent = np.clip(-a1 * K - a2 * np.sqrt(K) - a3, a_min=-70, a_max=70)  # np.exp's maximum safe value is around 709
    return np.exp(exponent)    
    #return np.exp(-a1 * K - a2* np.sqrt(K) - a3)

def simulate_answer(data_point, k=100, number_of_trials=5000, use_filter=False):
    if(use_filter):
        return simulate_answer_with_filter(data_point, k=k, number_of_trials=number_of_trials)
        
    # Ensure the data point counts is a concrete Python dictionary.
    if isinstance(data_point['full_possible_answer_count'], str):
        counts = eval(data_point['full_possible_answer_count'])
    else:
        counts = data_point['full_possible_answer_count']
    actual_answer = data_point['true_answer']

    # Columns to array
    labels, weights = zip(*counts.items())
    
    # Cast weights to NumPy array and compute probabilities
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()  # Convert to a vector of probabilities

    # Start trial and keep trial run to total check if most common (majority) vote is the same as the real
    trial_accuracies = 0
    for _ in range(number_of_trials):
        # Simulate the draw for 'k' individuals
        simulations = np.random.choice(labels, k, p=weights)
        
        # Get the most common choice (majority vote) from the choice
        trial_most_common, _ = max(Counter(simulations).items(), key=lambda items: items[1])
        
        # If most common vote is the same as the real outcome, success
        if trial_most_common == actual_answer:
            trial_accuracies += 1
    
    # Convert the sum of wins to a rate of success over the period
    return trial_accuracies / number_of_trials

def simulate_answer_with_filter(data_point, k=100, number_of_trials=5000):
    z = np.array(ast.literal_eval(data_point['z']))
    w = np.array(ast.literal_eval(data_point['w']))
    #print("z lens",len(z))
    results = 0
    for _ in range(number_of_trials):
        # Sample n indexes
        sampled_indexes = np.random.choice(len(z), size=int(k/2), replace=True)
        z_sampled = z[sampled_indexes]
        w_sampled = w[sampled_indexes]
        #print(type(w_sampled))

        # Calculate common_after_filter
        unique_values = np.unique(z_sampled)
        weighted_counts = {val: np.sum(w_sampled[z_sampled == val]) for val in unique_values}
        #print("w counts::",weighted_counts)
        if(len(weighted_counts)>0):
            common_after_filter = max(weighted_counts, key=weighted_counts.get)
        else:
            counter = Counter(z_sampled)
            # Find the most common element
            print(counter,z_sampled)
            most_common_element = counter.most_common(1)[0]  # Returns the most common element and its count
            common_after_filter = most_common_element[0]
        # Check if the common_after_filter is the true answer
        results += (common_after_filter == data_point['true_answer'])
          
    results /= number_of_trials
    return results

# Assuming `df` is your DataFrame and each row can be treated as a record for the simulation

def average_simulation_results(df, k=100, number_of_trials=5000):
    # Adapting `simulate_answer` to work with a single row from a DataFrame
    # and returning the processed data frame column by row.
    def apply_simulation(row):
        return simulate_answer(row, k=k, number_of_trials=number_of_trials)
    
    # The .apply() function traverses through the data frame, and per the code syntax,
    # aligns each data point properly into a series feed.
    simulation_results = df.apply(apply_simulation, axis=1)
    
    # Let's compute the mean/average.
    return simulation_results.mean()

# Apply this to your DataFrame `df` as:
# average_result = average_simulation_results(df, k=100, number_of_trials=5000)
# print(f"The average of simulation is: {average_result}")


def loss_function(params, K, G_actual,G):
    a_1, a_2, a_3 = params
    G_predicted = [G(k, a_1, a_2, a_3) for k in K]
    G_actual = np.array(G_actual)
    G_predicted = np.array(G_predicted)
    #print(G_actual)
    return np.sum((G_actual - G_predicted)**2)

# Function to fit the parameters a_1, a_2, a_3
def fit_parameters(K, G_actual,G):
    # Initial guess for the parameters
    initial_guess = [1, 1, 1]
    
    parameter_bounds = [(1e-6, None), (1e-6, None), (None, None)]

    # Minimize the loss function
    result = minimize(loss_function, initial_guess, args=(K, G_actual,G),bounds=parameter_bounds)
    
    # Extract the fitted parameters
    a_1_fitted, a_2_fitted, a_3_fitted = result.x
    return a_1_fitted, a_2_fitted, a_3_fitted


def simulate_answer_batch(df, k, number_of_trials=5000, use_filter=False):
    n = k
    samples = number_of_trials
    # Process the answer counts into a usable form for numpy operations
    answer_counts = eval(df['full_possible_answer_count']).apply(ast.literal_eval)
    
    # Create arrays for keys and probabilities, and normalize the probabilities
    keys = answer_counts.apply(lambda x: np.array(list(x.keys())))
    probabilities = answer_counts.apply(lambda x: np.array(list(x.values()), dtype=float))
    probabilities = probabilities.apply(lambda p: p / p.sum())

    results = np.zeros(len(df))

    for i, (k, p) in enumerate(zip(keys, probabilities)):
        #print(i,k,p)
        if use_filter:
            z = np.array(ast.literal_eval(df['z'].iloc[i]))
            w = np.array(ast.literal_eval(df['w'].iloc[i]))
            #print("z lens",len(z))
            for _ in range(samples):
                # Sample n indexes
                sampled_indexes = np.random.choice(len(z), size=int(n/2), replace=True)
                z_sampled = z[sampled_indexes]
                w_sampled = w[sampled_indexes]
                #print(type(w_sampled))

                #w_sampled = np.ones_like(z_sampled,dtype=float)  # Set each element in w_sampled to 1
                #print(w_sampled)
                #print(type(w_sampled))
                # Calculate common_after_filter
                unique_values = np.unique(z_sampled)
                weighted_counts = {val: np.sum(w_sampled[z_sampled == val]) for val in unique_values}
                #print("w counts::",weighted_counts)
                common_after_filter = max(weighted_counts, key=weighted_counts.get)

                # Check if the common_after_filter is the true answer
                results[i] += (common_after_filter == df.iloc[i]['true_answer'])
        else:
            # Simulate 'samples' number of trials at once
            sampled_matrix = np.random.choice(k, size=(samples, n), p=p)

            # Determine the most common answer for each simulation
            for j in range(samples):
                most_common_answer, _ = Counter(sampled_matrix[j]).most_common(1)[0]
                results[i] += (most_common_answer == df.iloc[i]['true_answer'])
    
    # Compute average accuracy per row, then mean over all rows
    results /= samples
    return results.mean()
    
def log_scale_integers(a, b, num):
    """
    Generates a list of integers evenly distributed on a logarithmic scale.
    
    Parameters:
    a (int): The starting integer (must be > 0).
    b (int): The ending integer (must be >= a).
    num (int): The number of integers to generate.
    
    Returns:
    list: A list of integers.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Both a and b must be positive integers")
    if b < a:
        raise ValueError("b must be greater than or equal to a")

    # Generate numbers spaced evenly on a log scale
    log_nums = np.logspace(np.log10(a), np.log10(b), num=num)
    
    # Convert to integers and ensure unique values by using set, then sort the list
    int_nums = sorted(set(np.round(log_nums).astype(int)))
    
    return int_nums


def plot_answer_distribution_pie(answer_set, true_answer, show_name=True):
    """
    Plots a pie chart of the answer distribution, highlights the correct answer, and adds edge colors to each slice.
    
    Parameters:
    answer_set (dict): Dictionary of answers with their probabilities.
    true_answer (str): The key in answer_set that corresponds to the true answer.
    show_name (bool): If True, labels and percentages are displayed on the chart; if False, no text is shown.
    """
    labels = list(answer_set.keys())
    values = list(answer_set.values())
    
    # Determine which answer is correct to 'pull' it in the pie chart
    pull = [0.1 if label == true_answer else 0 for label in labels]
    
    # Set colors to distinguish all answers
    base_colors = ['lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
    colors = [base_colors[i] if label != true_answer else 'green' for i, label in enumerate(labels)]
    
    # Decide what information to display based on show_name
    textinfo = 'label+percent' if show_name else 'none'
    
    # Create the pie chart with edge colors
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=pull, 
                                 marker=dict(colors=colors, line=dict(color='black', width=2)),
                                 textinfo=textinfo)])
    
    fig.update_traces(hoverinfo='label+percent', textfont_size=20)
    fig.update_layout(title_text=f"Answer Distribution: True Answer Highlighted '{true_answer}'",
                      title_font_size=24)
    fig.show()
    