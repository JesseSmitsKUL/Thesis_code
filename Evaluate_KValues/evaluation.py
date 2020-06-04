import plotly.graph_objects as go
from plotly.subplots import make_subplots

# file to create the plots used in the evaluation




def plotConstraints(armax,khronos, loc=''):
    keys = [str(constraint) for constraint in armax.constrains]

    #meanRootSquaredError

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=keys,
        y=armax.meanAbsError,
        name='Improved Khronos',
        marker_color='indianred',
        text=armax.meanAbsError
    ))
    fig.add_trace(go.Bar(
        x=keys,
        y=khronos.meanAbsError,
        name='Original Khronos',
        marker_color='lightsalmon',
        text=khronos.meanAbsError
    ))

    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=20,
                color="black"
            )
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.update_xaxes(title_text='Constraint' ,tickfont=dict(size=22),title_font = {"size": 22})
    fig.update_yaxes(title_text='Average Prediction Error',tickfont=dict(size=20), title_font = {"size": 22})

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_traces(texttemplate='<b>%{text:.2f}</b>', textposition='outside')
    fig.update_layout(width=1200, height=900, yaxis_type="log",barmode='group', xaxis_tickangle=-45, xaxis=dict(type='category'),uniformtext_minsize=10, uniformtext_mode='show',plot_bgcolor='rgb(255,255,255)')


    if loc == '':
        fig.show()
    else:
        fig.write_image(loc+"constraints.png")


def plotTimeouts(armax,khronos,loc=''):

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05)





    print(list(range(len(armax.timeouts[0.99]))))
    print(len(armax.predictedArrivals))
    print(len(khronos.predictedArrivals))
    print(len(armax.timeouts[0.8]))

    fig.add_trace(go.Scatter(x=list(range(len(armax.timeouts[0.95]))), y=armax.timeouts[0.99],name='Improved timeouts'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(armax.predictedArrivals))), y=armax.predictedArrivals, name='Arrivals'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(khronos.timeouts[0.95]))), y=khronos.timeouts[0.99],name='Khronos timeouts'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(khronos.predictedArrivals))), y=khronos.predictedArrivals,name='Arrivals'),
                  row=2, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.update_layout(title="Timeouts comparison",width=1400, height=900, yaxis_type="log")

    fig.update_xaxes(title_text='Packet number')
    fig.update_yaxes(title_text='Timeout (ms)',type="log")

    if loc == '':
        fig.show()
    else:
        pass
        #fig.write_image(loc+ str(key) +"/timeouts.png")


def plotTimeoutsSave(armax, khronos, loc=''):


    for key in armax.constrains:

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05)


        fig.add_trace(go.Scatter(x=list(range(len(armax.timeouts[key]))), y=armax.timeouts[key], name='Improved timeouts'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(armax.predictedArrivals))), y=armax.predictedArrivals, name='Arrivals'),
                      row=1, col=1)

        fig.add_trace(
            go.Scatter(x=list(range(len(khronos.timeouts[key]))), y=khronos.timeouts[key], name='Khronos timeouts'),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=list(range(len(khronos.predictedArrivals))), y=khronos.predictedArrivals, name='Arrivals'),
            row=2, col=1)

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

        fig.update_layout(title="Timeouts comparison", width=1400, height=900, yaxis_type="log")

        fig.update_xaxes(title_text='Packet number')
        fig.update_yaxes(title_text='Timeout (ms)',type="log")

        if loc == '':
            fig.show()
        else:
            fig.write_image(loc + str(key) + "/timeouts.png")



############################


def plotModel(model,key,loc=''):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(model.timeouts[key]))), y=model.timeouts[key],name='Timeouts'))
    fig.add_trace(go.Scatter(x=list(range(len(model.predictedArrivals))), y=model.predictedArrivals, name='Arrivals'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.update_layout(title="Timeouts comparison",width=1500, height=500, yaxis_type="log")

    fig.update_xaxes(title_text='Packet number')
    fig.update_yaxes(title_text='Timeout (ms)',type="log")

    if loc == '':
        fig.show()
    else:
        pass
        #fig.write_image(loc+ str(key) +"/timeouts.png")

def plotTwoModel(model,khronos,key,loc=''):
    fig = go.Figure()

    # mode='lines+markers',

    fig.add_trace(go.Scatter(x=list(range(len(model.timeouts[key]))), y=model.timeouts[key],name='Improved timeouts',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=list(range(len(khronos.timeouts[key]))), y=khronos.timeouts[key], name='Khronos Timeouts',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=list(range(len(model.predictedArrivals))), y=model.predictedArrivals, name='Arrivals',mode='lines+markers'))



    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.update_layout(title="Timeouts comparison",width=1000, height=400, plot_bgcolor='rgb(255,255,255)', yaxis_type="log")

    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            )
        )
    )

    fig.update_xaxes(title_text='Packet number')
    fig.update_yaxes(title_text='Timeout (ms)',type="log")

    if loc == '':
        fig.show()
    else:
        fig.write_image(loc+ str(key) +"/comparison.png")

def plotTwoModelRun(model, khronos, loc=''):

    for key in model.constrains:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=list(range(len(model.timeouts[key]))), y=model.timeouts[key], name='Improved timeouts'))
        fig.add_trace(
            go.Scatter(x=list(range(len(khronos.timeouts[key]))), y=khronos.timeouts[key], name='Khronos Timeouts'))
        fig.add_trace(
            go.Scatter(x=list(range(len(model.predictedArrivals))), y=model.predictedArrivals, name='Arrivals'))


        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

        fig.update_layout(width=1000, height=400, plot_bgcolor='rgb(255,255,255)',
                          yaxis_type="log")

        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                )
            )
        )

        fig.update_xaxes(title_text='Packet number')
        fig.update_yaxes(title_text='Timeout (ms)', type="log")

        if loc == '':
            fig.show()
        else:
            fig.write_image(loc + str(key) + "/comparison.png")


def plotTwomodelScenarioRun(armax, khronos, loc=''):


    for key in armax.constrains:

        fig = make_subplots(rows=2, cols=1,
                            vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=list(range(len(armax.scenario))), y=armax.scenario, name='Scenario'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(armax.timeouts[key]))), y=armax.timeouts[key], name='Improved timeouts'),
                      row=2, col=1)
        fig.add_trace(
            go.Scatter(x=list(range(len(khronos.timeouts[key]))), y=khronos.timeouts[key], name='Khronos timeouts'),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=list(range(len(khronos.predictedArrivals))), y=khronos.predictedArrivals, name='Arrivals'),
            row=2, col=1)


        fig.update_layout(width=1000, height=600, yaxis_type="log",plot_bgcolor='rgb(255,255,255)')

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

        fig.update_layout(
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
                )
            )
        )


        fig.update_xaxes(title_text='Packet number')
        fig.update_yaxes(title_text='Timeout (ms)',type="log", row=2, col=1)
        fig.update_yaxes(title_text=armax.sname, row=1, col=1)

        if loc == '':
            fig.show()
        else:
            fig.write_image(loc + str(key) + "/timeouts.png")



def plotdata(data, loc = ''):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(data))), y=data,name='Arrival times'))
    #fig.add_trace(go.Scatter(x=list(range(len(model.predictedArrivals))), y=model.predictedArrivals, name='Arrivals'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

    fig.update_layout(title="Timeouts comparison",width=1500, height=500, yaxis_type="log")

    fig.update_xaxes(title_text='Packet number')
    fig.update_yaxes(title_text='Relative arrival times') #,type="log"

    if loc == '':
        fig.show()
    else:
        pass
        #fig.write_image(loc+ str(key) +"/timeouts.png")