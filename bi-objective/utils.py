import plotly.express as px


# Helper function to plot 3D distribution of data points in the process parameters space
def plot3d_scatter(df, show_colorbar=True, width=800, height=600):
    fig = px.scatter_3d(df, x='P (W)', y='v (mm/s)', z='h (mu m)', color='round',
                        labels={'P (W)': 'Power (W)', 'v (mm/s)': 'Speed (mm/s)', 'h (mu m)': 'Hatching (μm)'},
                        )
    fig.update_traces(marker=dict(size=3))  

    # Compute tick values and round them to the nearest integer
    x_tickvals = [round(df['P (W)'].min()), round((df['P (W)'].min() + df['P (W)'].max()) / 2), round(df['P (W)'].max())]
    y_tickvals = [round(df['v (mm/s)'].min()), round((df['v (mm/s)'].min() + df['v (mm/s)'].max()) / 2), round(df['v (mm/s)'].max())]
    z_tickvals = [round(df['h (mu m)'].min()), round((df['h (mu m)'].min() + df['h (mu m)'].max()) / 2), round(df['h (mu m)'].max())]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=x_tickvals,
                ticktext=[str(val) for val in x_tickvals]
            ),
            yaxis=dict(
                tickvals=y_tickvals,
                ticktext=[str(val) for val in y_tickvals]
            ),
            zaxis=dict(
                tickvals=z_tickvals,
                ticktext=[str(val) for val in z_tickvals]
            )
        ),
        width=width,
        height=height
    )

    if not show_colorbar:
        fig.update_coloraxes(showscale=False)

    fig.show()



