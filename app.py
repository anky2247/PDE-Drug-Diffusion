import streamlit as st
import numpy as np
import plotly.graph_objects as go

from pde_solver import simulate


st.set_page_config(
    page_title="Drug Diffusion Simulator",
    page_icon="🧪",
    layout="wide"
)

st.title("2D Drug Diffusion Simulator")

st.write(
    "Explore how concentration changes over time "
    "using a diffusion-reaction PDE."
)

st.latex(
    r"\frac{\partial C}{\partial t}"
    r"=D\left(\frac{\partial^2 C}{\partial x^2}"
    r"+\frac{\partial^2 C}{\partial y^2}\right)-kC"
)


# --------------------------------
# SIDEBAR PARAMETERS
# --------------------------------

st.sidebar.header("Simulation parameters")

N = st.sidebar.select_slider(
    "Grid resolution",
    options=[51, 101, 151, 201],
    value=101
)

D = st.sidebar.number_input(
    "Diffusion coefficient D",
    min_value=1e-11,
    max_value=3e-10,
    value=1e-10,
    format="%.2e"
)

k = st.sidebar.slider(
    "Degradation rate k",
    min_value=0.0001,
    max_value=0.0020,
    value=1 / 1800,
    step=0.0001
)

sigma = st.sidebar.slider(
    "Initial spread (sigma)",
    min_value=0.0002,
    max_value=0.0010,
    value=0.0005,
    step=0.00005
)

total_time = st.sidebar.slider(
    "Total simulation time (seconds)",
    min_value=300,
    max_value=3600,
    value=1800,
    step=300
)


# --------------------------------
# RUN SIMULATION
# --------------------------------

@st.cache_data(show_spinner=False)
def run_model(N, D, k, sigma, total_time):

    return simulate(
        N=N,
        D=D,
        k=k,
        sigma=sigma,
        total_time=total_time,
        num_frames=41
    )


with st.spinner("Running diffusion simulation..."):

    result = run_model(N, D, k, sigma, total_time)


x = result["x"] * 1000
y = result["y"] * 1000

times = result["times"]
frames = result["frames"]

max_concentration = float(frames[0].max())


# --------------------------------
# METRICS
# --------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Grid", f"{N} × {N}")
col2.metric("Time steps", result["steps"])
col3.metric("Final maximum", f"{frames[-1].max():.5f}")


# --------------------------------
# ANIMATED HEATMAP
# --------------------------------

st.subheader("Concentration animation")

plot_frames = []

for i, concentration in enumerate(frames):

    plot_frames.append(
        go.Frame(
            data=[
                go.Heatmap(
                    z=concentration,
                    x=x,
                    y=y,
                    zmin=0,
                    zmax=max_concentration,
                    colorscale="Inferno",
                    colorbar=dict(title="Concentration")
                )
            ],
            name=str(i)
        )
    )


fig = go.Figure(
    data=[
        go.Heatmap(
            z=frames[0],
            x=x,
            y=y,
            zmin=0,
            zmax=max_concentration,
            colorscale="Inferno",
            colorbar=dict(title="Concentration")
        )
    ],
    frames=plot_frames
)

slider_steps = []

for i, t in enumerate(times):

    slider_steps.append(
        dict(
            method="animate",
            label=f"{t:.0f} s",
            args=[
                [str(i)],
                {
                    "mode": "immediate",
                    "frame": {
                        "duration": 0,
                        "redraw": True
                    }
                }
            ]
        )
    )


fig.update_layout(
    title="2D Drug Concentration",
    xaxis_title="X position (mm)",
    yaxis_title="Y position (mm)",
    height=600,
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {
                                "duration": 120,
                                "redraw": True
                            },
                            "fromcurrent": True
                        }
                    ]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": False
                            },
                            "mode": "immediate"
                        }
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            active=0,
            steps=slider_steps,
            currentvalue={"prefix": "Time: "}
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------
# CONCENTRATION GRAPH
# --------------------------------

st.subheader("Concentration at the centre")

center_fig = go.Figure()

center_fig.add_trace(
    go.Scatter(
        x=times,
        y=result["center_values"],
        mode="lines",
        name="Centre concentration"
    )
)

center_fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Concentration",
    height=350
)

st.plotly_chart(center_fig, use_container_width=True)


# --------------------------------
# TOTAL CONCENTRATION GRAPH
# --------------------------------

st.subheader("Total concentration in the domain")

mass_fig = go.Figure()

mass_fig.add_trace(
    go.Scatter(
        x=times,
        y=result["total_mass"],
        mode="lines",
        name="Integrated concentration"
    )
)

mass_fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Integrated concentration",
    height=350
)

st.plotly_chart(mass_fig, use_container_width=True)


st.success("Simulation completed.")

st.caption(
    "Educational mathematical model only. "
    "This is not a validated medical or clinical drug-delivery model."
)