import json
import asyncio

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from websockets import connect
import openai
import datetime

st.set_page_config(layout="wide")
import os

openai.api_key = os.environ["OPENAI_API_KEY"]
waves = {
    "delta": [0.1, 4],
    "theta": [4, 7.5],
    "alpha": [7.5, 12.5],
    "beta": [12.5, 30],
    "gamma": [30, 100],
}

figs = {}
pcas = {}
data = {"delta": [], "theta": [], "alpha": [], "beta": [], "gamma": []}
col1, col2, col3, col4, col5 = st.columns(5)
col6 = st.columns(1)[0]

empty = {
    "delta": col1.empty(),
    "theta": col2.empty(),
    "alpha": col3.empty(),
    "beta": col4.empty(),
    "gamma": col5.empty(),
}

poem = col6.empty()

for name, band in waves.items():
    figs[name] = go.Figure()
    figs[name].update_layout(title=f"{name.capitalize()} Waves")

    pcas[name] = IncrementalPCA(n_components=3)

batch_size = 100


async def print_messages():
    global data
    history = {}
    async with connect("ws://localhost:8080") as ws:
        while True:
            msg = await ws.recv()
            # Extract theta
            new_data = json.loads(msg)["data"]
            for name in waves:
                data[name].append(new_data[name])

            if len(data["alpha"]) >= batch_size:
                results = {}

                for name in waves:
                    # Extract theta
                    X = np.array(data[name])

                    # Reshape
                    X = X.reshape(len(data[name]), -1)

                    # Update PCA
                    pcas[name].partial_fit(X)
                    data[name] = []
                    results[name] = {
                        "components": pcas[name].components_[:3],
                        "explained_variance_ratio": pcas[
                            name
                        ].explained_variance_ratio_[:3],
                        "mean": np.mean(pcas[name].mean_),
                        "transform": pcas[name].transform(X),
                    }
                    # Create dataframe
                    df = pd.DataFrame(
                        pcas[name].transform(X), columns=["PC1", "PC2", "PC3"]
                    )

                    # Update figure
                    figs[name].add_trace(
                        go.Scatter3d(
                            x=df["PC1"], y=df["PC2"], z=df["PC3"], mode="markers"
                        )
                    )

                for name in empty:
                    empty[name].plotly_chart(figs[name], use_container_width=True)

                # Record new means
                time = datetime.datetime.now().strftime("%I:%M:%S %p")
                for name, pca_data in results.items():
                    history.setdefault(name, []).append((time, pca_data))

                    prompt = f"""This is the history of each brainwave band over the last 50 samples, processed with PCA:"""

                def format_pcs(pcs, variances, mean):
                    prompt = ""
                    print(mean)
                    for i in range(3):
                        prompt += f"- PC{i+1} explains {variances[i]:.0%} of variance\n"
                    # mean
                    prompt += f"- Mean: {mean:.2f}\n"
                    return prompt

                # if history is too big, delete a few old samples
                if len(history["alpha"]) > 10:
                    for name, pca_data in history.items():
                        history[name] = pca_data[-10:]

                for name, pca_data in history.items():
                    prompt += f"\n{name.upper()}:\n"
                    for time, d in pca_data:
                        prompt += f"{time}: \n"
                        prompt += format_pcs(
                            d["components"], d["explained_variance_ratio"], d["mean"]
                        )

                print(history)
                print("~" * 50)
                print(prompt)

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {
                            "role": "system",
                            "content": """Louis is computing PCA on his brain waves from a EEG brain computer interface.

FYI, this is a cheatsheet to interpret brain waves:

| Brainwave Frequency | Associated Characteristics |
|---------------------|---------------------------|
| Delta  | Deep relaxation, sleep, healing, unconscious mind |
| Theta    | Creativity, intuition, memory, learning, sleep, healing |
| Alpha   | Relaxation, meditation, memory, learning, creativity, reducing anxiety |
| Beta   | Conscious thought, mental activity, anxiety/excitement, movement, self-control, REM sleep |
| Gamma | Cognitive functioning, concentration/focus, creativity, positive mood states, increased brain activity, neurostimulation/meditation |

You write short poems inspired by Louis' brainwave PCA results.
You use Saadi Shirazi's poetry style.
Do not mention PCA, brainwaves, etc. Use nature as playground.""",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                )
                poem.markdown(completion.choices[0].message.content)


asyncio.run(print_messages())
