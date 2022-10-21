import requests
import json
import argparse

import streamlit as st


def main():
    parser = argparse.ArgumentParser("Streamlit Server")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--displayed_model_name", default="Language Model")
    args = parser.parse_args()

    st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title=f"{args.displayed_model_name} Playground",  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )
    st.title(f"{args.displayed_model_name} Playground")
    """This app enables you to interact with large language models in a friendly way!"""

    ex_names = [
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        "The ancient people of Arcadia achieved oustanding cultural and technological developments. Below we summarise some of the highlights of the Acadian society.",
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: My day has been 👍.
        Sentiment: Positive
        ###
        Tweet: This is the link to the article.
        Sentiment: Neutral
        ###
        Tweet: This new movie started strange but in the end it was awesome.
        Sentiment:""",
                """Q: Fetch the departments that have less than five people in it.\n
        A: SELECT DEPARTMENT, COUNT(WOKRED_ID) as "Number of Workers" FROM Worker GROUP BY DEPARTMENT HAVING COUNT(WORKED_ID) < 5;\n
        ###\n
        Q: Show all departments along with the number of people in each department\n
        A: SELECT DEPARTMENT, COUNT(DEPARTMENT) as "Number of Workers" FROM Worker GROUP BY DEPARTMENT;\n
        ###\n
        Q: Show the last record of the Worker table\n
        A: SELECT * FROM Worker ORDER BY LAST_NAME DESC LIMIT 1;\n
        ###\n
        Q: Fetch the three max salaries from the Worker table;\n
        A:""",
        "白日依山尽",
    ]
    example = st.selectbox("Choose an example prompt from this selector", ex_names)

    inp = st.text_area(
        "Or write your own prompt here!", example, max_chars=10000, height=150
    )

    try:
        rec = ex_names.index(inp)
    except ValueError:
        rec = 0

    with st.expander("Generation options..."):
        max_new_tokens = st.slider(
            "Max new tokens of the generated texts (in tokens)",
            min_value=1,
            max_value=128,
            value=10,
            step=1,
        )
        n = st.slider(
            "N",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        do_sample = st.checkbox("Check in to do sample in generate", True)
        temp = st.slider(
            "Choose the temperature (higher - more random, lower - more repetitive). For the code generation or sentence classification promps it's recommended to use a lower value, like 0.35",
            min_value=0.,
            max_value=1.5,
            value=1.0 if rec < 2 else 0.35,
        )
        top_p = st.slider(
            "Choose the top-p of the generated texts",
            min_value=0.,
            max_value=1.,
            value=0.95,
        )

    response = None
    with st.form(key="inputs"):
        submit_button = st.form_submit_button(label="Generate!")

        if submit_button:
            url = args.backend
            payload = {
                "context": inp,
                "max_new_tokens": max_new_tokens,
                "temperature": temp,
                "top_p": top_p,
                "do_sample": do_sample,
                "num_beams": n,
                "num_return_sequences": n,
            }

            query = requests.request("POST", url, json=payload)
            try:
                response = query.json()
                output = response["output"]

                if isinstance(output, dict):
                    output = output["sequences"]

                st.write(output)
                st.text(f"Generation done in {response['compute_time']:.3} s.")
            except Exception as exception:
                st.error(f"Request failed, status_code={query.status_code}, reason={query.reason}, content={query.content}")


if __name__ == "__main__":
    main()
