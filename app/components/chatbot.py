import streamlit as st

from services.rag import answer_lab_question


def render_chatbot():
    st.header("💬 Laboratory Chat")

    st.caption(
        "Ask questions about information stored "
        "in the laboratory notebook."
    )

    if "lab_chat_messages" not in st.session_state:
        st.session_state["lab_chat_messages"] = []

    for message in st.session_state["lab_chat_messages"]:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask the laboratory notebook..."
    )

    if not question:
        return

    st.session_state[
        "lab_chat_messages"
    ].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching laboratory notebook..."
        ):
            try:
                result = answer_lab_question(
                    question
                )

            except Exception as error:
                st.error(
                    f"Could not search the "
                    f"laboratory notebook: {error}"
                )
                return

        answer = result["answer"]

        st.markdown(answer)

        sources = result["sources"]

        if sources:
            with st.expander("Sources"):
                for source in sources:
                    experiment_id = source.get("id")
                    title = (
                        source.get("title")
                        or "Untitled experiment"
                    )
                    date = source.get("date") or ""

                    st.markdown(
                        f"**Experiment {experiment_id}: "
                        f"{title}**  \n"
                        f"{date}"
                    )

    st.session_state[
        "lab_chat_messages"
    ].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )