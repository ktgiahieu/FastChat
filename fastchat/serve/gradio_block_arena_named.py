"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import re
import os
import json
import time
import fitz
import openai
import numpy as np
import gradio as gr
import pandas as pd
from fpdf import FPDF
import concurrent.futures

from fastchat.serve.checklist.checklist_constants import checklist_data
from fastchat.serve.checklist.constants import prompt as PAPER_PROMPT

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

model_states = [gr.State(None) for _ in range(2)]

def get_api_config(model_name, config_path="api_endpoint.json"):
    with open(config_path, "r") as f:
        api_configs = json.load(f)
    if model_name not in api_configs:
        raise ValueError(f"Model {model_name} not found in {config_path}")
    return api_configs[model_name]

def parse_checklist(checklist, checklist_data):
    general_guidelines = """If you answer Yes to a question, in the justification please point to the section(s) where related material for the question can be found.
While "Yes" is generally preferable to "No", it is perfectly acceptable to answer "No" provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used").
"""
    checklist_df = pd.DataFrame(columns=['Question', 'Question_Title', 'Answer', 'Justification', 'Guidelines', 'Review', 'Score', 'LLM'])
    try:
        for question_index, checklist_item in enumerate(checklist_data):

            question_title = checklist_item["question_title"]
            question = checklist_item["question"]
            question_guidelines = checklist_item["guidelines"]

            question_regex = re.escape(question)
            pattern = re.compile(rf"Question:\s*{question_regex}(?:.*?Answer:\s*\[(.*?)\].*?Justification:\s*(.*?))(?:Guidelines:\s+(.*?))(?=Question:|\Z)", re.DOTALL)

            mtch = pattern.search(checklist)
            if mtch:
                answer = mtch.group(1).strip()
                justification = mtch.group(2).strip() if mtch.group(2).strip() else None

                if justification is not None and justification.isdigit():
                    justification = None

            else:
                answer, justification = "Not Found", "Not Found"

            temp_df = pd.DataFrame([
                {
                    'Question': question,
                    'Question_Title': question_title,
                    'Answer': answer,
                    'Justification': justification,
                    'Guidelines': general_guidelines + question_guidelines}
            ])

            checklist_df = pd.concat([checklist_df, temp_df], ignore_index=True)
        return checklist_df

    except Exception as e:
        raise ValueError(f"[-] Error in extracting answers and justifications: {e}")

def check_incomplete_questions(checklist_df):
    count_not_found = 0
    for i, row in checklist_df.iterrows():

        if row["Answer"] in ["TODO", "[TODO]"] or row["Justification"] in ["TODO", "[TODO]"] or row["Justification"] is None:
            print(f"[!] You haven't filled the answer or justificaiton for Question #: {i+1}")

        if row["Answer"] == "Not Found" or row["Justification"] == "Not Found":
            count_not_found += 1
            print(f"[!] There seems to be a problem with your answer or justificaiton for Question #: {i+1}. Please make sure that\n - you haven't changed the question in the checklist\n - you haven't removed the guidelines prodived for each question")

    if count_not_found == 15:
        raise ValueError("[-] All your answers or justifications are not found!. Please check that you have filled the checklist properly. If the problem is still there, please contact the organizers.")

def has_line_numbers(text):
    lines = text.split('\n')
    lines_to_iterate = len(lines) if len(lines) < 2000 else 2000
    numbered_lines = []
    for i in range(0, lines_to_iterate):
        numbered_lines.append(lines[i].strip().isdigit())
    if sum(numbered_lines) > int(lines_to_iterate * 0.10):
        return True
    return False

def clean_title(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\-\s*\n', ' ', text)
    text = text.strip()
    return text

def clean_paper(text):
    paper_has_line_numbers = has_line_numbers(text)
    if paper_has_line_numbers:
        text = re.sub(r'^(\d+)\n(\d+(\.\d+)?)\n(.+)$', r'\2 \4', text, flags=re.MULTILINE)
    else:
        text = re.sub(r'^(\d+)\n(.+)$', r'\1 \2', text, flags=re.MULTILINE)
    text = re.sub(r'\n\d+\n', r'\n', text)
    text = re.sub(r'\n\-\n', r'\n', text)
    text = re.sub(r'\-\s*\n', '', text)
    text = text.replace("’", "'")
    text = text.replace("\\'", "'")
    text = text.replace("- ", "")
    processed_text = ""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line.split()) < 6 and len(line.split()) > 1:
            processed_text += "\n"
            processed_text += line + "\n"
        else:
            processed_text += line
            processed_text += ' '
    text = processed_text.strip()
    return text
    
def clean_checklist(text):
    text = re.sub(r'\n\d+', ' ', text)
    text = re.sub(r'\-\s*\n', '', text)
    text = re.sub(r'  . ', '\n', text)
    text = re.sub(r'([a-zA-Z]\.\d+)\n', r'\1 ', text)
    text = re.sub(r'([a-zA-Z])\n', r'\1 ', text)
    text = text.replace("’", "'")
    text = text.replace("\\'", "'")
    text = text.replace("- ", "")
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("https://neurips.cc/public/ EthicsGuidelines", "https://neurips.cc/public/EthicsGuidelines")
    text = text.strip()
    return text

def clean(paper):
    paper["title"] = clean_title(paper["title"])
    paper["paper"] = clean_paper(paper["paper"])
    paper["checklist"] = clean_checklist(paper["checklist"])
    return paper

def get_paper_chunks(paper_text):
    try: 
        paper_end_index = paper_text.find("NeurIPS Paper Checklist")

        if paper_end_index == -1:
            raise ValueError("[-] Error: NeurIPS Paper Checklist not found. Please make sure that the checklist is at the end of the PDF.")

        paper = paper_text[:paper_end_index]

        total_allowed_words = 15000  # 400 per page (15 main pages, additional 20 pages)
        paper_words = paper.split()

        if len(paper_words) > total_allowed_words:
            paper_lines = paper.split('\n')
            paper_lines_to_keep = []
            word_count = 0
            for line in paper_lines:
                line_words = line.split()
                if word_count + len(line_words) <= total_allowed_words:
                    paper_lines_to_keep.append(line)
                    word_count += len(line_words)
                else:
                    break

            paper = '\n'.join(paper_lines_to_keep)
            end_of_words_to_remove = ' '.join(paper.split()[-20:])
            print(f"[!] The paper is too long! Text after \"{end_of_words_to_remove}\" is removed and will not be considered for the LLM review.")

        checklist_start_index = paper_end_index
        checklist = paper_text[checklist_start_index:]

        title_end_index = paper.find("Anonymous Author")
        if title_end_index == -1:
            title = paper.split("\n")[:2]
            title = ''.join(title)
        else:
            title = paper[:title_end_index]

        return {
            "title": title,
            "paper": paper,
            "checklist": checklist
        }
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"[-] Error occurred while extracting paper chunks in the {'paper' if not paper else 'checklist'} section: {e}")

def get_single_question_LLM_feedback(paper_prompt, model_name):
    config = get_api_config(model_name)

    api_base = config["api_base"]
    api_key = config["api_key"]
    api_version = config["azure_api_version"]
    temperature = config["recommended_config"]["temperature"]
    top_p = config["recommended_config"]["top_p"]

    openai.api_type = "azure"
    openai.api_base = api_base
    openai.api_version = api_version
    openai.api_key = api_key

    max_tokens = 1000
    n = 1
    number_of_times_question_processed = 0
    review_is_empty = True
    error_in_processing_question = False

    score = -1
    llm_review = ""

    while (review_is_empty or error_in_processing_question) and number_of_times_question_processed < 3:
        number_of_times_question_processed += 1

        if number_of_times_question_processed != 1:
            print("[!] Reprocessing this question!")

        try:
            user_prompt = {
                "role": "user",
                "content": paper_prompt
            }
            messages = [user_prompt]
            chat_completion = openai.ChatCompletion.create(
                engine=model_name,  
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n
            )
            llm_review = chat_completion["choices"][0]["message"]["content"]
            error_in_processing_question = False
        except Exception as e:
            error_in_processing_question = True
            llm_review = f"Error: LLM failed to process this question!\n{str(e)}"
            score = -1
            print(f"[-] {llm_review}")
            continue

    return llm_review, score, "GPT"

def get_LLM_feedback(paper, checklist_df, model):

    for index, row in checklist_df.iterrows():

        question_number = index + 1

        print(f"[*] Processing Question # {question_number}")

        q = row["Question"]
        a = row["Answer"]
        j = row["Justification"]
        g = row["Guidelines"]

        if a == "Not Found" or j == "Not Found":
            print(f"[!] Skipping Question # {question_number}. Answer or Justification for this question was not found!")
            checklist_df.loc[index, 'Review'] = "Answer or Justification for this question was not found!"
            checklist_df.loc[index, 'Score'] = 0
            continue

        paper_prompt = PAPER_PROMPT

        paper_prompt = paper_prompt.replace("{q}", q)
        paper_prompt = paper_prompt.replace("{a}", a)
        if j is None:
            j = ""
        paper_prompt = paper_prompt.replace("{j}", j)
        paper_prompt = paper_prompt.replace("{g}", g)
        paper_prompt = paper_prompt.replace("{paper}", paper)
        question_score, question_review, llm = get_single_question_LLM_feedback(paper_prompt, model)

        checklist_df.loc[index, 'Review'] = question_review
        checklist_df.loc[index, 'Score'] = question_score
        checklist_df.loc[index, 'LLM'] = llm

        print(f"[+] Question # {question_number}")

    return checklist_df


def save_uploaded_file_as_pdf(uploaded_file): 
    global model_states
    
    if uploaded_file is None:
        return "No file uploaded."

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(uploaded_file, "name"):
        file_name = os.path.basename(uploaded_file.name)
    elif isinstance(uploaded_file, dict) and "name" in uploaded_file:
        file_name = os.path.basename(uploaded_file["name"])
    else:
        file_name = "uploaded_file.pdf"   
    save_path = os.path.join(save_dir, file_name)

    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    elif isinstance(uploaded_file, dict):
        data = uploaded_file.get("data", b"")
        if isinstance(data, str):
            data = data.encode("utf-8")
    elif isinstance(uploaded_file, str):
        with open(uploaded_file, "rb") as f:
            data = f.read()
    else:
        return "Unsupported file format."

    with open(save_path, "wb") as f:
        f.write(data)
        
    # convert paper
    paper_text = ""
    with fitz.open(save_path) as doc:
        for page in doc:
            paper_text += page.get_text()
    if paper_text == "":
        raise ValueError("[-] Reading PDF file failed or your PDF has no text! Please check that you have a valid PDF file.")
    
    # clean paper
    paper = clean(get_paper_chunks(paper_text))
    paper["checklist_df"] = parse_checklist(paper["checklist"], checklist_data)
    check_incomplete_questions(paper["checklist_df"])
    
    # generate output 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(get_LLM_feedback, paper["paper"], paper["checklist_df"], model_states[0])
        future2 = executor.submit(get_LLM_feedback, paper["paper"], paper["checklist_df"], model_states[1])
        answer1 = future1.result()
        answer2 = future2.result()
    return str(answer1), str(answer2)


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = [None] * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 64)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = [
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    return (
        [None] * num_sides
        + [None] * num_sides
        + [""]
        + [invisible_btn] * 4
        + [disable_btn] * 2
    )


def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
    state0, state1, model_selector0, model_selector1, text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + ["", None]
            + [
                no_change_btn,
            ]
            * 6
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[1].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list)
    if flagged:
        logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 6
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                request,
            )
        )

    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen-vl-max-0809",
            "qwen1.5-110b-chat",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2.5-72b-instruct",
            "qwen2-72b-instruct",
            "qwen-plus-0828",
            "qwen-max-0919",
            "llama-3.1-405b-instruct-bf16",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 4 + [enable_btn] * 2,
        [enable_btn] * 6,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.3)


def build_side_by_side_ui_named(models):
    notice_markdown = f"""
    # ⚔️ [DEMO] LLM-Arena for Checklist Assistant

    ## 📜 How It Works
    - **Blind Test**: Ask any question to two anonymous AI chatbots.
    - **Vote for the Best**: Choose the best response.

    ## 👇 Chat now!
    """

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    file_upload = gr.File(label="Please upload your paper", file_types=[".pdf"])
    

    save_button = gr.Button(value="Upload PDF", variant="primary")
    with gr.Row():
        with gr.Column():
            output_text_left = gr.Textbox(label="Parsed Paper Output - Model 1", lines=10, interactive=True)
        with gr.Column():
            output_text_right = gr.Textbox(label="Parsed Paper Output - MOdel 2", lines=10, interactive=True)


    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_states[i] = models[i] if len(models) > i else ""
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                    
        with gr.Row():
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(models)} models", open=False
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id=f"chatbot",
                        height=650,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                    )

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
        
    save_button.click(
        save_uploaded_file_as_pdf, 
        inputs=file_upload, 
        outputs=[output_text_left, output_text_right]  
    ).then(
        add_text,
        states + model_selectors + [output_text_left, output_text_right], 
        states + chatbots + [output_text_left, output_text_right] + btn_list   
    )
    # .then(
    #     bot_response_multi,
    #     states + [temperature, top_p, max_output_tokens],
    #     states + chatbots + btn_list
    # ).then(
    #     flash_buttons, 
    #     [], 
    #     btn_list
    # )

    
    # save_button.click(
    # save_uploaded_file_as_pdf, 
    # inputs=file_upload, 
    # outputs=output_text_left
    # ).then(
    #     add_text,
    #     states + model_selectors + [output_text_left],
    #     states + chatbots + [output_text_left] + btn_list
    # ).then(
    #     bot_response_multi,
    #     states + [temperature, top_p, max_output_tokens],
    #     states + chatbots + btn_list
    # ).then(
    #     flash_buttons, 
    #     [], 
    #     btn_list
    # )


    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    send_btn.click(
        add_text,
        states + model_selectors + [textbox],
        states + chatbots + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    return states + model_selectors
