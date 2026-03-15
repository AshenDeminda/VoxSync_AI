import gradio as gr
from src.engines.stt_engine import STTEngine
from src.engines.llm_engine import LLMEngine
from src.engines.tts_engine import TTSEngine

print("Loading Engines... (Models will be loaded from D:\\VoxSync_Cache)")
stt = STTEngine()
llm = LLMEngine()
tts = TTSEngine()
print("All engines loaded successfully!")

def process_conversation(mic_audio, ref_audio, history):
    if not mic_audio or not ref_audio:
        return history, None, "Please provide both microphone and reference audio."
    
    # 1. Speech to Text
    user_text = stt.transcribe(mic_audio)
    if not user_text:
        return history, None, "Could not hear you. Try again."
        
    # 2. LLM response
    ai_text = llm.chat(user_text)
    
    # 3. Text to Speech (Voice Cloning)
    output_audio = tts.generate(ai_text, ref_audio)
    
    history.append((user_text, ai_text))
    return history, output_audio, "Success!"

with gr.Blocks(title="VoxSync AI") as ui:
    gr.Markdown("# 🎙️ VoxSync AI")
    
    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(label="1. Upload Reference Voice (10s)", type="filepath")
            mic_audio = gr.Audio(label="2. Record Message", type="filepath", sources=["microphone"])
            submit_btn = gr.Button("Send to AI", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation")
            output_audio = gr.Audio(label="AI Cloned Response", autoplay=True)

    submit_btn.click(
        fn=process_conversation,
        inputs=[mic_audio, ref_audio, chatbot],
        outputs=[chatbot, output_audio, status]
    )

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1", server_port=7860)