import customtkinter as ctk
from bot_engine import BotEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MAIN_FONT = "Helvetica"

class Chatbot(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.bot_engine = BotEngine()

        self.title("TARUMT FAQ Chatbot")
        self.geometry("1000x800")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)


        # ==========================================
        # LEFT PANE (Sidebar Controls)
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#212121")
        self.sidebar_frame.grid(row=0, column=0, sticky='nsew')
        self.sidebar_frame.grid_propagate(False)

        self.sidebar_label = ctk.CTkLabel(self.sidebar_frame, text="Campus Bot", font=ctk.CTkFont(family=MAIN_FONT, size=24, weight="bold"))
        self.sidebar_label.pack(pady=20, padx=20)

        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Select Model:", font=ctk.CTkFont(family=MAIN_FONT, size=13))
        self.model_label.pack(anchor="w", padx=20, pady=(10,0))

        self.model_selector = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values = ["SVM", "LSTM", "Transformer"],
            command=self.change_model_event,
            dynamic_resizing=False,
            font=ctk.CTkFont(family=MAIN_FONT, size=13)
        )
        self.model_selector.pack(padx=20, pady=10, fill="x")

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Waiting for selection...", text_color="orange", font=ctk.CTkFont(family=MAIN_FONT, size=12))
        self.status_label.pack(anchor="w", padx=20, pady=(0, 15))

        self.cap_label = ctk.CTkLabel(self.sidebar_frame, text="What can I answer?", font=ctk.CTkFont(family=MAIN_FONT, size=14, weight="bold"))
        self.cap_label.pack(anchor="w", padx=20, pady=(15, 0))

        self.info_box = ctk.CTkTextbox(self.sidebar_frame, height=140, fg_color="#2b2b2b", font=ctk.CTkFont(family=MAIN_FONT, size=12), wrap="word")
        self.info_box.pack(padx=20, pady=5, fill="x")

        capabilities_text = (
            "• Program Details & Faculties\n"
            "• Course Registration & Add/Drop\n"
            "• PTPTN Loans & Merit Scholarships\n"
            "• Tuition Fees & Installment Plans\n"
            "• New Enrollment & Orientation\n"
            "• Campus Facilities & Library\n"
            "• IT Support (WiFi, Portals, Passwords)"
        )
        self.info_box.insert("0.0", capabilities_text)
        self.info_box.configure(state="disabled")

        self.spacer = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.spacer.pack(expand=True, fill="both")

        self.clear_button = ctk.CTkButton(self.sidebar_frame, text="Clear Chat", command=self.clear_chat, font=ctk.CTkFont(family=MAIN_FONT, size=13))
        self.clear_button.pack(padx=20, pady=20, fill="x")

        # ==========================================
        # RIGHT PANE (Chat Area)
        # ==========================================
        self.main_chat_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#181818") # Very dark background for contrast
        self.main_chat_frame.grid(row=0, column=1, sticky="nsew")
        
        self.main_chat_frame.grid_rowconfigure(0, weight=1) 
        self.main_chat_frame.grid_rowconfigure(1, weight=0) 
        self.main_chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_history = ctk.CTkScrollableFrame(self.main_chat_frame, fg_color="transparent")
        self.chat_history.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)

        self.input_frame = ctk.CTkFrame(self.main_chat_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.input_frame.grid_columnconfigure(0, weight=1) 

        self.entry_box = ctk.CTkEntry(self.input_frame, placeholder_text="Type your question here...", font=ctk.CTkFont(family=MAIN_FONT, size=14))
        self.entry_box.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=8)
        self.entry_box.bind("<Return>", self.send_message_event)

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=80, font=ctk.CTkFont(family=MAIN_FONT, size=14, weight="bold"), command=self.send_message_event)
        self.send_button.grid(row=0, column=1, ipady=3)

        # Trigger the initial model load
        self.change_model_event(self.model_selector.get())
        self.typing_label = None # Placeholder for our typing indicator


    # ==========================================
    # EVENT FUNCTIONS 
    # ==========================================
    def change_model_event(self, selected_model):
        self.status_label.configure(text=f"Status: Loading {selected_model}...", text_color="orange")
        self.update()

        success = self.bot_engine.load_model(selected_model)

        if success:
            self.status_label.configure(text=f"Status: {selected_model} Active", text_color="green")
            self.display_message(f"System switched to {selected_model} model. How can I help you?", "bot")
        else:
            self.status_label.configure(text=f"Status: Error loading {selected_model}", text_color="red")
            self.display_message(f"Error loading the {selected_model} model. Check your terminal for details.", "bot")

    def clear_chat(self):
        for widget in self.chat_history.winfo_children():
            widget.destroy()
            
        print("Chat cleared!")

    def display_message(self, text, sender, metadata=None):
        row_frame = ctk.CTkFrame(self.chat_history, fg_color="transparent")
        row_frame.pack(fill="x", pady=5, padx=10)
        
        if sender == "user":
            bubble_color = "#094C4D"
            text_color = "white"
            alignment = "e" 
        else:
            bubble_color = "#2d2d2d" 
            text_color = "#e0e0e0"   
            alignment = "w"
            
        bubble = ctk.CTkFrame(row_frame, fg_color=bubble_color, corner_radius=15)
        bubble.pack(anchor=alignment)
        
        msg_label = ctk.CTkLabel(bubble, text=text, text_color=text_color, font=ctk.CTkFont(family=MAIN_FONT, size=14), wraplength=450, justify="left")
        msg_label.pack(padx=18, pady=12)
        
        if metadata and sender == "bot":
            meta_label = ctk.CTkLabel(row_frame, text=metadata, text_color="#7a7a7a", font=ctk.CTkFont(family=MAIN_FONT, size=11, slant="italic"))
            meta_label.pack(anchor=alignment, padx=10, pady=(4, 0))
            
        self.chat_history._parent_canvas.yview_moveto(1.0)

    def send_message_event(self, event=None):
        user_text = self.entry_box.get()
        if not user_text.strip():
            return
        
        self.entry_box.delete(0, 'end') 
        self.display_message(user_text, "user")
        
        # Disable inputs to prevent spamming
        self.entry_box.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        # Display "Bot is typing..." indicator
        self.typing_label = ctk.CTkLabel(self.chat_history, text="Bot is typing...", text_color="#7a7a7a", font=ctk.CTkFont(family=MAIN_FONT, size=12, slant="italic"))
        self.typing_label.pack(anchor="w", padx=15, pady=5)
        self.chat_history._parent_canvas.yview_moveto(1.0)
        
        # Wait 300ms so the UI can draw the disabled states and typing text before blocking the thread
        self.after(300, lambda: self.process_response(user_text))

    def process_response(self, user_text):
        # Destroy the typing indicator
        if self.typing_label:
            self.typing_label.destroy()
            
        # Run the model inference
        reply, intent, confidence = self.bot_engine.get_response(user_text)
        meta_string = f"Intent: {intent} | Confidence: {confidence:.2%}" if intent != "NO_INPUT" else None
        
        self.display_message(reply, "bot", metadata=meta_string)
        
        # Re-enable inputs
        self.entry_box.configure(state="normal")
        self.send_button.configure(state="normal")
        self.entry_box.focus() # Automatically put the typing cursor back in the box

if __name__ == "__main__":
    app = Chatbot()
    app.mainloop()