class ConversationMemory:

    def __init__(self):
        self.history = []

    def add_user_message(self, message):
        if self.history and "assistant" not in self.history[-1]:
            self.history.pop()
        self.history.append({"user": message})

    def add_assistant_message(self, message):
        if self.history and "assistant" not in self.history[-1]:
            self.history[-1]["assistant"] = message
        else:
            self.history.append({"assistant": message})

    def load(self):
        return [msg for msg in self.history if "user" in msg and "assistant" in msg]

    def get_last_n_messages(self, n):
        full_history = self.load()
        return full_history[-n:] if len(full_history) >= n else full_history

    def clear(self):
        self.history = []