from typing import Dict


class NodeTemplates:
    @staticmethod
    def get_message_node() -> Dict:
        return {
            "messages": [{"role": "system", "content": ""}],
            "functions": [],
            "pre_actions": None,
            "post_actions": None,
        }

    @staticmethod
    def get_function_node() -> Dict:
        return {
            "messages": [{"role": "system", "content": ""}],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "pre_actions": None,
            "post_actions": None,
        }

    @staticmethod
    def get_terminal_node() -> Dict:
        return {
            "messages": [{"role": "system", "content": "End of conversation."}],
            "functions": [],
            "pre_actions": [{"type": "tts_say", "text": "Goodbye!"}],
            "post_actions": [{"type": "end_conversation"}],
        }
