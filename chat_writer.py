import json
import uuid
from datetime import datetime
import os

class ConversationLogger:
    def __init__(self, filename="conversation_logs.jsonl"):
        self.filename = filename
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create file if it doesn't exist"""
        if not os.path.exists(self.filename):
            open(self.filename, 'w').close()

    def save_conversation(self, user_query, assistant_response):
        """
        Save a conversation pair in JSONL format
        
        Args:
            user_query (str): User's input/question
            assistant_response (str): Assistant's response
        """
        # Generate unique identifiers
        message_tree_id = str(uuid.uuid4())
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat() + '+00:00'
        
        # Prepare user message
        user_message = {
            "message_id": user_message_id,
            "user_id": str(uuid.uuid4()),
            "created_date": timestamp,
            "text": user_query,
            "role": "prompter",
            "lang": "en",
            "message_tree_id": message_tree_id,
            "tree_state": "ready_for_export"
        }
        
        # Prepare assistant message
        assistant_message = {
            "message_id": assistant_message_id,
            "parent_id": user_message_id,
            "user_id": str(uuid.uuid4()),
            "created_date": timestamp,
            "text": assistant_response,
            "role": "assistant",
            "lang": "en",
            "message_tree_id": message_tree_id,
            "tree_state": "ready_for_export"
        }
        
        # Write to JSONL file
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(user_message) + '\n')
            f.write(json.dumps(assistant_message) + '\n')

    def load_conversations(self):
        """Load conversations from JSONL file"""
        conversations = []
        current_conversation = {}
        
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                message = json.loads(line.strip())
                
                # Group messages by message_tree_id
                if 'message_tree_id' in message:
                    # If this is a new conversation tree, start a new entry
                    if not current_conversation or message['message_tree_id'] != current_conversation.get('message_tree_id'):
                        if current_conversation:
                            conversations.append(current_conversation)
                        current_conversation = {
                            'message_tree_id': message['message_tree_id'],
                            'created_date': message['created_date']
                        }
                    
                    # Add user or assistant message
                    if message['role'] == 'prompter':
                        current_conversation['user'] = message['text']
                    elif message['role'] == 'assistant':
                        current_conversation['assistant'] = message['text']
        
        # Add the last conversation
        if current_conversation:
            conversations.append(current_conversation)
        
        return conversations

# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = ConversationLogger()
    
    # Save a conversation pair
    logger.save_conversation(
        "Can you write a short introduction about the relevance of the term \"monopsony\" in economics?",
        "\"Monopsony\" refers to a market structure where there is only one buyer for a particular good or service. In economics, this term is particularly relevant in the labor market, where a monopsony employer has significant power over the wages and working conditions of their employees."
    )
    
    # Load and display conversations
    conversations = logger.load_conversations()
    print("\nStored Conversations:")
    for idx, convo in enumerate(conversations, 1):
        print(f"\nConversation {idx}:")
        print(f"Timestamp: {convo.get('created_date', 'N/A')}")
        print(f"User: {convo.get('user', 'N/A')}")
        print(f"Assistant: {convo.get('assistant', 'N/A')}")
