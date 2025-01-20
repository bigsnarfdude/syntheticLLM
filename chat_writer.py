import csv
from datetime import datetime
import os
import uuid

class ConversationLogger:
    def __init__(self, filename="conversation_logs.csv"):
        self.filename = filename
        self._ensure_file_header()

    def _ensure_file_header(self):
        """Create file with headers if it doesn't exist"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'ID', 
                    'Timestamp', 
                    'Role',
                    'Content'
                ])
                writer.writeheader()

    def save_conversation(self, user_query, assistant_response):
        """
        Save a conversation pair with shared ID and timestamp
        Args:
            user_query (str): User's input/question
            assistant_response (str): Assistant's response
        """
        conv_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'ID', 
                'Timestamp', 
                'Role',
                'Content'
            ])
            
            # Write user message
            writer.writerow({
                'ID': conv_id,
                'Timestamp': timestamp,
                'Role': 'user',
                'Content': user_query
            })
            
            # Write assistant message
            writer.writerow({
                'ID': conv_id,
                'Timestamp': timestamp,
                'Role': 'assistant',
                'Content': assistant_response
            })

    def load_conversations(self):
        """Load all conversations grouped by ID"""
        conversations = {}
        with open(self.filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                conv_id = row['ID']
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        'timestamp': row['Timestamp'],
                        'user': '',
                        'assistant': ''
                    }
                
                if row['Role'] == 'user':
                    conversations[conv_id]['user'] = row['Content']
                elif row['Role'] == 'assistant':
                    conversations[conv_id]['assistant'] = row['Content']
        
        return list(conversations.values())

# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = ConversationLogger()
    
    # Save a conversation pair
    logger.save_conversation(
        "What's the capital of France?",
        "The capital of France is Paris."
    )
    
    # Save another conversation
    logger.save_conversation(
        "Explain quantum computing basics",
        "Quantum computing uses qubits to represent 0 and 1 simultaneously."
    )
    
    # Load and display conversations
    conversations = logger.load_conversations()
    print("\nStored Conversations:")
    for idx, convo in enumerate(conversations, 1):
        print(f"\nConversation {idx}:")
        print(f"ID: {convo['timestamp']}")
        print(f"User: {convo['user']}")
        print(f"Assistant: {convo['assistant']}")