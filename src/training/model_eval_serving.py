import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import logging

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float

class ModelServer:
    def __init__(self, model_path='./intent_classification_model'):
        """
        Initialize model server
        
        Args:
            model_path (str): Path to saved model
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # Predefined intent labels (should match training)
        self.intent_labels = [
            'information_request', 
            'explanation', 
            'comparison', 
            'problem_solving', 
            'creative', 
            'opinion'
        ]

    def predict(self, text: str):
        """
        Predict intent for given text
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Prediction with intent and confidence
        """
        # Tokenize input
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get top prediction
            top_prob, top_class = probabilities.max(1)
            
            return {
                'intent': self.intent_labels[top_class.item()],
                'confidence': top_prob.item()
            }

# Create FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="Predict conversation intent",
    version="1.0.0"
)

# Initialize model
model_server = ModelServer()

@app.post("/predict", response_model=PredictionResponse)
def predict_intent(request: PredictionRequest):
    """
    Predict intent for given text
    
    Args:
        request (PredictionRequest): Input text
    
    Returns:
        PredictionResponse: Predicted intent and confidence
    """
    try:
        prediction = model_server.predict(request.text)
        return PredictionResponse(**prediction)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Healthcheck endpoint
@app.get("/health")
def health_check():
    """
    Simple healthcheck endpoint
    
    Returns:
        dict: Server status
    """
    return {"status": "healthy"}

# Run server
def start_server(host="0.0.0.0", port=8000):
    """
    Start FastAPI server
    
    Args:
        host (str): Server host
        port (int): Server port
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
