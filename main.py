from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crew.agents import Agent
from crew.tasks import Task
from crew.process import Process
from gpt4all import GPT4All

app = FastAPI()

class Query(BaseModel):
    text: str

# Inizializza il modello
model = GPT4All("gpt4-x-alpaca-native")

# Definizione degli agenti
manager_agent = Agent(
    name="Manager",
    goal="Analizzare le richieste e indirizzarle all'agente appropriato",
    backstory="Sono l'agente manager che coordina le richieste",
    model=model,
    tools=[])

general_agent = Agent(
    name="Agente1",
    goal="Rispondere a domande generali",
    backstory="Sono l'agente principale per domande generali",
    model=model,
    tools=[])

bill_expert_agent = Agent(
    name="Agente2",
    goal="Analizzare bollette e calcolare risparmi", 
    backstory="Sono l'esperto in analisi bollette e calcolo risparmi",
    model=model,
    tools=[])

def is_bill_related(query: str) -> bool:
    """Determina se la query è relativa a bollette."""
    bill_keywords = ["bolletta", "fattura", "consumo", "risparmio", "energia", "gas", "luce"]
    return any(keyword in query.lower() for keyword in bill_keywords)

@app.post("/query")
async def process_query(query: Query):
    try:
        # Crea il processo con gli agenti
        process = Process(
            agents=[manager_agent, general_agent, bill_expert_agent]
        )
        
        # Determina quale agente deve rispondere
        if is_bill_related(query.text):
            task = Task(
                description=query.text,
                agent=bill_expert_agent
            )
        else:
            task = Task(
                description=query.text,
                agent=general_agent
            )
        
        # Esegui il task
        result = process.execute([task])
        
        return {"response": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)