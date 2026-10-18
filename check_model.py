
import joblib

model = joblib.load("fin_shield_model_new.joblib")

print("Model loaded successfully!")
print("Model type:", type(model).__name__)
print("Pipeline steps:", list(model.named_steps.keys()))
