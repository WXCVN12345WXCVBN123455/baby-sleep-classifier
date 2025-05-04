# Baby Sleep Position Classifier

This project uses deep learning to classify baby sleep positions, specifically detecting if a baby is sleeping on their belly or not.

## Features

- Deep learning model for baby sleep position classification
- FastAPI backend with REST API
- Web interface for easy image upload and prediction
- Deployed on Render for easy access

## API Endpoints

- `GET /`: Web interface for image upload
- `POST /predict`: API endpoint for image classification
- `GET /docs`: API documentation (Swagger UI)

## Local Development

1. Create a virtual environment:
```bash
python -m venv baby-env
```

2. Activate the virtual environment:
```bash
# Windows
.\baby-env\Scripts\activate
# Linux/Mac
source baby-env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
python api.py
```

5. Access the web interface at `http://localhost:8000`

## Deployment

This project is configured for deployment on Render. The deployment process is automated through the `render.yaml` configuration file.

## Model

The model is a CNN-based classifier trained on baby sleep position images. It outputs a binary classification (belly/not belly) with confidence scores.

## License

MIT License 