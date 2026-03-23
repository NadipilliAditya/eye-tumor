---
description: Run the full project stack (backend + frontend)
---

// turbo-all

1. Start the React frontend development server:
```bash
cd frontend; npm run dev
```

2. Start the Flask backend server:
```bash
cd backend; .\venv\Scripts\Activate.ps1; python app.py

### Running with Docker (Single Command)

If you have Docker installed, you can run the entire project with:

```bash
docker-compose up --build
```

This will automatically set up the backend and frontend in isolated containers and connect them.
