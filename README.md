# TorchWeaver
*A visual, no-code neural network builder that generates executable PyTorch code.*

TorchWeaver is a client-server web application that allows users to **design neural network architectures visually** using a drag-and-drop interface. It lowers the barrier for experimenting with custom architectures by turning graphical designs into clean, runnable PyTorch code.


## Project Overview
Neural networks are powerful but designing architectures often requires both domain knowledge and programming skills. TorchWeaver aims to reduce this gap by making neural network design:

- **Visual** — build models interactively
- **Intuitive** — configure layers without writing code
- **Reliable** — real-time validation and error checking
- **Production-ready** — export clean PyTorch code


## Key Features
### **Frontend (React)**
- Drag-and-drop canvas for building neural networks  
- Editable parameters for each layer  
- Real-time validation and error checking  
- Export neural network architectures as excutable Pytorch code
- User authentication: signup, login, logout
- Dashboard for managing saved architectures  

### **Backend (Flask & SQLite)**
- Database storage of network designs  
- Rest API for saving, retrieving, deleting, updating, exporting architectures  
- PyTorch code generation endpoint  
- SQLite database for persistent storage  


## Getting Started
1. Clone the repository
```bash 
git clone https://github.com/unsw-cse-comp99-3900/capstone-project-25t3-3900-f09b-cherry.git
cd capstone-project-25t3-3900-f09b-cherry
```
2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
3. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python api.py
```
4. Open http://localhost:3000 with your browser.
